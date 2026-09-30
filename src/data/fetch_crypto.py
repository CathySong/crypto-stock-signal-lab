"""
fetch_crypto.py
================
Download daily price/market-cap/volume history for the crypto universe
defined in config/assets.yaml, using CoinGecko's public API.

Source: CoinGecko API (free public tier)
Docs:   https://www.coingecko.com/en/api
Endpoint used: GET /coins/{id}/market_chart
    https://www.coingecko.com/en/api/documentation -> Coins -> "Coin Data by ID"
    Returns parallel arrays of [timestamp, value] for prices, market_caps,
    total_volumes at daily granularity when `days` > 1 (interval=daily).

Free tier rate limits are informal and low (roughly 10-30 req/min per IP);
this script sleeps between calls and retries once on 429.

Output: one CSV per coin at data/raw/crypto/{SYMBOL}.csv with columns:
    date, price_usd, market_cap_usd, volume_usd

Usage:
    python -m src.data.fetch_crypto
    python -m src.data.fetch_crypto --days 365
    python -m src.data.fetch_crypto --coins bitcoin,ethereum
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone

import pandas as pd
import requests

from src.data.utils import DATA_RAW_DIR, ensure_dir, get_logger, load_asset_config

logger = get_logger("fetch_crypto")

CRYPTO_DIR = DATA_RAW_DIR / "crypto"
COINGECKO_BASE = "https://api.coingecko.com/api/v3"


def fetch_market_chart(
    session: requests.Session, cg_id: str, days: str, vs_currency: str = "usd", max_retries: int = 3
) -> dict:
    """
    GET /coins/{id}/market_chart?vs_currency=usd&days={days}
    `days` can be an integer string ("365") or "max". CoinGecko auto-selects
    daily granularity for days > 90 on the free tier.
    """
    url = f"{COINGECKO_BASE}/coins/{cg_id}/market_chart"
    params = {"vs_currency": vs_currency, "days": days}

    for attempt in range(1, max_retries + 1):
        resp = session.get(url, params=params, timeout=30)
        if resp.status_code == 429:
            wait = 15 * attempt
            logger.warning("Rate limited on %s (attempt %d/%d); sleeping %ds", cg_id, attempt, max_retries, wait)
            time.sleep(wait)
            continue
        resp.raise_for_status()
        return resp.json()

    raise RuntimeError(f"Exceeded retries fetching {cg_id} from CoinGecko (rate limited)")


def to_dataframe(payload: dict, symbol: str) -> pd.DataFrame:
    prices = payload.get("prices", [])
    caps = payload.get("market_caps", [])
    vols = payload.get("total_volumes", [])

    if not prices:
        return pd.DataFrame()

    df_price = pd.DataFrame(prices, columns=["ts_ms", "price_usd"])
    df_cap = pd.DataFrame(caps, columns=["ts_ms", "market_cap_usd"])
    df_vol = pd.DataFrame(vols, columns=["ts_ms", "volume_usd"])

    df = df_price.merge(df_cap, on="ts_ms", how="outer").merge(df_vol, on="ts_ms", how="outer")
    df["date"] = df["ts_ms"].apply(lambda ms: datetime.fromtimestamp(ms / 1000, tz=timezone.utc).date())
    df["symbol"] = symbol

    # CoinGecko's market_chart returns roughly-daily points but not aligned
    # to exact midnight; collapse to one row per calendar date (keep last).
    df = df.sort_values("ts_ms").groupby("date", as_index=False).last()
    return df[["date", "symbol", "price_usd", "market_cap_usd", "volume_usd"]]


def main() -> None:
    parser = argparse.ArgumentParser(description="Download crypto market data via CoinGecko.")
    parser.add_argument("--days", default="365", help="Lookback window: integer days or 'max' (default: 365)")
    parser.add_argument(
        "--coins",
        default=None,
        help="Comma-separated CoinGecko ids, e.g. bitcoin,ethereum. Defaults to config/assets.yaml.",
    )
    parser.add_argument("--vs-currency", default="usd", help="Quote currency (default: usd)")
    parser.add_argument("--sleep", type=float, default=2.0, help="Seconds between coins (free tier is rate-limited)")
    args = parser.parse_args()

    if args.coins:
        coin_list = [
            {"cg_id": c.strip(), "symbol": c.strip().upper()} for c in args.coins.split(",") if c.strip()
        ]
    else:
        cfg = load_asset_config()
        coin_list = cfg.get("crypto", [])

    if not coin_list:
        logger.error("No coins to fetch. Check config/assets.yaml or pass --coins.")
        sys.exit(1)

    ensure_dir(CRYPTO_DIR)
    session = requests.Session()
    session.headers.update({"User-Agent": "crypto-stock-signal-lab/0.1 (research use)"})

    ok, failed = [], []
    for i, coin in enumerate(coin_list):
        cg_id, symbol = coin["cg_id"], coin["symbol"]
        try:
            logger.info("Fetching %s (%s) days=%s", cg_id, symbol, args.days)
            payload = fetch_market_chart(session, cg_id, args.days, args.vs_currency)
            df = to_dataframe(payload, symbol)
            if df.empty:
                logger.warning("No data returned for %s", cg_id)
                failed.append(cg_id)
                continue
            out_path = CRYPTO_DIR / f"{symbol}.csv"
            df.to_csv(out_path, index=False)
            logger.info("Saved %d rows -> %s", len(df), out_path)
            ok.append(symbol)
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to fetch %s: %s", cg_id, exc)
            failed.append(cg_id)

        if i < len(coin_list) - 1:
            time.sleep(args.sleep)

    logger.info("Done. ok=%s failed=%s", ok, failed)


if __name__ == "__main__":
    main()

"""
fetch_stocks.py
================
Download daily OHLCV price history for the stock/ETF universe defined in
config/assets.yaml, using yfinance.

Source: yfinance (wraps Yahoo Finance's public endpoints)
Docs:   https://ranaroussi.github.io/yfinance/reference/api/yfinance.download.html

Output: one CSV per ticker at data/raw/stocks/{TICKER}.csv with columns:
    date, open, high, low, close, adj_close, volume

Usage:
    python -m src.data.fetch_stocks
    python -m src.data.fetch_stocks --period 5y --interval 1d
    python -m src.data.fetch_stocks --tickers AAPL,NVDA
"""
from __future__ import annotations

import argparse
import sys
import time

import pandas as pd

try:
    import yfinance as yf
except ImportError:  # pragma: no cover
    print(
        "yfinance is not installed. Run `pip install -r requirements.txt` first.",
        file=sys.stderr,
    )
    raise

from src.data.utils import DATA_RAW_DIR, ensure_dir, get_logger, load_asset_config

logger = get_logger("fetch_stocks")

STOCKS_DIR = DATA_RAW_DIR / "stocks"

# Supported yfinance intervals for reference (see yfinance.download docs).
# Intraday intervals (<1d) are limited by Yahoo to short lookback windows
# (e.g. 1m data only goes back ~7 days), so the MVP defaults to daily bars.
VALID_INTERVALS = {
    "1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h",
    "1d", "5d", "1wk", "1mo", "3mo",
}


def fetch_one(ticker: str, period: str, interval: str) -> pd.DataFrame:
    """Download one ticker's OHLCV history via yfinance.download()."""
    logger.info("Fetching %s (period=%s, interval=%s)", ticker, period, interval)
    df = yf.download(
        tickers=ticker,
        period=period,
        interval=interval,
        auto_adjust=False,   # keep raw Close + separate Adj Close column
        progress=False,
        threads=False,
    )

    if df is None or df.empty:
        logger.warning("No data returned for %s", ticker)
        return pd.DataFrame()

    # yfinance can return a MultiIndex column header even for a single
    # ticker depending on version; flatten defensively.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]

    df = df.reset_index()
    df.columns = [str(c).lower().replace(" ", "_") for c in df.columns]
    # normalize date column name across yfinance versions (Date vs Datetime)
    if "datetime" in df.columns and "date" not in df.columns:
        df = df.rename(columns={"datetime": "date"})

    keep = [c for c in ["date", "open", "high", "low", "close", "adj_close", "volume"] if c in df.columns]
    df = df[keep]
    df["ticker"] = ticker
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Download stock/ETF OHLCV data via yfinance.")
    parser.add_argument("--period", default="5y", help="yfinance period, e.g. 1y, 5y, max (default: 5y)")
    parser.add_argument("--interval", default="1d", help="yfinance interval (default: 1d)")
    parser.add_argument(
        "--tickers",
        default=None,
        help="Comma-separated ticker override, e.g. AAPL,NVDA. Defaults to config/assets.yaml.",
    )
    parser.add_argument("--sleep", type=float, default=1.0, help="Seconds to sleep between tickers")
    args = parser.parse_args()

    if args.interval not in VALID_INTERVALS:
        logger.warning("Interval '%s' is not in yfinance's documented list; proceeding anyway.", args.interval)

    if args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    else:
        cfg = load_asset_config()
        tickers = [s["ticker"] for s in cfg.get("stocks", [])]

    if not tickers:
        logger.error("No tickers to fetch. Check config/assets.yaml or pass --tickers.")
        sys.exit(1)

    ensure_dir(STOCKS_DIR)

    ok, failed = [], []
    for i, ticker in enumerate(tickers):
        try:
            df = fetch_one(ticker, args.period, args.interval)
            if df.empty:
                failed.append(ticker)
                continue
            out_path = STOCKS_DIR / f"{ticker}.csv"
            df.to_csv(out_path, index=False)
            logger.info("Saved %d rows -> %s", len(df), out_path)
            ok.append(ticker)
        except Exception as exc:  # noqa: BLE001 - log and continue with remaining tickers
            logger.error("Failed to fetch %s: %s", ticker, exc)
            failed.append(ticker)

        if i < len(tickers) - 1:
            time.sleep(args.sleep)

    logger.info("Done. ok=%s failed=%s", ok, failed)


if __name__ == "__main__":
    main()

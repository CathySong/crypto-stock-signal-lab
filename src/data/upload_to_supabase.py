"""
upload_to_supabase.py
=======================
Push locally-generated CSV/JSONL data into Supabase Postgres tables via
the REST API (PostgREST), using upsert semantics so re-running this script
after a fresh scrape or research run just updates rows instead of
duplicating them.

Requires SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY in .env (service role
key is required because these tables have Row Level Security enabled and
the anon/publishable key cannot write to them).

Tables expected to already exist (see sql/001_create_tables.sql, run once
via the Supabase SQL Editor before using this script):
    stock_prices(ticker, date, open, high, low, close, adj_close, volume)
    stock_fundamentals(ticker, concept, unit, start_date, end_date, val,
                        fy, fp, form, filed, frame)
    crypto_prices(symbol, date, price_usd, market_cap_usd, volume_usd)
    asset_scores(symbol, date, ...)
    backtest_results(symbol, strategy, start_date, end_date, ...)
    ml_walkforward_results(symbol, horizon_days, ...)

Usage:
    python -m src.data.upload_to_supabase
    python -m src.data.upload_to_supabase --only stocks
    python -m src.data.upload_to_supabase --only scores,backtests,ml
    python -m src.data.upload_to_supabase --only scores --dry-run
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from src.data.utils import DATA_PROCESSED_DIR, DATA_RAW_DIR, get_logger

logger = get_logger("upload_to_supabase")

STOCKS_DIR = DATA_RAW_DIR / "stocks"
FUNDAMENTALS_DIR = DATA_RAW_DIR / "fundamentals"
CRYPTO_DIR = DATA_RAW_DIR / "crypto"
SCORES_DIR = DATA_PROCESSED_DIR / "scores"
BACKTEST_DIR = DATA_PROCESSED_DIR / "backtests"
ML_DIR = DATA_PROCESSED_DIR / "ml"

BATCH_SIZE = 500  # rows per REST request; keeps payloads well under limits

TARGETS = {"stocks", "fundamentals", "crypto", "scores", "backtests", "ml"}


def get_client(env_path: Path | None = None) -> tuple[str, dict]:
    load_dotenv(env_path or (Path(__file__).resolve().parents[2] / ".env"))
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
    if not url or not key:
        logger.error(
            "Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY. "
            "Copy .env.example to .env and fill in real values."
        )
        sys.exit(1)
    headers = {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "Prefer": "resolution=merge-duplicates",
    }
    return url.rstrip("/"), headers


def clean_records(records: list[dict]) -> list[dict]:
    """Replace NaN/NaT/Infinity with None so the JSON payload is valid."""
    cleaned = []
    for row in records:
        clean_row = {}
        for k, v in row.items():
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                clean_row[k] = None
            elif pd.isna(v) if not isinstance(v, (list, dict)) else False:
                clean_row[k] = None
            else:
                clean_row[k] = v
        cleaned.append(clean_row)
    return cleaned


def dedupe_on_conflict(records: list[dict], on_conflict: str) -> list[dict]:
    """
    Keep only the last occurrence of each conflict-key tuple.

    Postgres' ON CONFLICT DO UPDATE raises "cannot affect row a second
    time" if a single INSERT statement contains duplicate rows on the
    conflict target -- which happens with SEC XBRL data (the same
    concept/period can appear across more than one filing). De-duplicating
    client-side, keeping the most recently seen row, avoids this.
    """
    keys = [k.strip() for k in on_conflict.split(",")]
    deduped: dict[tuple, dict] = {}
    for row in records:
        key = tuple(row.get(k) for k in keys)
        deduped[key] = row  # later rows overwrite earlier ones with same key
    return list(deduped.values())


def upsert_table(base_url: str, headers: dict, table: str, records: list[dict], on_conflict: str) -> int:
    """POST rows to PostgREST with upsert (merge-duplicates on conflict target)."""
    import requests

    if not records:
        return 0
    records = dedupe_on_conflict(records, on_conflict)
    total = 0
    url = f"{base_url}/rest/v1/{table}"
    params = {"on_conflict": on_conflict}
    for i in range(0, len(records), BATCH_SIZE):
        batch = clean_records(records[i : i + BATCH_SIZE])
        resp = requests.post(url, headers=headers, params=params, data=json.dumps(batch), timeout=60)
        if resp.status_code not in (200, 201, 204):
            logger.error("Upsert failed for %s (batch %d): %s %s", table, i // BATCH_SIZE, resp.status_code, resp.text[:500])
            resp.raise_for_status()
        total += len(batch)
        logger.info("  %s: upserted %d/%d rows", table, total, len(records))
    return total


def maybe_upsert(
    base_url: str | None,
    headers: dict | None,
    table: str,
    records: list[dict],
    on_conflict: str,
    dry_run: bool,
) -> int:
    if dry_run:
        deduped = dedupe_on_conflict(records, on_conflict)
        logger.info("Dry run: %s would upsert %d rows", table, len(deduped))
        return len(deduped)
    if base_url is None or headers is None:
        raise ValueError("Supabase client is required unless dry_run=True")
    return upsert_table(base_url, headers, table, records, on_conflict)


def upload_stocks(base_url: str | None, headers: dict | None, dry_run: bool = False) -> None:
    if not STOCKS_DIR.exists():
        return
    for csv_path in sorted(STOCKS_DIR.glob("*.csv")):
        ticker = csv_path.stem
        df = pd.read_csv(csv_path)
        if df.empty:
            continue
        df = df.rename(columns={"ticker": "ticker"})  # already named correctly
        records = df[["ticker", "date", "open", "high", "low", "close", "adj_close", "volume"]].to_dict("records")
        logger.info("Uploading %s stock_prices (%d rows)", ticker, len(records))
        maybe_upsert(base_url, headers, "stock_prices", records, on_conflict="ticker,date", dry_run=dry_run)


def upload_fundamentals(base_url: str | None, headers: dict | None, dry_run: bool = False) -> None:
    if not FUNDAMENTALS_DIR.exists():
        return
    for jsonl_path in sorted(FUNDAMENTALS_DIR.glob("*.jsonl")):
        ticker = jsonl_path.stem
        records = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                records.append(
                    {
                        "ticker": row.get("ticker"),
                        "concept": row.get("concept"),
                        "unit": row.get("unit"),
                        "start_date": row.get("start"),
                        "end_date": row.get("end"),
                        "val": row.get("val"),
                        "fy": row.get("fy"),
                        "fp": row.get("fp"),
                        "form": row.get("form"),
                        "filed": row.get("filed"),
                        "frame": row.get("frame"),
                    }
                )
        logger.info("Uploading %s stock_fundamentals (%d rows)", ticker, len(records))
        maybe_upsert(
            base_url,
            headers,
            "stock_fundamentals",
            records,
            on_conflict="ticker,concept,unit,end_date,form,fy,fp",
            dry_run=dry_run,
        )


def upload_crypto(base_url: str | None, headers: dict | None, dry_run: bool = False) -> None:
    if not CRYPTO_DIR.exists():
        return
    for csv_path in sorted(CRYPTO_DIR.glob("*.csv")):
        symbol = csv_path.stem
        df = pd.read_csv(csv_path)
        if df.empty:
            continue
        records = df[["symbol", "date", "price_usd", "market_cap_usd", "volume_usd"]].to_dict("records")
        logger.info("Uploading %s crypto_prices (%d rows)", symbol, len(records))
        maybe_upsert(base_url, headers, "crypto_prices", records, on_conflict="symbol,date", dry_run=dry_run)


def infer_price_col(df: pd.DataFrame) -> str | None:
    for col in ("adj_close", "price_usd", "close"):
        if col in df.columns:
            return col
    return None


def upload_scores(base_url: str | None, headers: dict | None, dry_run: bool = False) -> None:
    if not SCORES_DIR.exists():
        return
    columns = [
        "symbol",
        "date",
        "price",
        "timing_score",
        "timing_label",
        "value_score",
        "value_label",
        "risk_score",
        "risk_label",
        "final_score",
        "stop_distance_pct",
        "stop_loss_price",
        "suggested_pct_of_account",
        "suggested_dollar_amount",
        "suggested_units",
    ]
    for csv_path in sorted(SCORES_DIR.glob("*.csv")):
        symbol = csv_path.stem
        df = pd.read_csv(csv_path)
        if df.empty:
            continue
        price_col = infer_price_col(df)
        df["price"] = df[price_col] if price_col else None
        records = df[columns].to_dict("records")
        logger.info("Uploading %s asset_scores (%d rows)", symbol, len(records))
        maybe_upsert(base_url, headers, "asset_scores", records, on_conflict="symbol,date", dry_run=dry_run)


def upload_backtest_results(base_url: str | None, headers: dict | None, dry_run: bool = False) -> None:
    path = BACKTEST_DIR / "summary.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    if df.empty:
        return
    records = df.to_dict("records")
    logger.info("Uploading backtest_results (%d rows)", len(records))
    maybe_upsert(
        base_url,
        headers,
        "backtest_results",
        records,
        on_conflict="symbol,strategy,start_date,end_date,fee_bps,slippage_bps",
        dry_run=dry_run,
    )


def upload_ml_results(base_url: str | None, headers: dict | None, dry_run: bool = False) -> None:
    path = ML_DIR / "walkforward_summary.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    if df.empty:
        return
    records = df.to_dict("records")
    logger.info("Uploading ml_walkforward_results (%d rows)", len(records))
    maybe_upsert(
        base_url,
        headers,
        "ml_walkforward_results",
        records,
        on_conflict="symbol,horizon_days",
        dry_run=dry_run,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload local raw and processed data files into Supabase Postgres tables.")
    parser.add_argument(
        "--only",
        default=None,
        help="Comma-separated subset: stocks,fundamentals,crypto,scores,backtests,ml (default: all)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate local files and print row counts without uploading.")
    args = parser.parse_args()

    targets = set(TARGETS)
    if args.only:
        targets = {t.strip() for t in args.only.split(",") if t.strip()}
    unknown = targets - TARGETS
    if unknown:
        logger.error("Unknown upload target(s): %s", ", ".join(sorted(unknown)))
        sys.exit(1)

    base_url, headers = (None, None) if args.dry_run else get_client()

    if "stocks" in targets:
        upload_stocks(base_url, headers, dry_run=args.dry_run)
    if "fundamentals" in targets:
        upload_fundamentals(base_url, headers, dry_run=args.dry_run)
    if "crypto" in targets:
        upload_crypto(base_url, headers, dry_run=args.dry_run)
    if "scores" in targets:
        upload_scores(base_url, headers, dry_run=args.dry_run)
    if "backtests" in targets:
        upload_backtest_results(base_url, headers, dry_run=args.dry_run)
    if "ml" in targets:
        upload_ml_results(base_url, headers, dry_run=args.dry_run)

    logger.info("Done.")


if __name__ == "__main__":
    main()

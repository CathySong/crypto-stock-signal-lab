"""
upload_to_supabase.py
=======================
Push locally-fetched CSV/JSONL data (data/raw/*) into Supabase Postgres
tables via the REST API (PostgREST), using upsert semantics so re-running
this script after a fresh scrape just updates rows instead of duplicating
them.

Requires SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY in .env (service role
key is required because these tables have Row Level Security enabled and
the anon/publishable key cannot write to them).

Tables expected to already exist (see sql/001_create_tables.sql, run once
via the Supabase SQL Editor before using this script):
    stock_prices(ticker, date, open, high, low, close, adj_close, volume)
    stock_fundamentals(ticker, concept, unit, start_date, end_date, val,
                        fy, fp, form, filed, frame)
    crypto_prices(symbol, date, price_usd, market_cap_usd, volume_usd)

Usage:
    python -m src.data.upload_to_supabase
    python -m src.data.upload_to_supabase --only stocks
    python -m src.data.upload_to_supabase --only crypto,fundamentals
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

from src.data.utils import DATA_RAW_DIR, get_logger

logger = get_logger("upload_to_supabase")

STOCKS_DIR = DATA_RAW_DIR / "stocks"
FUNDAMENTALS_DIR = DATA_RAW_DIR / "fundamentals"
CRYPTO_DIR = DATA_RAW_DIR / "crypto"

BATCH_SIZE = 500  # rows per REST request; keeps payloads well under limits


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


def upsert_table(base_url: str, headers: dict, table: str, records: list[dict], on_conflict: str) -> int:
    """POST rows to PostgREST with upsert (merge-duplicates on conflict target)."""
    if not records:
        return 0
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


def upload_stocks(base_url: str, headers: dict) -> None:
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
        upsert_table(base_url, headers, "stock_prices", records, on_conflict="ticker,date")


def upload_fundamentals(base_url: str, headers: dict) -> None:
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
        upsert_table(
            base_url,
            headers,
            "stock_fundamentals",
            records,
            on_conflict="ticker,concept,unit,end_date,form,fy,fp",
        )


def upload_crypto(base_url: str, headers: dict) -> None:
    if not CRYPTO_DIR.exists():
        return
    for csv_path in sorted(CRYPTO_DIR.glob("*.csv")):
        symbol = csv_path.stem
        df = pd.read_csv(csv_path)
        if df.empty:
            continue
        records = df[["symbol", "date", "price_usd", "market_cap_usd", "volume_usd"]].to_dict("records")
        logger.info("Uploading %s crypto_prices (%d rows)", symbol, len(records))
        upsert_table(base_url, headers, "crypto_prices", records, on_conflict="symbol,date")


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload local raw data files into Supabase Postgres tables.")
    parser.add_argument(
        "--only",
        default=None,
        help="Comma-separated subset to upload: stocks,fundamentals,crypto (default: all)",
    )
    args = parser.parse_args()

    targets = {"stocks", "fundamentals", "crypto"}
    if args.only:
        targets = {t.strip() for t in args.only.split(",") if t.strip()}

    base_url, headers = get_client()

    if "stocks" in targets:
        upload_stocks(base_url, headers)
    if "fundamentals" in targets:
        upload_fundamentals(base_url, headers)
    if "crypto" in targets:
        upload_crypto(base_url, headers)

    logger.info("Done.")


if __name__ == "__main__":
    main()

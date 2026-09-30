"""
fetch_fundamentals.py
======================
Pull US-GAAP fundamentals (revenue, net income, EPS, assets, liabilities, ...)
for the stock universe in config/assets.yaml, using SEC EDGAR's public
companyfacts API.

Source: SEC EDGAR "companyfacts" JSON API
Docs:   https://www.sec.gov/search-filings/edgar-application-programming-interfaces
        (company facts endpoint: https://data.sec.gov/api/xbrl/companyfacts/CIK##########.json)

SEC requires a descriptive User-Agent identifying the requester
(name + contact email) on every request, and asks callers to stay under
~10 requests/second. See "Developer Resources" on sec.gov/edgar.

Output: one JSON-lines file per ticker at
    data/raw/fundamentals/{TICKER}.jsonl
with one row per (concept, fiscal period, unit), e.g.:
    {"ticker": "AAPL", "concept": "NetIncomeLoss", "unit": "USD",
     "end": "2024-09-28", "val": 93736000000, "fy": 2024, "fp": "FY",
     "form": "10-K", "filed": "2024-11-01"}

Usage:
    SEC_USER_AGENT="Cathy Doe cathy@example.com" python -m src.data.fetch_fundamentals
    python -m src.data.fetch_fundamentals --tickers AAPL,NVDA
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import requests

from src.data.utils import DATA_RAW_DIR, ensure_dir, get_logger, load_asset_config

logger = get_logger("fetch_fundamentals")

FUNDAMENTALS_DIR = DATA_RAW_DIR / "fundamentals"

SEC_TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_COMPANYFACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"

# SEC explicitly asks for a real User-Agent with contact info. Override via
# env var SEC_USER_AGENT or --user-agent before running for real.
DEFAULT_USER_AGENT = "crypto-stock-signal-lab research-use contact@example.com"


def get_user_agent(cli_value: str | None) -> str:
    ua = cli_value or os.environ.get("SEC_USER_AGENT") or DEFAULT_USER_AGENT
    if ua == DEFAULT_USER_AGENT:
        logger.warning(
            "Using placeholder SEC User-Agent. Set SEC_USER_AGENT env var or --user-agent "
            "to 'Your Name your@email.com' before real use (SEC EDGAR API policy)."
        )
    return ua


def build_ticker_to_cik(session: requests.Session) -> dict[str, int]:
    """Download SEC's ticker -> CIK mapping (small JSON, refreshed by SEC regularly)."""
    resp = session.get(SEC_TICKER_MAP_URL, timeout=30)
    resp.raise_for_status()
    raw = resp.json()  # {"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}, ...}
    return {row["ticker"].upper(): int(row["cik_str"]) for row in raw.values()}


def fetch_companyfacts(session: requests.Session, cik: int) -> dict:
    url = SEC_COMPANYFACTS_URL.format(cik=cik)
    resp = session.get(url, timeout=30)
    resp.raise_for_status()
    return resp.json()


def extract_concepts(facts_json: dict, ticker: str, concepts: list[str]) -> list[dict]:
    """Flatten companyfacts JSON into rows for the requested us-gaap concepts."""
    rows: list[dict] = []
    us_gaap = facts_json.get("facts", {}).get("us-gaap", {})
    for concept in concepts:
        concept_data = us_gaap.get(concept)
        if not concept_data:
            continue
        for unit, entries in concept_data.get("units", {}).items():
            for entry in entries:
                rows.append(
                    {
                        "ticker": ticker,
                        "concept": concept,
                        "unit": unit,
                        "start": entry.get("start"),
                        "end": entry.get("end"),
                        "val": entry.get("val"),
                        "fy": entry.get("fy"),
                        "fp": entry.get("fp"),
                        "form": entry.get("form"),
                        "filed": entry.get("filed"),
                        "frame": entry.get("frame"),
                    }
                )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Download SEC EDGAR companyfacts fundamentals.")
    parser.add_argument(
        "--tickers",
        default=None,
        help="Comma-separated tickers, e.g. AAPL,NVDA. Defaults to has_fundamentals=true entries in config/assets.yaml.",
    )
    parser.add_argument("--user-agent", default=None, help="SEC User-Agent string: 'Name email@domain.com'")
    parser.add_argument("--sleep", type=float, default=0.3, help="Seconds between SEC requests (SEC asks <=10 req/s)")
    args = parser.parse_args()

    user_agent = get_user_agent(args.user_agent)
    session = requests.Session()
    session.headers.update({"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"})

    cfg = load_asset_config()
    concepts = cfg.get("sec_concepts", [])

    if args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    else:
        tickers = [s["ticker"] for s in cfg.get("stocks", []) if s.get("has_fundamentals")]

    if not tickers:
        logger.error("No fundamentals-eligible tickers found (check config/assets.yaml has_fundamentals flags).")
        sys.exit(1)

    logger.info("Building SEC ticker->CIK map...")
    ticker_to_cik = build_ticker_to_cik(session)
    time.sleep(args.sleep)

    ensure_dir(FUNDAMENTALS_DIR)

    ok, failed, skipped = [], [], []
    for i, ticker in enumerate(tickers):
        cik = ticker_to_cik.get(ticker)
        if cik is None:
            logger.warning("No CIK found for %s in SEC ticker map; skipping.", ticker)
            skipped.append(ticker)
            continue
        try:
            logger.info("Fetching companyfacts for %s (CIK %010d)", ticker, cik)
            facts = fetch_companyfacts(session, cik)
            rows = extract_concepts(facts, ticker, concepts)
            out_path = FUNDAMENTALS_DIR / f"{ticker}.jsonl"
            with open(out_path, "w", encoding="utf-8") as f:
                for row in rows:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            logger.info("Saved %d fact rows -> %s", len(rows), out_path)
            ok.append(ticker)
        except requests.HTTPError as exc:
            logger.error("HTTP error for %s: %s", ticker, exc)
            failed.append(ticker)
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to fetch fundamentals for %s: %s", ticker, exc)
            failed.append(ticker)

        if i < len(tickers) - 1:
            time.sleep(args.sleep)

    logger.info("Done. ok=%s failed=%s skipped=%s", ok, failed, skipped)


if __name__ == "__main__":
    main()

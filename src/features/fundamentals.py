"""
fundamentals.py
=================
Derive valuation ratios (P/E, P/B, FCF yield, revenue growth, margins)
from the raw SEC XBRL facts (src/data/fetch_fundamentals.py) combined
with daily stock prices (src/data/fetch_stocks.py).

These are the building blocks for the project's "Value Score" — see
README.md for the overall scoring design.

Approach:
    1. Pivot the long-format fundamentals JSONL (one row per concept per
       period) into one row per fiscal period with a column per concept.
    2. Prefer 10-K (annual) and 10-Q (quarterly) filings; use the most
       recently *filed* value when a concept has multiple entries for
       the same fiscal period (covers amendments/restatements).
    3. Compute trailing-twelve-month (TTM) net income and revenue by
       summing the last 4 quarterly (10-Q-style) values, falling back to
       the latest annual (10-K) figure when quarterly history is thin.
    4. Join against daily close price to get point-in-time P/E and P/B,
       using shares outstanding to get per-share book value and EPS.

This is intentionally a first pass: good enough to rank assets and build
a baseline Value Score, not a fully GAAP-correct TTM engine. Known
simplifications are called out inline.

Output: one CSV per ticker at data/processed/fundamentals/{TICKER}.csv
with one row per date (daily), carrying forward the latest known
fundamentals until the next filing, joined with daily close price and
derived ratios.

Usage:
    python -m src.features.fundamentals
    python -m src.features.fundamentals --tickers AAPL,NVDA
"""
from __future__ import annotations

import argparse
import json
import sys

import pandas as pd

from src.data.utils import DATA_RAW_DIR, PROJECT_ROOT, ensure_dir, get_logger, load_asset_config

logger = get_logger("features.fundamentals")

FUNDAMENTALS_RAW_DIR = DATA_RAW_DIR / "fundamentals"
STOCKS_RAW_DIR = DATA_RAW_DIR / "stocks"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "fundamentals"

# Concepts we need for the ratios below. Mapped to short internal names.
CONCEPT_MAP = {
    "NetIncomeLoss": "net_income",
    "Revenues": "revenue",
    "RevenueFromContractWithCustomerExcludingAssessedTax": "revenue_alt",
    "Assets": "assets",
    "Liabilities": "liabilities",
    "StockholdersEquity": "stockholders_equity",
    "OperatingIncomeLoss": "operating_income",
    "CashAndCashEquivalentsAtCarryingValue": "cash",
    "ResearchAndDevelopmentExpense": "rd_expense",
    "NetCashProvidedByUsedInOperatingActivities": "operating_cash_flow",
    "PaymentsToAcquirePropertyPlantAndEquipment": "capex",
    "EarningsPerShareDiluted": "eps_diluted",
    "EarningsPerShareBasic": "eps_basic",
    "CommonStockSharesOutstanding": "shares_outstanding",
    "WeightedAverageNumberOfDilutedSharesOutstanding": "diluted_shares_wavg",
    "WeightedAverageNumberOfSharesOutstandingBasic": "basic_shares_wavg",
}


def load_fundamentals_long(jsonl_path) -> pd.DataFrame:
    rows = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["end"] = pd.to_datetime(df["end"])
    df["filed"] = pd.to_datetime(df["filed"])
    return df


def pivot_quarterly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build one row per (fiscal period end date, form) with a column per
    mapped concept, keeping only concepts we care about and, for
    duplicate (concept, end, form) combinations, the most recently filed
    value.
    """
    df = df[df["concept"].isin(CONCEPT_MAP)].copy()
    if df.empty:
        return df

    df["short_name"] = df["concept"].map(CONCEPT_MAP)
    df = df.sort_values("filed")
    # Keep the latest-filed value per (short_name, end, form) to handle
    # amendments/restatements.
    df = df.drop_duplicates(subset=["short_name", "end", "form"], keep="last")

    pivoted = df.pivot_table(
        index=["end", "form"], columns="short_name", values="val", aggfunc="last"
    ).reset_index()
    pivoted = pivoted.sort_values("end").reset_index(drop=True)

    # Merge the two possible revenue tags into one column.
    if "revenue" not in pivoted.columns:
        pivoted["revenue"] = pd.NA
    if "revenue_alt" in pivoted.columns:
        pivoted["revenue"] = pivoted["revenue"].fillna(pivoted["revenue_alt"])
        pivoted = pivoted.drop(columns=["revenue_alt"])

    return pivoted


def compute_ttm(pivoted: pd.DataFrame) -> pd.DataFrame:
    """
    Approximate trailing-twelve-month net income / revenue by summing the
    last 4 quarterly (10-Q) rows. Falls back to the latest 10-K value
    when fewer than 4 quarterly rows are available (e.g. early in the
    fetched history).

    Known simplification: SEC's 10-Q figures are often *quarterly*
    (3-month) deltas but occasionally cumulative year-to-date; this
    function does not attempt to detect/correct that distinction. Good
    enough for a first-pass ranking signal, not for GAAP-exact TTM.
    """
    out = pivoted.copy()
    quarterly = out[out["form"] == "10-Q"].sort_values("end")
    annual = out[out["form"] == "10-K"].sort_values("end")

    for col, ttm_col in [("net_income", "net_income_ttm"), ("revenue", "revenue_ttm")]:
        if col not in out.columns:
            continue
        ttm_from_quarters = (
            quarterly[col].rolling(window=4, min_periods=4).sum()
            if col in quarterly.columns
            else pd.Series(dtype=float)
        )
        quarterly = quarterly.assign(**{ttm_col: ttm_from_quarters})

    out = out.merge(
        quarterly[["end"] + [c for c in ["net_income_ttm", "revenue_ttm"] if c in quarterly.columns]],
        on="end",
        how="left",
    )

    # Fallback: where TTM is still missing, use the latest annual figure
    # as of that date (forward-filled from 10-K rows).
    for col, ttm_col in [("net_income", "net_income_ttm"), ("revenue", "revenue_ttm")]:
        if ttm_col not in out.columns or col not in annual.columns:
            continue
        annual_sorted = annual.set_index("end")[col].sort_index()
        out[f"{col}_annual_fallback"] = out["end"].map(
            lambda d: annual_sorted.loc[:d].iloc[-1] if len(annual_sorted.loc[:d]) else pd.NA
        )
        out[ttm_col] = out[ttm_col].fillna(out[f"{col}_annual_fallback"])
        out = out.drop(columns=[f"{col}_annual_fallback"])

    return out


def join_with_price(fund_df: pd.DataFrame, price_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge daily close price with the latest known fundamentals as-of
    each date (point-in-time join: only use fundamentals data that had
    already been filed by that date, via merge_asof on filing date logic
    approximated here by fiscal period end -- a simplification since we
    don't carry `filed` through compute_ttm; documented limitation).
    """
    price_df = price_df[["date", "close", "adj_close"]].copy()
    price_df = price_df.sort_values("date")

    fund_df = fund_df.sort_values("end")
    fund_df = fund_df.rename(columns={"end": "date"})

    merged = pd.merge_asof(price_df, fund_df, on="date", direction="backward")
    return merged


def compute_ratios(merged: pd.DataFrame) -> pd.DataFrame:
    out = merged.copy()

    shares = None
    for candidate in ["diluted_shares_wavg", "shares_outstanding", "basic_shares_wavg"]:
        if candidate in out.columns:
            shares = out[candidate]
            break

    if shares is not None and "net_income_ttm" in out.columns:
        out["eps_ttm"] = out["net_income_ttm"] / shares
        out["pe_ratio"] = out["close"] / out["eps_ttm"]

    if shares is not None and "stockholders_equity" in out.columns:
        out["book_value_per_share"] = out["stockholders_equity"] / shares
        out["pb_ratio"] = out["close"] / out["book_value_per_share"]

    if "operating_cash_flow" in out.columns and "capex" in out.columns and shares is not None:
        out["free_cash_flow"] = out["operating_cash_flow"] - out["capex"].abs()
        out["fcf_per_share"] = out["free_cash_flow"] / shares
        out["fcf_yield"] = out["fcf_per_share"] / out["close"]

    if "net_income_ttm" in out.columns and "revenue_ttm" in out.columns:
        out["net_margin_ttm"] = out["net_income_ttm"] / out["revenue_ttm"]

    if "revenue_ttm" in out.columns:
        out["revenue_ttm_yoy_growth"] = out["revenue_ttm"].pct_change(periods=252)  # ~1 trading year

    return out


def process_ticker(ticker: str) -> pd.DataFrame | None:
    jsonl_path = FUNDAMENTALS_RAW_DIR / f"{ticker}.jsonl"
    price_path = STOCKS_RAW_DIR / f"{ticker}.csv"

    if not jsonl_path.exists():
        logger.warning("No fundamentals file for %s (%s); skipping.", ticker, jsonl_path)
        return None
    if not price_path.exists():
        logger.warning("No price file for %s (%s); skipping.", ticker, price_path)
        return None

    raw = load_fundamentals_long(jsonl_path)
    if raw.empty:
        logger.warning("Fundamentals file for %s is empty; skipping.", ticker)
        return None

    pivoted = pivot_quarterly(raw)
    if pivoted.empty:
        logger.warning("No matching concepts found for %s; skipping.", ticker)
        return None

    with_ttm = compute_ttm(pivoted)
    price_df = pd.read_csv(price_path, parse_dates=["date"])
    merged = join_with_price(with_ttm, price_df)
    ratios = compute_ratios(merged)
    ratios.insert(0, "ticker", ticker)
    return ratios


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute valuation ratios from SEC fundamentals + price data.")
    parser.add_argument(
        "--tickers",
        default=None,
        help="Comma-separated tickers, e.g. AAPL,NVDA. Defaults to has_fundamentals=true entries in config/assets.yaml.",
    )
    args = parser.parse_args()

    if args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    else:
        cfg = load_asset_config()
        tickers = [s["ticker"] for s in cfg.get("stocks", []) if s.get("has_fundamentals")]

    if not tickers:
        logger.error("No tickers to process. Check config/assets.yaml or pass --tickers.")
        sys.exit(1)

    ensure_dir(PROCESSED_DIR)

    for ticker in tickers:
        logger.info("Computing fundamentals ratios for %s", ticker)
        result = process_ticker(ticker)
        if result is None:
            continue
        out_path = PROCESSED_DIR / f"{ticker}.csv"
        result.to_csv(out_path, index=False)
        logger.info("Saved %d rows -> %s", len(result), out_path)

    logger.info("Done.")


if __name__ == "__main__":
    main()

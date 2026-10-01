"""
run_scores.py
==============
End-to-end scoring runner: reads the processed technical/fundamentals
feature files, computes Timing/Value/Risk scores for every asset, prints
a one-line summary per asset for the latest available date, and writes a
full per-date score history CSV per asset to data/processed/scores/.

Usage:
    python -m src.scoring.run_scores
    python -m src.scoring.run_scores --symbols AAPL,BTC
"""
from __future__ import annotations

import argparse
import sys

import pandas as pd

from src.data.utils import PROJECT_ROOT, ensure_dir, get_logger, load_asset_config
from src.scoring.explain import explain_asset
from src.scoring.position_sizing import (
    DEFAULT_MAX_POSITION_PCT,
    DEFAULT_RISK_PER_TRADE_PCT,
    MAX_STOP_DISTANCE_PCT,
    MIN_STOP_DISTANCE_PCT,
)
from src.scoring.risk import compute_risk_score
from src.scoring.timing import compute_timing_score
from src.scoring.value import compute_crypto_value_score, compute_equity_value_score

logger = get_logger("scoring.run_scores")

TECHNICAL_DIR = PROJECT_ROOT / "data" / "processed" / "technical"
FUNDAMENTALS_DIR = PROJECT_ROOT / "data" / "processed" / "fundamentals"
SCORES_DIR = PROJECT_ROOT / "data" / "processed" / "scores"


def add_decision_columns(
    df: pd.DataFrame,
    price_col: str,
    account_value: float,
    risk_per_trade_pct: float = DEFAULT_RISK_PER_TRADE_PCT,
    max_position_pct: float = DEFAULT_MAX_POSITION_PCT,
) -> pd.DataFrame:
    """
    Add final score and beginner-friendly position-sizing columns.

    Missing value scores (common for ETFs) are treated as neutral in the
    final score calculation, but remain blank in the raw value_score
    column so the output does not pretend valuation data exists.
    """
    out = df.copy()
    value_for_final = pd.to_numeric(out["value_score"], errors="coerce").fillna(50)
    out["final_score"] = (
        out["timing_score"].fillna(0) * 0.45
        + value_for_final * 0.35
        + (100 - out["risk_score"].fillna(100)) * 0.20
    ).clip(0, 100)

    stop_span = MAX_STOP_DISTANCE_PCT - MIN_STOP_DISTANCE_PCT
    out["stop_distance_pct"] = MIN_STOP_DISTANCE_PCT + (out["risk_score"].fillna(100).clip(0, 100) / 100) * stop_span
    out["stop_loss_price"] = out[price_col] * (1 - out["stop_distance_pct"])
    raw_position_pct = risk_per_trade_pct / out["stop_distance_pct"]
    out["suggested_pct_of_account"] = raw_position_pct.clip(upper=max_position_pct)
    out["suggested_dollar_amount"] = account_value * out["suggested_pct_of_account"]
    out["suggested_units"] = out["suggested_dollar_amount"] / out[price_col]
    return out


def score_stock(ticker: str, account_value: float) -> pd.DataFrame | None:
    tech_path = TECHNICAL_DIR / f"{ticker}.csv"
    if not tech_path.exists():
        logger.warning("No technical file for %s; run src/features/technical.py first.", ticker)
        return None

    tech_df = pd.read_csv(tech_path, parse_dates=["date"])
    tech_df = compute_timing_score(tech_df, price_col="adj_close")
    tech_df = compute_risk_score(tech_df)

    fund_path = FUNDAMENTALS_DIR / f"{ticker}.csv"
    if fund_path.exists():
        fund_df = pd.read_csv(fund_path, parse_dates=["date"])
        fund_df = compute_equity_value_score(fund_df)
        value_cols = fund_df[["date", "value_score", "value_label"]]
        merged = tech_df.merge(value_cols, on="date", how="left")
    else:
        logger.info("No fundamentals file for %s (ETF or missing data); value_score left blank.", ticker)
        merged = tech_df.copy()
        merged["value_score"] = pd.NA
        merged["value_label"] = pd.NA

    merged["value_score"] = merged["value_score"].ffill()
    merged["value_label"] = merged["value_label"].ffill()
    merged.insert(0, "symbol", ticker)
    return add_decision_columns(merged, price_col="adj_close", account_value=account_value)


def score_crypto(symbol: str, account_value: float) -> pd.DataFrame | None:
    tech_path = TECHNICAL_DIR / f"{symbol}.csv"
    if not tech_path.exists():
        logger.warning("No technical file for %s; run src/features/technical.py first.", symbol)
        return None

    df = pd.read_csv(tech_path, parse_dates=["date"])
    df = compute_timing_score(df, price_col="price_usd")
    df = compute_risk_score(df)
    df = compute_crypto_value_score(df, price_col="price_usd")
    if "symbol" not in df.columns:
        df.insert(0, "symbol", symbol)
    return add_decision_columns(df, price_col="price_usd", account_value=account_value)


def summarize_latest(df: pd.DataFrame) -> None:
    latest = df.dropna(subset=["timing_score"]).iloc[-1]
    raw_value_score = latest.get("value_score")
    value_score = None if pd.isna(raw_value_score) else raw_value_score
    raw_value_label = latest.get("value_label")
    value_label = "unavailable" if pd.isna(raw_value_label) else str(raw_value_label)

    explanation = explain_asset(
        symbol=latest["symbol"],
        timing_score=latest["timing_score"],
        timing_label=str(latest["timing_label"]),
        value_score=value_score,
        value_label=value_label,
        risk_score=latest["risk_score"],
        risk_label=str(latest["risk_label"]),
    )
    print(f"\n{explanation.detail}")
    print(f"  -> {explanation.summary}")
    print(
        "  -> Position guide: "
        f"{latest['suggested_units']:.4f} units "
        f"(~${latest['suggested_dollar_amount']:,.2f}, "
        f"{latest['suggested_pct_of_account'] * 100:.1f}% of account), "
        f"stop ~{latest['stop_distance_pct'] * 100:.1f}% below entry."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute Timing/Value/Risk scores for the configured asset universe.")
    parser.add_argument(
        "--symbols",
        default=None,
        help="Comma-separated symbol override, e.g. AAPL,BTC. Defaults to all assets in config/assets.yaml.",
    )
    parser.add_argument(
        "--account-value",
        type=float,
        default=10_000,
        help="Account value used for educational position-size columns (default: 10000).",
    )
    args = parser.parse_args()

    cfg = load_asset_config()
    stock_tickers = [s["ticker"] for s in cfg.get("stocks", [])]
    crypto_symbols = [c["symbol"] for c in cfg.get("crypto", [])]

    if args.symbols:
        wanted = {s.strip().upper() for s in args.symbols.split(",") if s.strip()}
        stock_tickers = [t for t in stock_tickers if t in wanted]
        crypto_symbols = [s for s in crypto_symbols if s in wanted]

    if not stock_tickers and not crypto_symbols:
        logger.error("No matching symbols found. Check --symbols or config/assets.yaml.")
        sys.exit(1)

    ensure_dir(SCORES_DIR)

    for ticker in stock_tickers:
        logger.info("Scoring %s (stock)", ticker)
        df = score_stock(ticker, account_value=args.account_value)
        if df is None:
            continue
        out_path = SCORES_DIR / f"{ticker}.csv"
        df.to_csv(out_path, index=False)
        logger.info("Saved %d rows -> %s", len(df), out_path)
        summarize_latest(df)

    for symbol in crypto_symbols:
        logger.info("Scoring %s (crypto)", symbol)
        df = score_crypto(symbol, account_value=args.account_value)
        if df is None:
            continue
        out_path = SCORES_DIR / f"{symbol}.csv"
        df.to_csv(out_path, index=False)
        logger.info("Saved %d rows -> %s", len(df), out_path)
        summarize_latest(df)

    logger.info("Done.")


if __name__ == "__main__":
    main()

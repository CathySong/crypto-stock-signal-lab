"""
build_watchlist.py
==================
Build a ranked research watchlist from the latest score, backtest, and ML
benchmark outputs.

Usage:
    python -m src.reports.build_watchlist
    python -m src.reports.build_watchlist --symbols AAPL,BTC
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import pandas as pd

from src.data.utils import DATA_PROCESSED_DIR, ensure_dir, get_logger, load_asset_config

logger = get_logger("reports.build_watchlist")

SCORES_DIR = DATA_PROCESSED_DIR / "scores"
BACKTEST_DIR = DATA_PROCESSED_DIR / "backtests"
ML_DIR = DATA_PROCESSED_DIR / "ml"
REPORTS_DIR = DATA_PROCESSED_DIR / "reports"


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def metric_or_nan(row: pd.Series | None, name: str) -> float:
    if row is None or name not in row or pd.isna(row[name]):
        return float("nan")
    return float(row[name])


def score_from_sharpe(sharpe: float) -> float:
    if math.isnan(sharpe):
        return 50.0
    return clamp((sharpe + 0.5) / 2.5 * 100, 0, 100)


def score_from_auc(auc: float) -> float:
    if math.isnan(auc):
        return 50.0
    return clamp((auc - 0.45) / 0.25 * 100, 0, 100)


def classify_asset(
    final_score: float,
    timing_score: float,
    value_score: float,
    risk_score: float,
    best_sharpe: float,
    ml_auc: float,
) -> tuple[str, list[str]]:
    flags: list[str] = []

    if final_score >= 65 and risk_score <= 55 and best_sharpe > 0:
        action = "Research now"
        flags.append("strong blended score with manageable measured risk")
    elif final_score >= 55 and timing_score >= 60 and risk_score <= 70:
        action = "Watch closely"
        flags.append("constructive timing setup but not a full green light")
    elif risk_score >= 75:
        action = "Wait"
        flags.append("risk score is high enough to demand extra caution")
    elif final_score < 45:
        action = "Deprioritize"
        flags.append("blended score is weak relative to the universe")
    else:
        action = "Review"
        flags.append("mixed signal stack")

    if timing_score >= 70:
        flags.append("technical timing is strong")
    elif timing_score < 45:
        flags.append("technical timing is weak")

    if not math.isnan(value_score):
        if value_score >= 60:
            flags.append("value score is supportive")
        elif value_score < 35:
            flags.append("valuation score is expensive or unattractive")

    if risk_score <= 35:
        flags.append("risk score is low")
    elif risk_score >= 60:
        flags.append("risk score is elevated")

    if not math.isnan(best_sharpe):
        if best_sharpe >= 0.75:
            flags.append("best baseline backtest has strong Sharpe")
        elif best_sharpe <= 0:
            flags.append("baseline backtests are not supportive")

    if not math.isnan(ml_auc):
        if ml_auc >= 0.60:
            flags.append("ML benchmark shows directional separation")
        elif ml_auc <= 0.50:
            flags.append("ML benchmark is weak or coin-flip")

    return action, flags


def research_priority_score(
    final_score: float,
    risk_score: float,
    best_sharpe: float,
    ml_auc: float,
) -> float:
    return round(
        0.45 * final_score
        + 0.25 * (100 - risk_score)
        + 0.20 * score_from_sharpe(best_sharpe)
        + 0.10 * score_from_auc(ml_auc),
        2,
    )


def load_symbols(symbol_override: str | None) -> list[str]:
    cfg = load_asset_config()
    symbols = [s["ticker"] for s in cfg.get("stocks", [])] + [c["symbol"] for c in cfg.get("crypto", [])]
    if symbol_override:
        wanted = {s.strip().upper() for s in symbol_override.split(",") if s.strip()}
        symbols = [s for s in symbols if s in wanted]
    return symbols


def latest_score_row(symbol: str) -> pd.Series | None:
    path = SCORES_DIR / f"{symbol}.csv"
    if not path.exists():
        logger.warning("Missing score file for %s; run src.scoring.run_scores first.", symbol)
        return None
    df = pd.read_csv(path, parse_dates=["date"]).sort_values("date")
    if df.empty:
        return None
    return df.iloc[-1]


def best_backtest_row(backtests: pd.DataFrame, symbol: str) -> pd.Series | None:
    if backtests.empty:
        return None
    matches = backtests[backtests["symbol"] == symbol]
    if matches.empty:
        return None
    return matches.sort_values("sharpe", ascending=False).iloc[0]


def ml_row(ml_summary: pd.DataFrame, symbol: str) -> pd.Series | None:
    if ml_summary.empty:
        return None
    matches = ml_summary[ml_summary["symbol"] == symbol]
    if matches.empty:
        return None
    return matches.iloc[0]


def load_optional_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        logger.warning("Optional report input missing: %s", path)
        return pd.DataFrame()
    return pd.read_csv(path)


def build_watchlist(symbols: list[str], backtests: pd.DataFrame, ml_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for symbol in symbols:
        latest = latest_score_row(symbol)
        if latest is None:
            continue

        backtest = best_backtest_row(backtests, symbol)
        ml = ml_row(ml_summary, symbol)

        final_score = metric_or_nan(latest, "final_score")
        timing_score = metric_or_nan(latest, "timing_score")
        value_score = metric_or_nan(latest, "value_score")
        risk_score = metric_or_nan(latest, "risk_score")
        best_sharpe = metric_or_nan(backtest, "sharpe")
        auc = metric_or_nan(ml, "roc_auc")
        action, flags = classify_asset(
            final_score=final_score,
            timing_score=timing_score,
            value_score=value_score,
            risk_score=risk_score,
            best_sharpe=best_sharpe,
            ml_auc=auc,
        )

        price = latest.get("adj_close", latest.get("price_usd", latest.get("close", float("nan"))))
        rows.append(
            {
                "symbol": symbol,
                "date": pd.to_datetime(latest["date"]).date().isoformat(),
                "research_action": action,
                "priority_score": research_priority_score(final_score, risk_score, best_sharpe, auc),
                "final_score": final_score,
                "timing_score": timing_score,
                "value_score": value_score,
                "risk_score": risk_score,
                "price": price,
                "suggested_pct_of_account": metric_or_nan(latest, "suggested_pct_of_account"),
                "stop_distance_pct": metric_or_nan(latest, "stop_distance_pct"),
                "best_backtest_strategy": None if backtest is None else str(backtest["strategy"]),
                "best_backtest_sharpe": best_sharpe,
                "best_backtest_return": metric_or_nan(backtest, "total_return"),
                "best_backtest_drawdown": metric_or_nan(backtest, "max_drawdown"),
                "ml_auc": auc,
                "ml_strategy_return": metric_or_nan(ml, "strategy_total_return"),
                "flags": "; ".join(flags),
            }
        )

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["priority_score", "symbol"], ascending=[False, True]).reset_index(drop=True)


def fmt_pct(value: float) -> str:
    if pd.isna(value):
        return "n/a"
    return f"{float(value) * 100:.1f}%"


def fmt_num(value: float, digits: int = 1) -> str:
    if pd.isna(value):
        return "n/a"
    return f"{float(value):.{digits}f}"


def render_markdown(watchlist: pd.DataFrame) -> str:
    if watchlist.empty:
        return "# Research watchlist\n\nNo rows generated.\n"

    generated_at = pd.Timestamp.now(tz="America/New_York").strftime("%Y-%m-%d %H:%M %Z")
    lines = [
        "# Research watchlist",
        "",
        f"Generated: {generated_at}",
        "",
        "Educational research only. This is not financial advice or a trade recommendation.",
        "",
        "| Rank | Asset | Action | Priority | Final | Risk | Best strategy | Sharpe | ML AUC | Flags |",
        "|---:|---|---|---:|---:|---:|---|---:|---:|---|",
    ]
    for idx, row in watchlist.iterrows():
        flags = str(row["flags"]).replace("|", "/")
        lines.append(
            "| "
            f"{idx + 1} | {row['symbol']} | {row['research_action']} | "
            f"{fmt_num(row['priority_score'])} | {fmt_num(row['final_score'])} | "
            f"{fmt_num(row['risk_score'])} | {row['best_backtest_strategy'] or 'n/a'} | "
            f"{fmt_num(row['best_backtest_sharpe'], 2)} | {fmt_num(row['ml_auc'], 2)} | {flags} |"
        )

    lines.extend(
        [
            "",
            "## Position context",
            "",
            "| Asset | Price | Suggested account % | Stop distance | Best return | Best drawdown | ML strategy return |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in watchlist.iterrows():
        lines.append(
            "| "
            f"{row['symbol']} | ${float(row['price']):,.2f} | "
            f"{fmt_pct(row['suggested_pct_of_account'])} | {fmt_pct(row['stop_distance_pct'])} | "
            f"{fmt_pct(row['best_backtest_return'])} | {fmt_pct(row['best_backtest_drawdown'])} | "
            f"{fmt_pct(row['ml_strategy_return'])} |"
        )
    lines.append("")
    return "\n".join(lines)


def print_summary(watchlist: pd.DataFrame) -> None:
    cols = ["symbol", "research_action", "priority_score", "final_score", "risk_score", "best_backtest_strategy", "ml_auc"]
    print("\nResearch watchlist:")
    print(watchlist[cols].to_string(index=False, formatters={
        "priority_score": "{:.1f}".format,
        "final_score": "{:.1f}".format,
        "risk_score": "{:.1f}".format,
        "ml_auc": "{:.2f}".format,
    }))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a ranked research watchlist from processed outputs.")
    parser.add_argument("--symbols", default=None, help="Comma-separated symbol override, e.g. AAPL,BTC.")
    parser.add_argument("--csv-output", default=str(REPORTS_DIR / "watchlist.csv"), help="Output CSV path.")
    parser.add_argument("--md-output", default=str(REPORTS_DIR / "watchlist.md"), help="Output Markdown path.")
    args = parser.parse_args()

    symbols = load_symbols(args.symbols)
    if not symbols:
        logger.error("No matching symbols found. Check --symbols or config/assets.yaml.")
        sys.exit(1)

    backtests = load_optional_csv(BACKTEST_DIR / "summary.csv")
    ml_summary = load_optional_csv(ML_DIR / "walkforward_summary.csv")
    watchlist = build_watchlist(symbols, backtests=backtests, ml_summary=ml_summary)
    if watchlist.empty:
        logger.error("No watchlist rows produced. Run scores first.")
        sys.exit(1)

    csv_output = Path(args.csv_output)
    md_output = Path(args.md_output)
    ensure_dir(csv_output.parent)
    ensure_dir(md_output.parent)
    watchlist.to_csv(csv_output, index=False)
    md_output.write_text(render_markdown(watchlist), encoding="utf-8")

    logger.info("Saved watchlist CSV -> %s", csv_output)
    logger.info("Saved watchlist Markdown -> %s", md_output)
    print_summary(watchlist)


if __name__ == "__main__":
    main()


"""
run_backtests.py
================
Run daily long-only backtests for every scored asset.

The runner compares four transparent baselines:
    1. buy_and_hold
    2. ma_50_200_crossover
    3. timing_score_rule
    4. score_weighted_risk_sized

Positions are shifted one day before returns are applied to avoid using
same-day closing information to trade the same close.

Usage:
    python -m src.backtest.run_backtests
    python -m src.backtest.run_backtests --symbols AAPL,BTC
    python -m src.backtest.run_backtests --fee-bps 10 --slippage-bps 5
"""
from __future__ import annotations

import argparse
import sys

import pandas as pd

from src.backtest.metrics import summarize_performance
from src.backtest.strategies import STRATEGIES
from src.data.utils import PROJECT_ROOT, ensure_dir, get_logger, load_asset_config

logger = get_logger("backtest.run_backtests")

SCORES_DIR = PROJECT_ROOT / "data" / "processed" / "scores"
BACKTEST_DIR = PROJECT_ROOT / "data" / "processed" / "backtests"
EQUITY_DIR = BACKTEST_DIR / "equity_curves"


def infer_price_col(df: pd.DataFrame) -> str:
    if "adj_close" in df.columns:
        return "adj_close"
    if "price_usd" in df.columns:
        return "price_usd"
    if "close" in df.columns:
        return "close"
    raise ValueError("No supported price column found; expected adj_close, price_usd, or close")


def load_score_file(symbol: str) -> pd.DataFrame | None:
    path = SCORES_DIR / f"{symbol}.csv"
    if not path.exists():
        logger.warning("No score file for %s; run src.scoring.run_scores first.", symbol)
        return None

    df = pd.read_csv(path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    df["symbol"] = symbol
    return df


def backtest_strategy(
    df: pd.DataFrame,
    strategy_name: str,
    fee_bps: float,
    slippage_bps: float,
    initial_capital: float,
) -> tuple[dict[str, float | str], pd.DataFrame]:
    price_col = infer_price_col(df)
    work = df.copy()
    work["asset_return"] = work[price_col].pct_change().fillna(0)

    target_exposure = STRATEGIES[strategy_name](work).clip(0, 1).fillna(0)
    traded_exposure = target_exposure.shift(1).fillna(0)
    turnover = traded_exposure.diff().abs().fillna(traded_exposure.abs())

    cost_rate = (fee_bps + slippage_bps) / 10_000
    strategy_return = traded_exposure * work["asset_return"] - turnover * cost_rate
    equity_curve = initial_capital * (1 + strategy_return).cumprod()

    metrics = summarize_performance(
        daily_returns=strategy_return,
        equity_curve=equity_curve,
        exposure=traded_exposure,
        turnover=turnover,
    )
    metrics.update(
        {
            "symbol": str(work["symbol"].iloc[0]),
            "strategy": strategy_name,
            "start_date": work["date"].iloc[0].date().isoformat(),
            "end_date": work["date"].iloc[-1].date().isoformat(),
            "fee_bps": fee_bps,
            "slippage_bps": slippage_bps,
            "initial_capital": initial_capital,
            "ending_equity": float(equity_curve.iloc[-1]),
        }
    )

    curve = pd.DataFrame(
        {
            "date": work["date"],
            "symbol": work["symbol"],
            "strategy": strategy_name,
            "price": work[price_col],
            "asset_return": work["asset_return"],
            "target_exposure": target_exposure,
            "traded_exposure": traded_exposure,
            "turnover": turnover,
            "strategy_return": strategy_return,
            "equity": equity_curve,
        }
    )
    return metrics, curve


def run_for_symbol(
    symbol: str,
    fee_bps: float,
    slippage_bps: float,
    initial_capital: float,
) -> tuple[list[dict[str, float | str]], list[pd.DataFrame]]:
    df = load_score_file(symbol)
    if df is None:
        return [], []

    metrics_rows = []
    curves = []
    for strategy_name in STRATEGIES:
        metrics, curve = backtest_strategy(
            df=df,
            strategy_name=strategy_name,
            fee_bps=fee_bps,
            slippage_bps=slippage_bps,
            initial_capital=initial_capital,
        )
        metrics_rows.append(metrics)
        curves.append(curve)

    return metrics_rows, curves


def print_summary(summary: pd.DataFrame) -> None:
    view = summary.sort_values(["symbol", "sharpe"], ascending=[True, False])
    cols = ["symbol", "strategy", "total_return", "cagr", "sharpe", "max_drawdown", "average_exposure"]
    print("\nBacktest summary (sorted by symbol, then Sharpe):")
    print(view[cols].to_string(index=False, formatters={
        "total_return": "{:.1%}".format,
        "cagr": "{:.1%}".format,
        "sharpe": "{:.2f}".format,
        "max_drawdown": "{:.1%}".format,
        "average_exposure": "{:.1%}".format,
    }))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run strategy backtests for scored assets.")
    parser.add_argument(
        "--symbols",
        default=None,
        help="Comma-separated symbol override, e.g. AAPL,BTC. Defaults to all assets in config/assets.yaml.",
    )
    parser.add_argument("--fee-bps", type=float, default=10, help="Trading fee in basis points per turnover unit.")
    parser.add_argument("--slippage-bps", type=float, default=5, help="Slippage in basis points per turnover unit.")
    parser.add_argument("--initial-capital", type=float, default=10_000, help="Starting capital for each strategy.")
    args = parser.parse_args()

    cfg = load_asset_config()
    symbols = [s["ticker"] for s in cfg.get("stocks", [])] + [c["symbol"] for c in cfg.get("crypto", [])]
    if args.symbols:
        wanted = {s.strip().upper() for s in args.symbols.split(",") if s.strip()}
        symbols = [s for s in symbols if s in wanted]

    if not symbols:
        logger.error("No matching symbols found. Check --symbols or config/assets.yaml.")
        sys.exit(1)

    ensure_dir(BACKTEST_DIR)
    ensure_dir(EQUITY_DIR)

    all_metrics: list[dict[str, float | str]] = []
    all_curves: list[pd.DataFrame] = []
    for symbol in symbols:
        logger.info("Running backtests for %s", symbol)
        metrics_rows, curves = run_for_symbol(
            symbol=symbol,
            fee_bps=args.fee_bps,
            slippage_bps=args.slippage_bps,
            initial_capital=args.initial_capital,
        )
        all_metrics.extend(metrics_rows)
        all_curves.extend(curves)

    if not all_metrics:
        logger.error("No backtests ran. Run features and scores first.")
        sys.exit(1)

    summary = pd.DataFrame(all_metrics).sort_values(["symbol", "strategy"]).reset_index(drop=True)
    summary_path = BACKTEST_DIR / "summary.csv"
    summary.to_csv(summary_path, index=False)
    logger.info("Saved summary -> %s", summary_path)

    curves_df = pd.concat(all_curves, ignore_index=True)
    for (symbol, strategy), curve_df in curves_df.groupby(["symbol", "strategy"], sort=True):
        out_path = EQUITY_DIR / f"{symbol}_{strategy}.csv"
        curve_df.to_csv(out_path, index=False)

    print_summary(summary)
    logger.info("Done.")


if __name__ == "__main__":
    main()


"""
run_walkforward.py
==================
Walk-forward direction-probability benchmark.

This Day 5 layer trains a small logistic-regression model to estimate the
probability that an asset's forward return will be positive over a fixed
horizon. It is intentionally simple and transparent: a benchmark that
future tree/boosting/deep-learning models must beat, not a claim that ML
can predict markets reliably.

Usage:
    python -m src.ml.run_walkforward
    python -m src.ml.run_walkforward --symbols AAPL,BTC
    python -m src.ml.run_walkforward --horizon-days 5 --min-train-days 252
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from src.backtest.metrics import summarize_performance
from src.data.utils import PROJECT_ROOT, ensure_dir, get_logger, load_asset_config
from src.ml.logistic import fit_logistic_regression, predict_proba
from src.ml.metrics import summarize_classifier

logger = get_logger("ml.run_walkforward")

SCORES_DIR = PROJECT_ROOT / "data" / "processed" / "scores"
ML_DIR = PROJECT_ROOT / "data" / "processed" / "ml"
PREDICTIONS_DIR = ML_DIR / "predictions"

BASE_FEATURES = [
    "timing_score",
    "value_score",
    "risk_score",
    "final_score",
    "rsi_14",
    "macd_hist",
    "volatility_30d_annualized",
    "drawdown_from_ath",
    "suggested_pct_of_account",
]


def infer_price_col(df: pd.DataFrame) -> str:
    if "adj_close" in df.columns:
        return "adj_close"
    if "price_usd" in df.columns:
        return "price_usd"
    if "close" in df.columns:
        return "close"
    raise ValueError("No supported price column found; expected adj_close, price_usd, or close")


def load_scores(symbol: str) -> pd.DataFrame | None:
    path = SCORES_DIR / f"{symbol}.csv"
    if not path.exists():
        logger.warning("No score file for %s; run src.scoring.run_scores first.", symbol)
        return None

    df = pd.read_csv(path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    df["symbol"] = symbol
    return df


def prepare_dataset(df: pd.DataFrame, horizon_days: int) -> tuple[pd.DataFrame, list[str]]:
    out = df.copy()
    price_col = infer_price_col(out)

    out["future_return"] = out[price_col].shift(-horizon_days) / out[price_col] - 1
    out["target_up"] = (out["future_return"] > 0).astype(int)

    if "value_score" in out.columns:
        out["value_score"] = pd.to_numeric(out["value_score"], errors="coerce").fillna(50)
    if "volume_to_mcap_ratio" not in out.columns:
        out["volume_to_mcap_ratio"] = 0.0
    if "above_sma_50" in out.columns:
        out["above_sma_50_num"] = out["above_sma_50"].astype(float)
    if "golden_cross" in out.columns:
        out["golden_cross_num"] = out["golden_cross"].astype(float)

    feature_cols = [c for c in BASE_FEATURES if c in out.columns]
    feature_cols.extend([c for c in ["volume_to_mcap_ratio", "above_sma_50_num", "golden_cross_num"] if c in out.columns])

    clean = out.dropna(subset=["future_return", *feature_cols]).reset_index(drop=True)
    return clean, feature_cols


def walk_forward_predict(
    df: pd.DataFrame,
    feature_cols: list[str],
    min_train_days: int,
    train_window_days: int,
    retrain_every_days: int,
) -> pd.DataFrame:
    predictions: list[pd.DataFrame] = []

    if len(df) <= min_train_days + 5:
        return pd.DataFrame()

    for start in range(min_train_days, len(df), retrain_every_days):
        stop = min(start + retrain_every_days, len(df))
        train_start = max(0, start - train_window_days)
        train = df.iloc[train_start:start]
        test = df.iloc[start:stop]
        if train["target_up"].nunique() < 2 or test.empty:
            continue

        model = fit_logistic_regression(
            x=train[feature_cols].to_numpy(dtype=float),
            y=train["target_up"].to_numpy(dtype=float),
        )
        proba = predict_proba(model, test[feature_cols].to_numpy(dtype=float))

        block = test[["date", "symbol", "future_return", "target_up"]].copy()
        block["probability_up"] = proba
        block["predicted_up"] = (block["probability_up"] >= 0.50).astype(int)
        predictions.append(block)

    if not predictions:
        return pd.DataFrame()
    return pd.concat(predictions, ignore_index=True)


def backtest_probability_signal(
    scored_df: pd.DataFrame,
    predictions: pd.DataFrame,
    fee_bps: float,
    slippage_bps: float,
    initial_capital: float,
) -> dict[str, float]:
    price_col = infer_price_col(scored_df)
    work = scored_df[["date", "symbol", price_col]].merge(
        predictions[["date", "probability_up"]],
        on="date",
        how="inner",
    )
    work["asset_return"] = work[price_col].pct_change().fillna(0)

    target_exposure = pd.Series(0.0, index=work.index)
    target_exposure[work["probability_up"] >= 0.58] = 1.0
    target_exposure[(work["probability_up"] >= 0.52) & (work["probability_up"] < 0.58)] = 0.5

    traded_exposure = target_exposure.shift(1).fillna(0)
    turnover = traded_exposure.diff().abs().fillna(traded_exposure.abs())
    cost_rate = (fee_bps + slippage_bps) / 10_000
    strategy_return = traded_exposure * work["asset_return"] - turnover * cost_rate
    equity_curve = initial_capital * (1 + strategy_return).cumprod()

    metrics = summarize_performance(strategy_return, equity_curve, traded_exposure, turnover)
    metrics["ending_equity"] = float(equity_curve.iloc[-1]) if len(equity_curve) else initial_capital
    metrics["average_probability_up"] = float(work["probability_up"].mean()) if len(work) else float("nan")
    return metrics


def run_symbol(
    symbol: str,
    horizon_days: int,
    min_train_days: int,
    train_window_days: int,
    retrain_every_days: int,
    fee_bps: float,
    slippage_bps: float,
    initial_capital: float,
) -> dict[str, float | str] | None:
    scored = load_scores(symbol)
    if scored is None:
        return None

    dataset, feature_cols = prepare_dataset(scored, horizon_days=horizon_days)
    if len(dataset) <= min_train_days:
        logger.warning("Not enough rows for %s after feature preparation.", symbol)
        return None

    predictions = walk_forward_predict(
        df=dataset,
        feature_cols=feature_cols,
        min_train_days=min_train_days,
        train_window_days=train_window_days,
        retrain_every_days=retrain_every_days,
    )
    if predictions.empty:
        logger.warning("No predictions generated for %s.", symbol)
        return None

    out_path = PREDICTIONS_DIR / f"{symbol}_predictions.csv"
    predictions.to_csv(out_path, index=False)

    classifier_metrics = summarize_classifier(predictions["target_up"], predictions["probability_up"])
    strategy_metrics = backtest_probability_signal(
        scored_df=scored,
        predictions=predictions,
        fee_bps=fee_bps,
        slippage_bps=slippage_bps,
        initial_capital=initial_capital,
    )

    return {
        "symbol": symbol,
        "horizon_days": horizon_days,
        "features": float(len(feature_cols)),
        **classifier_metrics,
        "strategy_total_return": strategy_metrics["total_return"],
        "strategy_cagr": strategy_metrics["cagr"],
        "strategy_sharpe": strategy_metrics["sharpe"],
        "strategy_max_drawdown": strategy_metrics["max_drawdown"],
        "strategy_average_exposure": strategy_metrics["average_exposure"],
        "strategy_turnover": strategy_metrics["turnover"],
        "ending_equity": strategy_metrics["ending_equity"],
        "average_probability_up": strategy_metrics["average_probability_up"],
    }


def print_summary(summary: pd.DataFrame) -> None:
    view = summary.sort_values("roc_auc", ascending=False)
    cols = [
        "symbol",
        "samples",
        "accuracy",
        "roc_auc",
        "brier_score",
        "strategy_total_return",
        "strategy_sharpe",
        "strategy_max_drawdown",
        "strategy_average_exposure",
    ]
    print("\nWalk-forward ML benchmark summary (sorted by ROC AUC):")
    print(view[cols].to_string(index=False, formatters={
        "samples": "{:.0f}".format,
        "accuracy": "{:.1%}".format,
        "roc_auc": "{:.2f}".format,
        "brier_score": "{:.3f}".format,
        "strategy_total_return": "{:.1%}".format,
        "strategy_sharpe": "{:.2f}".format,
        "strategy_max_drawdown": "{:.1%}".format,
        "strategy_average_exposure": "{:.1%}".format,
    }))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run walk-forward ML direction benchmark.")
    parser.add_argument("--symbols", default=None, help="Comma-separated symbol override, e.g. AAPL,BTC.")
    parser.add_argument("--horizon-days", type=int, default=5, help="Forward-return horizon for the target.")
    parser.add_argument("--min-train-days", type=int, default=252, help="Minimum rows before first prediction.")
    parser.add_argument("--train-window-days", type=int, default=756, help="Rolling training window length.")
    parser.add_argument("--retrain-every-days", type=int, default=63, help="Retraining cadence.")
    parser.add_argument("--fee-bps", type=float, default=10, help="Trading fee in basis points per turnover unit.")
    parser.add_argument("--slippage-bps", type=float, default=5, help="Slippage in basis points per turnover unit.")
    parser.add_argument("--initial-capital", type=float, default=10_000, help="Starting capital for strategy test.")
    args = parser.parse_args()

    cfg = load_asset_config()
    symbols = [s["ticker"] for s in cfg.get("stocks", [])] + [c["symbol"] for c in cfg.get("crypto", [])]
    if args.symbols:
        wanted = {s.strip().upper() for s in args.symbols.split(",") if s.strip()}
        symbols = [s for s in symbols if s in wanted]

    if not symbols:
        logger.error("No matching symbols found. Check --symbols or config/assets.yaml.")
        sys.exit(1)

    ensure_dir(ML_DIR)
    ensure_dir(PREDICTIONS_DIR)

    rows = []
    for symbol in symbols:
        logger.info("Running walk-forward ML benchmark for %s", symbol)
        row = run_symbol(
            symbol=symbol,
            horizon_days=args.horizon_days,
            min_train_days=args.min_train_days,
            train_window_days=args.train_window_days,
            retrain_every_days=args.retrain_every_days,
            fee_bps=args.fee_bps,
            slippage_bps=args.slippage_bps,
            initial_capital=args.initial_capital,
        )
        if row:
            rows.append(row)

    if not rows:
        logger.error("No ML benchmark rows produced. Run scoring first or lower --min-train-days.")
        sys.exit(1)

    summary = pd.DataFrame(rows).sort_values("symbol").reset_index(drop=True)
    summary_path = ML_DIR / "walkforward_summary.csv"
    summary.to_csv(summary_path, index=False)
    logger.info("Saved summary -> %s", summary_path)
    print_summary(summary)


if __name__ == "__main__":
    # Keep NumPy's printed warnings out of the user-facing CLI output; the
    # runner explicitly handles NaNs and single-class windows.
    np.seterr(all="ignore")
    main()


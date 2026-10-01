"""
metrics.py
==========
Performance metrics for daily long-only strategy backtests.

The functions here intentionally stay small and dependency-free. They
operate on pandas Series produced by src/backtest/run_backtests.py and
return plain dictionaries that are easy to write to CSV or compare in a
report.
"""
from __future__ import annotations

import math

import pandas as pd


def max_drawdown(equity_curve: pd.Series) -> float:
    """Return max drawdown as a negative fraction."""
    running_max = equity_curve.cummax()
    drawdown = equity_curve / running_max - 1
    return float(drawdown.min())


def summarize_performance(
    daily_returns: pd.Series,
    equity_curve: pd.Series,
    exposure: pd.Series,
    turnover: pd.Series,
    periods_per_year: int = 252,
) -> dict[str, float]:
    """
    Summarize daily strategy returns into standard backtest metrics.

    Returns are assumed to be net of fees/slippage. Risk-free rate is
    kept at zero for a transparent first baseline.
    """
    daily_returns = daily_returns.fillna(0)
    equity_curve = equity_curve.dropna()

    if equity_curve.empty:
        raise ValueError("equity_curve must contain at least one value")

    total_return = float(equity_curve.iloc[-1] / equity_curve.iloc[0] - 1)
    periods = max(len(daily_returns), 1)
    years = periods / periods_per_year
    cagr = (1 + total_return) ** (1 / years) - 1 if years > 0 and total_return > -1 else -1.0

    volatility = float(daily_returns.std() * math.sqrt(periods_per_year))
    sharpe = float((daily_returns.mean() / daily_returns.std()) * math.sqrt(periods_per_year)) if daily_returns.std() else 0.0

    return {
        "total_return": total_return,
        "cagr": cagr,
        "annualized_volatility": volatility,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown(equity_curve),
        "win_rate": float((daily_returns > 0).mean()),
        "average_exposure": float(exposure.mean()),
        "turnover": float(turnover.sum()),
        "days": float(periods),
    }


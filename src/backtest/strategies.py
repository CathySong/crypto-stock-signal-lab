"""
strategies.py
=============
Simple long-only baseline strategies used by the Day 4 backtest layer.

All functions return a daily target exposure from 0.0 to 1.0, where:
    0.0 = cash
    1.0 = fully invested in the asset

Positions are shifted by one day in the backtest runner before returns
are applied, so today's signal is only tradable on the next bar.
"""
from __future__ import annotations

import pandas as pd


def buy_and_hold(df: pd.DataFrame) -> pd.Series:
    """Fully invested after the first available row."""
    return pd.Series(1.0, index=df.index, name="buy_and_hold")


def ma_50_200_crossover(df: pd.DataFrame) -> pd.Series:
    """Long when SMA50 is above SMA200, otherwise cash."""
    signal = (df["sma_50"] > df["sma_200"]).astype(float)
    return signal.rename("ma_50_200_crossover")


def timing_score_rule(df: pd.DataFrame) -> pd.Series:
    """
    Long-only exposure based on Timing Score:
      - >= 70: fully invested
      - 50-70: half exposure
      - < 50: cash
    """
    score = df["timing_score"].fillna(0)
    signal = pd.Series(0.0, index=df.index, name="timing_score_rule")
    signal[score >= 70] = 1.0
    signal[(score >= 50) & (score < 70)] = 0.5
    return signal


def score_weighted_risk_sized(df: pd.DataFrame) -> pd.Series:
    """
    Uses the project's final_score and suggested_pct_of_account columns.

    This is intentionally conservative: the strategy only enters when
    final_score is at least 55, risk is not high, and the suggested
    position size is positive. The resulting exposure stays capped by
    the position-sizing layer (default 20% of account), leaving the rest
    in cash.
    """
    active = (df["final_score"].fillna(0) >= 55) & (df["risk_score"].fillna(100) < 75)
    exposure = df["suggested_pct_of_account"].where(active, 0.0).fillna(0.0)
    return exposure.clip(0, 1).rename("score_weighted_risk_sized")


STRATEGIES = {
    "buy_and_hold": buy_and_hold,
    "ma_50_200_crossover": ma_50_200_crossover,
    "timing_score_rule": timing_score_rule,
    "score_weighted_risk_sized": score_weighted_risk_sized,
}


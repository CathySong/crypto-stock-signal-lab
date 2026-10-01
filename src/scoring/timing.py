"""
timing.py
==========
Compute a rule-based "Timing Score" (0-100) from the technical indicators
produced by src/features/technical.py.

This is a baseline, not a prediction. It encodes a well-known textbook
heuristic for "is this a reasonable moment to consider entering, from a
pure technical-analysis standpoint" -- it does not know anything about
news, fundamentals, or forward-looking risk. See docs/scoring.md for the
full write-up of the formula and its limitations.

Score components (must sum to 100 at max):
    1. Trend alignment      (0-40 pts)
       +20 if price is above its 50-day moving average (short-term uptrend)
       +20 if the 50-day average is above the 200-day average
           (classic "golden cross" / intermediate uptrend)
    2. MACD momentum        (0-30 pts)
       Scaled by the MACD histogram's sign and magnitude relative to the
       asset's own recent histogram range, so it's comparable across
       assets with very different price scales.
    3. RSI positioning      (0-30 pts)
       Rewards RSI readings that suggest room to run without being
       overbought; penalizes chasing an already-overbought move.
       RSI <= 30  -> 30 pts (oversold, potential bounce)
       30 < RSI <= 50 -> scaled 20-30 pts
       50 < RSI <= 70 -> scaled 5-20 pts
       RSI > 70   -> scaled down to 0 pts (overbought, risky to chase)

Usage (as a library):
    from src.scoring.timing import compute_timing_score
    df = compute_timing_score(df)  # adds 'timing_score' column
"""
from __future__ import annotations

import pandas as pd


def _trend_component(df: pd.DataFrame) -> pd.Series:
    above_50 = (df["close_for_trend"] > df["sma_50"]).astype(float) * 20
    golden_cross = (df["sma_50"] > df["sma_200"]).astype(float) * 20
    return above_50 + golden_cross


def _macd_component(df: pd.DataFrame, window: int = 252) -> pd.Series:
    hist = df["macd_hist"]
    # Normalize histogram by its own trailing range so very different
    # price scales (e.g. BTC at $80k vs a $5 stock) are comparable.
    rolling_abs_max = hist.abs().rolling(window=window, min_periods=20).max()
    normalized = (hist / rolling_abs_max).clip(-1, 1).fillna(0)
    # Map [-1, 1] -> [0, 30], so a neutral/negative histogram gets 0-15
    # and a strong positive histogram gets up to 30.
    return ((normalized + 1) / 2 * 30).clip(0, 30)


def _rsi_component(df: pd.DataFrame) -> pd.Series:
    rsi = df["rsi_14"]
    score = pd.Series(index=df.index, dtype=float)

    oversold = rsi <= 30
    healthy_low = (rsi > 30) & (rsi <= 50)
    healthy_high = (rsi > 50) & (rsi <= 70)
    overbought = rsi > 70

    score[oversold] = 30
    score[healthy_low] = 20 + (rsi[healthy_low] - 30) / 20 * 10
    score[healthy_high] = 20 - (rsi[healthy_high] - 50) / 20 * 15
    score[overbought] = (5 - (rsi[overbought] - 70).clip(upper=30) / 30 * 5).clip(lower=0)

    return score


def compute_timing_score(df: pd.DataFrame, price_col: str = "adj_close") -> pd.DataFrame:
    """
    Add a 'timing_score' column (0-100) and a 'timing_label' column to a
    technical-indicator dataframe (output of src/features/technical.py).

    `price_col` should match whichever price column was used to compute
    sma_50/sma_200 in technical.py (adj_close for stocks, price_usd for
    crypto).
    """
    out = df.copy()
    out["close_for_trend"] = out[price_col]

    trend = _trend_component(out)
    macd_pts = _macd_component(out)
    rsi_pts = _rsi_component(out)

    out["timing_score"] = (trend + macd_pts + rsi_pts).clip(0, 100)
    out = out.drop(columns=["close_for_trend"])

    out["timing_label"] = pd.cut(
        out["timing_score"],
        bins=[-0.1, 30, 50, 70, 100],
        labels=["weak", "neutral", "favorable", "strong"],
    )
    return out

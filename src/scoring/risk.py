"""
risk.py
========
Compute a rule-based "Risk Score" (0-100, where 100 = highest risk) from
volatility, drawdown, and liquidity signals.

This score is deliberately kept separate from Timing and Value: a high
Timing Score or Value Score does not mean an asset is safe to bet a large
position on. Risk Score exists so the position-sizing layer can size
down exposure to genuinely volatile/illiquid assets even when the entry
signal looks good.

Components:
    1. Volatility        (0-50 pts)
       Annualized 30-day volatility, scaled against a fixed reference
       band (10% = very low risk, 120% = very high risk). Crypto will
       generally land higher on this axis than large-cap equities,
       which is expected and correct.
    2. Drawdown depth     (0-30 pts)
       How far below its own all-time high the asset currently sits.
       Deeper current drawdown = more risk of further downside/illiquidity
       stress, independent of whether it also looks "cheap" on the Value
       Score.
    3. Liquidity          (0-20 pts, crypto only; stocks default to 0
                           extra risk here since yfinance volume for
                           large caps/ETFs is reliably liquid)
       Low daily volume relative to market cap scores higher risk.

Usage (as a library):
    from src.scoring.risk import compute_risk_score
"""
from __future__ import annotations

import pandas as pd

# Reference bands for annualized volatility, expressed as decimals
# (0.10 = 10%). Chosen from rough real-world ranges: large-cap equities
# typically run 15-30% annualized, major crypto 40-100%+.
VOL_FLOOR = 0.10
VOL_CEILING = 1.20


def _volatility_component(df: pd.DataFrame) -> pd.Series:
    if "volatility_30d_annualized" not in df.columns:
        return pd.Series(0, index=df.index)
    vol = df["volatility_30d_annualized"]
    normalized = ((vol - VOL_FLOOR) / (VOL_CEILING - VOL_FLOOR)).clip(0, 1)
    return normalized * 50


def _drawdown_component(df: pd.DataFrame) -> pd.Series:
    if "drawdown_from_ath" not in df.columns:
        return pd.Series(0, index=df.index)
    # drawdown_from_ath is a negative fraction; deeper drawdown (more
    # negative, floor at -80%) -> higher risk points.
    dd = df["drawdown_from_ath"].clip(lower=-0.80, upper=0)
    return (dd / -0.80) * 30


def _liquidity_component(df: pd.DataFrame) -> pd.Series:
    if "volume_to_mcap_ratio" not in df.columns:
        return pd.Series(0, index=df.index)
    ratio = df["volume_to_mcap_ratio"].fillna(0)
    # Below 1% daily turnover is treated as the high-risk end; above 10%
    # daily turnover is treated as fully liquid (0 extra risk points).
    normalized = (1 - ((ratio - 0.01) / (0.10 - 0.01)).clip(0, 1))
    return normalized * 20


def compute_risk_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a 'risk_score' column (0-100, higher = riskier) and a
    'risk_label' column to a technical-indicator dataframe (output of
    src/features/technical.py). Works for both stocks and crypto; the
    liquidity component only contributes when volume_to_mcap_ratio is
    present (crypto only, added in technical.py).
    """
    out = df.copy()

    vol_pts = _volatility_component(out)
    dd_pts = _drawdown_component(out)
    liq_pts = _liquidity_component(out)

    out["risk_score"] = (vol_pts.fillna(0) + dd_pts.fillna(0) + liq_pts.fillna(0)).clip(0, 100)
    out["risk_label"] = pd.cut(
        out["risk_score"],
        bins=[-0.1, 25, 50, 75, 100],
        labels=["low", "moderate", "elevated", "high"],
    )
    return out

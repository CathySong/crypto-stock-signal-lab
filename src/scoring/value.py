"""
value.py
=========
Compute a rule-based "Value Score" (0-100) that answers: is this asset
cheap or expensive relative to common valuation yardsticks?

Two separate paths, because equities and crypto are valued completely
differently:

Equities (uses src/features/fundamentals.py output):
    - P/E ratio percentile vs. the asset's own trailing history
      (cheap relative to its own past, not an absolute "good" P/E)
    - P/B ratio percentile, same approach
    - FCF yield level (higher is better, uses absolute bands since FCF
      yield is already a normalized, comparable-across-assets metric)

Crypto (uses src/features/technical.py output; crypto has no
cash-flow-based valuation, so this is explicitly a relative-price-level
proxy, not a "fair value" model):
    - Drawdown from all-time-high (deeper drawdown = scores higher,
      i.e. "closer to historical cheap", with explicit caveats that a
      large drawdown can also mean a broken asset, not a bargain)
    - Price percentile within its own trailing 1-year range (lower in
      its own range scores higher)

This is intentionally simple and self-referential (an asset is scored
against its *own* history, not against other assets) because cross-asset
"fair value" comparisons for crypto require assumptions this project
does not want to silently bake in. See docs/scoring.md for the full
write-up and limitations.

Usage (as a library):
    from src.scoring.value import compute_equity_value_score, compute_crypto_value_score
"""
from __future__ import annotations

import pandas as pd


def _percentile_rank(series: pd.Series, window: int = 756) -> pd.Series:
    """
    Rolling percentile rank of the latest value within its own trailing
    `window` history (~3 trading years at 252/yr). Returns a 0-1 value;
    NaN until enough history has accumulated.
    """

    def _rank_last(x: pd.Series) -> float:
        last = x.iloc[-1]
        if pd.isna(last):
            return float("nan")
        valid = x.dropna()
        if len(valid) < 2:
            return float("nan")
        return (valid <= last).sum() / len(valid)

    return series.rolling(window=window, min_periods=60).apply(_rank_last, raw=False)


def compute_equity_value_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a 'value_score' (0-100) and 'value_label' column to a
    fundamentals-ratio dataframe (output of src/features/fundamentals.py).

    Lower P/E and P/B *percentile* (cheap vs. own history) scores higher.
    Higher FCF yield scores higher on an absolute band, since FCF yield
    is already comparable across assets without needing a historical
    baseline.
    """
    out = df.copy()

    pe_pctile = _percentile_rank(out["pe_ratio"]) if "pe_ratio" in out.columns else pd.Series(dtype=float)
    pb_pctile = _percentile_rank(out["pb_ratio"]) if "pb_ratio" in out.columns else pd.Series(dtype=float)

    # Invert percentile: being in the bottom 10% of historical P/E (i.e.
    # pctile ~0.10) should score close to 100 on "cheapness".
    pe_score = (1 - pe_pctile) * 50 if len(pe_pctile) else pd.Series(0, index=out.index)
    pb_score = (1 - pb_pctile) * 30 if len(pb_pctile) else pd.Series(0, index=out.index)

    if "fcf_yield" in out.columns:
        # Absolute bands: FCF yield < 0% -> 0 pts, 0-8% scaled linearly,
        # >= 8% -> full 20 pts. These thresholds are a simplification;
        # see docs/scoring.md.
        fcf_score = (out["fcf_yield"].clip(lower=0, upper=0.08) / 0.08 * 20)
    else:
        fcf_score = pd.Series(0, index=out.index)

    out["value_score"] = (pe_score.fillna(0) + pb_score.fillna(0) + fcf_score.fillna(0)).clip(0, 100)
    out["value_label"] = pd.cut(
        out["value_score"],
        bins=[-0.1, 30, 50, 70, 100],
        labels=["expensive", "fair", "attractive", "cheap"],
    )
    return out


def compute_crypto_value_score(df: pd.DataFrame, price_col: str = "price_usd") -> pd.DataFrame:
    """
    Add a 'value_score' (0-100) and 'value_label' column to a
    technical-indicator dataframe for a crypto asset (output of
    src/features/technical.py).

    IMPORTANT CAVEAT: this is a relative-price-level proxy, not a
    fundamental valuation. A coin deep in drawdown may be "cheap" in the
    sense of trading far below its own prior high, or it may be reacting
    to a genuine deterioration (lost relevance, hack, delisting risk,
    etc.). Treat this score as one input among several, never as
    standalone advice to buy a dip.
    """
    out = df.copy()

    # drawdown_from_ath is already a negative fraction (e.g. -0.60).
    # More negative (deeper drawdown) -> higher score, capped at -80%.
    if "drawdown_from_ath" in out.columns:
        drawdown_score = (out["drawdown_from_ath"].clip(lower=-0.80, upper=0) / -0.80 * 60)
    else:
        drawdown_score = pd.Series(0, index=out.index)

    price_pctile = _percentile_rank(out[price_col], window=365) if price_col in out.columns else pd.Series(dtype=float)
    price_range_score = (1 - price_pctile) * 40 if len(price_pctile) else pd.Series(0, index=out.index)

    out["value_score"] = (drawdown_score.fillna(0) + price_range_score.fillna(0)).clip(0, 100)
    out["value_label"] = pd.cut(
        out["value_score"],
        bins=[-0.1, 30, 50, 70, 100],
        labels=["expensive", "fair", "attractive", "cheap"],
    )
    return out

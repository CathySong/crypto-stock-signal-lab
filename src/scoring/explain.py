"""
explain.py
===========
Turn the three numeric scores (Timing, Value, Risk) for a single asset on
a single date into a short, beginner-friendly explanation in plain
English. This is the human-readable layer the README promises: not just
a number, but a reason.

Deliberately template-based (no LLM call) so output is deterministic,
free to run at scale, and auditable -- every sentence traces back to a
specific score band defined in src/scoring/timing.py, value.py, risk.py.

Usage (as a library):
    from src.scoring.explain import explain_asset
    text = explain_asset(
        symbol="AAPL",
        timing_score=72, timing_label="favorable",
        value_score=38, value_label="fair",
        risk_score=22, risk_label="low",
    )
"""
from __future__ import annotations

from dataclasses import dataclass


TIMING_PHRASES = {
    "strong": "Momentum and trend indicators are strongly aligned to the upside",
    "favorable": "Momentum and trend indicators lean positive",
    "neutral": "Momentum and trend indicators are mixed, with no clear direction",
    "weak": "Momentum and trend indicators are leaning negative",
}

VALUE_PHRASES = {
    "cheap": "it looks cheap relative to its own history",
    "attractive": "it looks reasonably priced relative to its own history",
    "fair": "it's trading close to its typical historical valuation",
    "expensive": "it looks expensive relative to its own history",
    "unavailable": "valuation data is not available for this asset type",
}

RISK_PHRASES = {
    "low": "with relatively low volatility and drawdown risk",
    "moderate": "with moderate volatility; expect noticeable price swings",
    "elevated": "with elevated volatility and/or a deep current drawdown",
    "high": "with high volatility and/or thin liquidity -- size positions accordingly",
}


@dataclass
class AssetExplanation:
    symbol: str
    timing_score: float
    value_score: float
    risk_score: float
    summary: str
    detail: str


def _overall_lean(timing_score: float, value_score: float | None, risk_score: float) -> str:
    """
    One-line summary combining all three scores. Intentionally
    conservative: a high Timing Score alone never says "buy" outright
    if Risk Score is also high.
    """
    if risk_score >= 75:
        return "High risk reading -- any entry here should use a small position size regardless of timing/value."
    if value_score is None:
        if timing_score >= 70:
            return "Technical setup looks favorable, but no valuation score is available -- treat this as a momentum-only signal."
        if timing_score < 40:
            return "Technical setup is weak and no valuation score is available to offset it."
        return "Mixed technical setup with no valuation score available; wait for a clearer signal or use a smaller exploratory size."
    if timing_score >= 70 and value_score >= 60:
        return "Technical setup and valuation both look favorable; a reasonable moment to research further."
    if timing_score >= 70 and value_score < 40:
        return "Technical setup looks favorable but the asset looks expensive vs. its own history -- momentum play, not a value play."
    if timing_score < 40 and value_score >= 60:
        return "Looks cheap vs. its own history but momentum hasn't turned yet -- could be early, could be a falling knife."
    if timing_score < 40 and value_score < 40:
        return "Neither the technical setup nor the valuation picture looks favorable right now."
    return "Mixed signals -- no strong technical or valuation edge in either direction at the moment."


def explain_asset(
    symbol: str,
    timing_score: float,
    timing_label: str,
    value_score: float,
    value_label: str,
    risk_score: float,
    risk_label: str,
) -> AssetExplanation:
    timing_phrase = TIMING_PHRASES.get(str(timing_label), "Momentum signals are unclear")
    value_phrase = VALUE_PHRASES.get(str(value_label), "its valuation versus history is unclear")
    risk_phrase = RISK_PHRASES.get(str(risk_label), "with an unclear risk profile")

    summary = _overall_lean(timing_score, value_score, risk_score)
    value_display = "n/a" if value_score is None else f"{value_score:.0f}/100"

    detail = (
        f"{symbol}: Timing {timing_score:.0f}/100 ({timing_label}), "
        f"Value {value_display} ({value_label}), "
        f"Risk {risk_score:.0f}/100 ({risk_label}). "
        f"{timing_phrase}, and {value_phrase}, {risk_phrase}."
    )

    return AssetExplanation(
        symbol=symbol,
        timing_score=timing_score,
        value_score=0 if value_score is None else value_score,
        risk_score=risk_score,
        summary=summary,
        detail=detail,
    )

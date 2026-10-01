"""
position_sizing.py
====================
Turn a Risk Score into a suggested position size, using a fixed-fractional
risk model (a standard, conservative approach used in trading education,
not an optimization or prediction).

Core idea: decide in advance how much of the account you're willing to
lose if the trade goes wrong (risk_per_trade_pct of total account), and a
stop-loss distance (how far below entry you'd exit to cap the loss). The
dollar amount to risk divided by the stop distance gives the position
size. A higher Risk Score score tightens the stop distance and/or caps
the position size further, so riskier assets automatically get smaller
suggested allocations for the same account-risk tolerance.

This is explicitly NOT:
    - A guarantee of limiting losses to the stated amount (slippage, gaps,
      and liquidity can cause real losses to exceed the planned risk).
    - Leveraged position sizing (assumes cash/spot, no margin).
    - A recommendation to risk any specific amount; the "risk_per_trade_pct"
      and "max_position_pct" inputs are conservative defaults a beginner
      can start from, not a personalized recommendation.

Usage (as a library):
    from src.scoring.position_sizing import suggest_position_size

    result = suggest_position_size(
        account_value=10_000,
        entry_price=150.0,
        risk_score=35,          # from src.scoring.risk.compute_risk_score
        risk_per_trade_pct=0.01,  # risk 1% of account per trade (common beginner default)
    )
"""
from __future__ import annotations

from dataclasses import dataclass

# Conservative beginner defaults. These are starting points for education,
# not personalized financial advice -- see module docstring.
DEFAULT_RISK_PER_TRADE_PCT = 0.01   # risk 1% of account value per trade
DEFAULT_MAX_POSITION_PCT = 0.20     # never suggest more than 20% of account in one asset
MIN_STOP_DISTANCE_PCT = 0.03        # never suggest a stop tighter than 3% (avoids noise stop-outs)
MAX_STOP_DISTANCE_PCT = 0.25        # cap stop distance at 25% even for very high risk scores


@dataclass
class PositionSizeResult:
    account_value: float
    entry_price: float
    risk_score: float
    risk_per_trade_pct: float
    stop_distance_pct: float
    stop_loss_price: float
    dollar_risk: float
    suggested_shares_or_units: float
    suggested_dollar_amount: float
    suggested_pct_of_account: float
    capped_by_max_position: bool
    explanation: str


def _stop_distance_from_risk_score(risk_score: float) -> float:
    """
    Map a 0-100 Risk Score to a stop-loss distance percentage.

    Higher risk score (more volatile/illiquid asset) -> wider stop,
    because a tight stop on a volatile asset gets triggered by normal
    noise, not a real trend change. Wider stop means a smaller position
    size for the same dollar risk, which is the actual risk-reduction
    mechanism here.
    """
    risk_score = max(0.0, min(100.0, risk_score))
    span = MAX_STOP_DISTANCE_PCT - MIN_STOP_DISTANCE_PCT
    return MIN_STOP_DISTANCE_PCT + (risk_score / 100.0) * span


def suggest_position_size(
    account_value: float,
    entry_price: float,
    risk_score: float,
    risk_per_trade_pct: float = DEFAULT_RISK_PER_TRADE_PCT,
    max_position_pct: float = DEFAULT_MAX_POSITION_PCT,
) -> PositionSizeResult:
    """
    Compute a suggested position size using fixed-fractional risk sizing.

    Formula:
        stop_distance_pct = f(risk_score)   # wider for riskier assets
        stop_loss_price   = entry_price * (1 - stop_distance_pct)
        dollar_risk        = account_value * risk_per_trade_pct
        suggested_shares   = dollar_risk / (entry_price - stop_loss_price)
        suggested_dollars  = suggested_shares * entry_price

    The suggested dollar amount is then capped at `max_position_pct` of
    the account, which matters most for low-risk-score (tight-stop)
    assets where the risk formula alone could suggest an oversized
    position.
    """
    if account_value <= 0:
        raise ValueError("account_value must be positive")
    if entry_price <= 0:
        raise ValueError("entry_price must be positive")

    stop_distance_pct = _stop_distance_from_risk_score(risk_score)
    stop_loss_price = entry_price * (1 - stop_distance_pct)
    dollar_risk = account_value * risk_per_trade_pct
    per_unit_risk = entry_price - stop_loss_price

    raw_shares = dollar_risk / per_unit_risk
    raw_dollar_amount = raw_shares * entry_price

    max_dollar_amount = account_value * max_position_pct
    capped = raw_dollar_amount > max_dollar_amount

    final_dollar_amount = min(raw_dollar_amount, max_dollar_amount)
    final_shares = final_dollar_amount / entry_price

    explanation = (
        f"Risking {risk_per_trade_pct * 100:.1f}% of your ${account_value:,.0f} account "
        f"(${dollar_risk:,.2f}) with a stop-loss {stop_distance_pct * 100:.1f}% below entry "
        f"(${stop_loss_price:,.2f}) suggests a position of about "
        f"{final_shares:,.4f} units (~${final_dollar_amount:,.2f}, "
        f"{final_dollar_amount / account_value * 100:.1f}% of account)."
    )
    if capped:
        explanation += (
            f" This was capped at the {max_position_pct * 100:.0f}% max-position-size "
            f"guardrail, since the raw risk-based size would have been larger."
        )

    return PositionSizeResult(
        account_value=account_value,
        entry_price=entry_price,
        risk_score=risk_score,
        risk_per_trade_pct=risk_per_trade_pct,
        stop_distance_pct=stop_distance_pct,
        stop_loss_price=stop_loss_price,
        dollar_risk=dollar_risk,
        suggested_shares_or_units=final_shares,
        suggested_dollar_amount=final_dollar_amount,
        suggested_pct_of_account=final_dollar_amount / account_value,
        capped_by_max_position=capped,
        explanation=explanation,
    )

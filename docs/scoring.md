# Scoring design

This document explains the three scores produced by `src/scoring/` and
their known limitations. All scores are rule-based (no machine learning)
by design for this first version, so every number can be traced back to
a specific formula instead of a black box.

## Timing Score (0-100)

Source: `src/scoring/timing.py`

Answers: "Do current trend and momentum indicators favor an entry, from a
pure technical-analysis standpoint?"

| Component | Points | Logic |
|---|---|---|
| Trend alignment | 0-40 | +20 if price > SMA50, +20 if SMA50 > SMA200 (golden cross) |
| MACD momentum | 0-30 | MACD histogram normalized against its own trailing range |
| RSI positioning | 0-30 | Oversold (<=30) scores highest; overbought (>70) scores near zero |

**Limitations:** purely technical, has no concept of news, earnings
events, or macro conditions. A "strong" Timing Score during a broad
market selloff is still a bad entry if the selloff continues.

## Value Score (0-100)

Source: `src/scoring/value.py`

Equities: percentile rank of current P/E and P/B against the asset's own
trailing ~3-year history (cheap/expensive relative to itself, not to
peers), plus an absolute FCF yield band.

Crypto: no cash-flow-based valuation exists, so this is an explicit
**relative-price-level proxy** — depth of drawdown from all-time-high,
plus percentile position within the trailing 1-year price range.

**Limitations:** equities score is self-referential (doesn't compare
against sector peers or the market). Crypto score can rate a
structurally broken project as "cheap" purely because it has crashed —
this is called out directly in the code docstring and should never be
read as "this coin is undervalued" without further research.

## Risk Score (0-100, higher = riskier)

Source: `src/scoring/risk.py`

| Component | Points | Logic |
|---|---|---|
| Volatility | 0-50 | 30-day annualized volatility, scaled 10%-120% |
| Drawdown depth | 0-30 | How far below all-time-high, scaled 0%-80% |
| Liquidity (crypto only) | 0-20 | Daily volume / market cap; thin liquidity scores higher risk |

Risk Score is intentionally independent of Timing/Value — a "strong"
timing signal on a high-risk asset should still get a smaller suggested
position size, which is what `position_sizing.py` uses it for.

## Position sizing

Source: `src/scoring/position_sizing.py`

Standard fixed-fractional risk model used in trading education:

1. Decide how much of the account to risk per trade (default 1%).
2. Risk Score sets the stop-loss distance (3%-25%): riskier assets get a
   wider stop, because a tight stop on a volatile asset just gets
   stopped out by noise.
3. Position size = (account risk $) / (stop distance in $).
4. Hard cap at 20% of account in any single position, regardless of
   what the formula above suggests.

**This is not a guarantee.** Gaps, slippage, and illiquidity can cause
real losses to exceed the planned risk amount. It is a starting
discipline for beginners, not a promise.

`src/scoring/run_scores.py` also writes practical position-sizing columns
into each `data/processed/scores/{SYMBOL}.csv` file:

- `final_score`: weighted blend of Timing, Value, and inverse Risk
- `stop_distance_pct`
- `stop_loss_price`
- `suggested_pct_of_account`
- `suggested_dollar_amount`
- `suggested_units`

The default account value is `$10,000` for educational examples and can
be changed with `--account-value`. Missing valuation scores (for ETFs,
for example) remain blank, while the final score treats them as neutral
so the output does not falsely imply valuation data exists.

## Explanations

Source: `src/scoring/explain.py`

Template-based (no LLM call), so every sentence in the output maps
directly to a specific score band defined above. This keeps the
explanation reproducible and auditable — rerunning the same inputs
always produces the same explanation text.

## Overall philosophy

- Baselines before ML: this scoring layer is the documented baseline
  every future ML model in this repo must beat.
- Every score should be explainable in one sentence a beginner can
  understand.
- Position sizing, never price targets: nothing in this repo promises a
  specific future return.

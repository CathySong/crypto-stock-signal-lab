# Backtesting design

This document explains the Day 4 backtest layer in `src/backtest/`.

The goal is not to prove a strategy is profitable. The goal is to make
the scoring system testable against simple baselines before any ML model
is added.

## Runner

Source: `src/backtest/run_backtests.py`

Usage:

```bash
python -m src.scoring.run_scores
python -m src.backtest.run_backtests
```

Optional overrides:

```bash
python -m src.backtest.run_backtests --symbols AAPL,BTC
python -m src.backtest.run_backtests --fee-bps 10 --slippage-bps 5
python -m src.backtest.run_backtests --initial-capital 10000
```

## Strategies

Source: `src/backtest/strategies.py`

Implemented baselines:

- `buy_and_hold`: fully invested after the first row
- `ma_50_200_crossover`: long when SMA50 is above SMA200
- `timing_score_rule`: 100% exposure when Timing Score >= 70, 50% when
  Timing Score is 50-70, cash below 50
- `score_weighted_risk_sized`: enters only when Final Score is at least
  55 and Risk Score is below 75, then uses the project's position-sizing
  output as the actual exposure

## Lookahead control

Signals are generated from each row's available indicators, then shifted
by one day before returns are applied. In other words, today's score can
only affect tomorrow's position. This avoids the common mistake of using
same-day closing information to trade the same close.

## Costs

Default cost assumption:

- 10 bps trading fee
- 5 bps slippage

Costs are applied on turnover, not on every day. A strategy that sits in
cash or holds steady pays no extra daily cost.

## Metrics

Source: `src/backtest/metrics.py`

Each strategy produces:

- total return
- CAGR
- annualized volatility
- Sharpe ratio
- max drawdown
- win rate
- average exposure
- total turnover
- ending equity

Generated files:

- `data/processed/backtests/summary.csv`
- `data/processed/backtests/equity_curves/{SYMBOL}_{STRATEGY}.csv`

These are generated artifacts and are not committed.

## Important limitations

- The first version is a single-asset backtest, not a portfolio
  optimizer.
- Crypto history currently comes from the CoinGecko free tier, so the
  available window is 365 days unless another source is added.
- Equity histories are much longer than crypto histories, so raw results
  should not be compared across asset classes without considering date
  range.
- The `score_weighted_risk_sized` strategy intentionally leaves unused
  capital in cash. Its lower return and lower drawdown are both expected.
- This is still a research baseline, not a live trading system.


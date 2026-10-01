# ML benchmark design

This document explains the first machine-learning layer in `src/ml/`.

The goal is deliberately modest: estimate the probability that an asset's
forward return will be positive over a fixed horizon, then compare that
probability signal against the rule-based baselines. It is a benchmark,
not a promise that the model can predict markets.

## Runner

Source: `src/ml/run_walkforward.py`

Usage:

```bash
python -m src.scoring.run_scores
python -m src.ml.run_walkforward
```

Optional overrides:

```bash
python -m src.ml.run_walkforward --symbols AAPL,BTC
python -m src.ml.run_walkforward --horizon-days 5
python -m src.ml.run_walkforward --min-train-days 252
```

## Model

Source: `src/ml/logistic.py`

The first model is a small in-repo logistic regression implemented with
NumPy. That keeps the benchmark lightweight and auditable before the
project adds heavier dependencies such as scikit-learn, XGBoost, or
LightGBM.

Features come from the existing scoring pipeline:

- Timing Score
- Value Score
- Risk Score
- Final Score
- RSI
- MACD histogram
- annualized volatility
- drawdown from all-time high
- suggested position size
- trend flags
- crypto liquidity ratio when available

Missing ETF valuation scores are treated as neutral for modeling, the
same way the scoring layer handles them.

## Walk-forward protocol

The model uses rolling walk-forward evaluation:

1. Train only on historical rows before the prediction block.
2. Predict the next block.
3. Move forward and retrain.

Defaults:

- target horizon: 5 days
- first prediction after 252 rows
- training window: 756 rows
- retrain cadence: 63 rows

This avoids random train/test splits, which are usually misleading for
time-series data.

## Output

Generated files:

- `data/processed/ml/walkforward_summary.csv`
- `data/processed/ml/predictions/{SYMBOL}_predictions.csv`

These are generated artifacts and are not committed.

The summary includes:

- accuracy
- precision
- recall
- ROC AUC
- Brier score
- probability-signal strategy return
- Sharpe
- max drawdown
- average exposure

## Trading rule for the probability signal

The probability backtest uses a simple long-only rule:

- probability >= 0.58: fully invested
- probability 0.52-0.58: half exposure
- probability < 0.52: cash

Like the Day 4 backtest, exposure is shifted by one day before returns
are applied and transaction costs are charged on turnover.

## Limitations

- Logistic regression is a linear baseline, not a state-of-the-art
  market model.
- The feature set is intentionally small and explainable.
- Crypto still has a shorter 365-day CoinGecko window in the free data
  path.
- AUC or accuracy above 50% is not enough; the probability signal must
  also survive fees, slippage, and drawdown analysis.


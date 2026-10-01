# Static dashboard

The dashboard turns the generated research outputs into a single offline
HTML report for quick review.

## Runner

Source: `src/dashboard/build_static.py`

Usage:

```bash
python -m src.scoring.run_scores
python -m src.backtest.run_backtests
python -m src.ml.run_walkforward
python -m src.dashboard.build_static
```

Optional overrides:

```bash
python -m src.dashboard.build_static --symbols AAPL,BTC
python -m src.dashboard.build_static --output reports/dashboard.html
```

## Inputs

The dashboard reads the existing generated CSV artifacts:

- `data/processed/scores/{SYMBOL}.csv`
- `data/processed/backtests/summary.csv`
- `data/processed/backtests/equity_curves/{SYMBOL}_{STRATEGY}.csv`
- `data/processed/ml/walkforward_summary.csv`

## Output

Default output:

- `data/processed/dashboard/index.html`

The HTML file is generated and intentionally ignored by Git, the same way
raw data, processed features, backtest summaries, and ML predictions are.
It includes latest Timing/Value/Risk scores, position-sizing fields,
best backtest strategy by Sharpe, ML benchmark metrics, and embedded SVG
sparklines for equity curves.


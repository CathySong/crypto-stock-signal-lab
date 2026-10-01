# Research watchlist

The watchlist report is the final v1 layer: it turns the separate scoring,
backtesting, and ML outputs into one ranked research queue.

## Runner

Source: `src/reports/build_watchlist.py`

Usage:

```bash
python -m src.scoring.run_scores
python -m src.backtest.run_backtests
python -m src.ml.run_walkforward
python -m src.reports.build_watchlist
```

Optional overrides:

```bash
python -m src.reports.build_watchlist --symbols AAPL,BTC
python -m src.reports.build_watchlist --csv-output reports/watchlist.csv
```

## Inputs

The report reads:

- `data/processed/scores/{SYMBOL}.csv`
- `data/processed/backtests/summary.csv`
- `data/processed/ml/walkforward_summary.csv`

## Outputs

Generated files:

- `data/processed/reports/watchlist.csv`
- `data/processed/reports/watchlist.md`

Both are generated artifacts and are ignored by Git.

To upload the generated watchlist to Supabase after creating the
`research_watchlist` table from `sql/001_create_tables.sql`:

```bash
python -m src.data.upload_to_supabase --only watchlist
```

## Ranking logic

The `priority_score` is a transparent blend:

- 45% latest Final Score
- 25% inverse Risk Score
- 20% best baseline-strategy Sharpe ratio
- 10% ML ROC AUC

The report also assigns a plain-language `research_action`:

- `Research now`
- `Watch closely`
- `Review`
- `Wait`
- `Deprioritize`

These are research triage labels, not trading instructions.

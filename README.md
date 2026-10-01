# crypto-stock-signal-lab

An open-source research toolkit for building explainable entry-timing,
valuation, and risk signals across crypto and equities, using only public
market data.

This is **not** a "predict tomorrow's price" model. It is a small, testable
pipeline that turns technical indicators, fundamentals, and public market
data into three human-readable scores per asset:

- **Timing Score** — trend + momentum + oversold/overbought signal
- **Value Score** — is this asset cheap or expensive relative to its own
  history and peers (PE/PB/FCF for equities, market-cap/drawdown/liquidity
  for crypto)
- **Risk Score** — volatility, drawdown, leverage, liquidity

...plus a plain-language explanation and a suggested position size based on
account risk, not "buy this now."

> Educational / research project. Not financial advice. Nothing here is a
> signal to trade with real capital before you've run it through paper
> trading and understood the assumptions.

## Why this exists

Most public crypto/stock "prediction" repos on GitHub do one of two things:
predict a raw price number (which ages badly and is hard to trust), or hide
every assumption behind a black-box model. This project aims for the
opposite: simple, inspectable baselines first, ML/DL models compared
against those baselines, and a walk-forward backtest so every score can be
audited instead of taken on faith.

## Project status

Early stage / actively under construction. Current milestone: a full
baseline research pipeline: data ingestion, feature engineering, scoring,
position sizing, and backtesting across the v1 asset universe.

## Asset universe (v1)

Kept intentionally small while the pipeline stabilizes:

| Type   | Assets           |
|--------|------------------|
| Crypto | BTC, ETH, SOL    |
| Stocks | AAPL, NVDA, SPY  |

Defined in [`config/assets.yaml`](config/assets.yaml).

## Data sources

| Domain              | Source                          | Notes |
|----------------------|----------------------------------|-------|
| Stock/ETF prices     | [yfinance](https://ranaroussi.github.io/yfinance/reference/api/yfinance.download.html) | Daily OHLCV via Yahoo Finance; free, no key |
| Equity fundamentals  | [SEC EDGAR companyfacts API](https://www.sec.gov/search-filings/edgar-application-programming-interfaces) | XBRL facts: revenue, net income, EPS, assets, liabilities, R&D spend |
| Crypto market data   | [CoinGecko API](https://www.coingecko.com/en/api) | `/coins/{id}/market_chart`: price, market cap, volume |

All three fetchers live in `src/data/` and write flat CSV/JSONL files to
`data/raw/` (not committed — see `.gitignore`).

## Quickstart

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# edit .env: set SEC_USER_AGENT to "Your Name your-email@example.com"
# (SEC requires a real contact identifier on every request)

# Stock/ETF daily prices
python -m src.data.fetch_stocks --period max

# Equity fundamentals (SEC XBRL facts)
python -m src.data.fetch_fundamentals

# Crypto market data
python -m src.data.fetch_crypto --days 365

# Feature engineering, scoring, and backtests
python -m src.features.technical
python -m src.features.fundamentals
python -m src.scoring.run_scores
python -m src.backtest.run_backtests
python -m src.ml.run_walkforward
python -m src.dashboard.build_static

# Optional: push raw + processed outputs to Supabase
python -m src.data.upload_to_supabase --only scores,backtests,ml

# Local smoke tests
python -m unittest discover -s tests
```

Each script also accepts a `--tickers` / `--coins` override if you want to
pull a different asset than what's in `config/assets.yaml`.

## Roadmap

- [x] Data ingestion: stock prices, equity fundamentals, crypto market data
- [x] Feature engineering: technical indicators (RSI, MACD, moving averages)
      and valuation ratios (PE, PB, FCF yield)
- [x] Rule-based baseline strategies (MA crossover, buy & hold, score-based)
- [x] Timing / Value / Risk scoring layer with plain-language explanations
- [x] Position sizing helper (account risk % + stop distance -> suggested
      dollar allocation)
- [x] Daily backtest framework with fees, slippage, and no-lookahead signals
- [x] ML classifier (direction / probability of positive return) benchmarked
      against the rule-based baselines
- [x] Static dashboard for browsing per-asset scores, backtest reports, and
      ML benchmark results
- [x] Supabase upload support for processed scores, backtest summaries, and
      ML benchmark results
- [x] CI smoke tests for core scoring, ML, backtest, and upload helpers

## Design principles

1. **Baselines before ML.** Every model has to beat a documented rule-based
   baseline, or it doesn't ship.
2. **Explainable over clever.** Each score ships with a plain-language
   reason, not just a number.
3. **Position sizing, not price targets.** The goal is risk-aware entry
   sizing, never a promise of returns.
4. **Public data only, clearly attributed.** No paid data feeds required to
   reproduce results.

## Disclaimer

This project is for educational and research purposes only. It does not
constitute financial advice, and nothing produced by this codebase should
be used to make real trading decisions without independent due diligence
and, ideally, a period of paper trading first.

## License

MIT — see [LICENSE](LICENSE).

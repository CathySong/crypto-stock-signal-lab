-- crypto-stock-signal-lab: initial schema
-- Run this once in Supabase SQL Editor (Project > SQL Editor > New query).

create table if not exists stock_prices (
    ticker      text not null,
    date        date not null,
    open        numeric,
    high        numeric,
    low         numeric,
    close       numeric,
    adj_close   numeric,
    volume      bigint,
    primary key (ticker, date)
);

create table if not exists stock_fundamentals (
    id       bigint generated always as identity primary key,
    ticker   text not null,
    concept  text not null,
    unit     text,
    start_date date,
    end_date   date,
    val      numeric,
    fy       integer,
    fp       text,
    form     text,
    filed    date,
    frame    text,
    unique (ticker, concept, unit, end_date, form, fy, fp)
);

create table if not exists crypto_prices (
    symbol         text not null,
    date           date not null,
    price_usd      numeric,
    market_cap_usd numeric,
    volume_usd     numeric,
    primary key (symbol, date)
);

create table if not exists asset_scores (
    symbol                   text not null,
    date                     date not null,
    price                    numeric,
    timing_score             numeric,
    timing_label             text,
    value_score              numeric,
    value_label              text,
    risk_score               numeric,
    risk_label               text,
    final_score              numeric,
    stop_distance_pct        numeric,
    stop_loss_price          numeric,
    suggested_pct_of_account numeric,
    suggested_dollar_amount  numeric,
    suggested_units          numeric,
    primary key (symbol, date)
);

create table if not exists backtest_results (
    symbol                text not null,
    strategy              text not null,
    start_date            date not null,
    end_date              date not null,
    fee_bps               numeric not null,
    slippage_bps          numeric not null,
    initial_capital       numeric,
    ending_equity         numeric,
    total_return          numeric,
    cagr                  numeric,
    annualized_volatility numeric,
    sharpe                numeric,
    max_drawdown          numeric,
    win_rate              numeric,
    average_exposure      numeric,
    turnover              numeric,
    days                  numeric,
    primary key (symbol, strategy, start_date, end_date, fee_bps, slippage_bps)
);

create table if not exists ml_walkforward_results (
    symbol                    text not null,
    horizon_days              integer not null,
    features                  numeric,
    samples                   numeric,
    positive_rate             numeric,
    accuracy                  numeric,
    precision                 numeric,
    recall                    numeric,
    brier_score               numeric,
    roc_auc                   numeric,
    strategy_total_return     numeric,
    strategy_cagr             numeric,
    strategy_sharpe           numeric,
    strategy_max_drawdown     numeric,
    strategy_average_exposure numeric,
    strategy_turnover         numeric,
    ending_equity             numeric,
    average_probability_up    numeric,
    primary key (symbol, horizon_days)
);

create table if not exists research_watchlist (
    symbol                    text not null,
    date                      date not null,
    research_action           text,
    priority_score            numeric,
    final_score               numeric,
    timing_score              numeric,
    value_score               numeric,
    risk_score                numeric,
    price                     numeric,
    suggested_pct_of_account  numeric,
    stop_distance_pct         numeric,
    best_backtest_strategy    text,
    best_backtest_sharpe      numeric,
    best_backtest_return      numeric,
    best_backtest_drawdown    numeric,
    ml_auc                    numeric,
    ml_strategy_return        numeric,
    flags                     text,
    primary key (symbol, date)
);

-- Helpful indexes for time-series lookups
create index if not exists idx_stock_prices_date on stock_prices (date);
create index if not exists idx_crypto_prices_date on crypto_prices (date);
create index if not exists idx_stock_fundamentals_ticker on stock_fundamentals (ticker);
create index if not exists idx_asset_scores_date on asset_scores (date);
create index if not exists idx_backtest_results_symbol on backtest_results (symbol);
create index if not exists idx_ml_walkforward_results_symbol on ml_walkforward_results (symbol);
create index if not exists idx_research_watchlist_date on research_watchlist (date);

-- Row Level Security: enabled by default on new Supabase projects.
-- Server-side scripts use the service role key, which bypasses RLS
-- automatically, so no public read/write policies are required here.
alter table stock_prices enable row level security;
alter table stock_fundamentals enable row level security;
alter table crypto_prices enable row level security;
alter table asset_scores enable row level security;
alter table backtest_results enable row level security;
alter table ml_walkforward_results enable row level security;
alter table research_watchlist enable row level security;

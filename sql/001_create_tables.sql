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

-- Helpful indexes for time-series lookups
create index if not exists idx_stock_prices_date on stock_prices (date);
create index if not exists idx_crypto_prices_date on crypto_prices (date);
create index if not exists idx_stock_fundamentals_ticker on stock_fundamentals (ticker);

-- Row Level Security: enabled by default on new Supabase projects.
-- Server-side scripts use the service role key, which bypasses RLS
-- automatically, so no public read/write policies are required here.
alter table stock_prices enable row level security;
alter table stock_fundamentals enable row level security;
alter table crypto_prices enable row level security;

"""
technical.py
=============
Compute technical indicators (moving averages, RSI, MACD, volatility,
drawdown) from the raw OHLCV price files produced by
src/data/fetch_stocks.py and src/data/fetch_crypto.py.

These are the building blocks for the project's "Timing Score" — see
README.md for the overall scoring design.

Indicators implemented here are the standard textbook definitions
(Wilder's RSI, classic 12/26/9 MACD, simple/exponential moving averages).
No external TA library is required; everything is plain pandas so the
math is easy to audit.

Output: one CSV per asset at data/processed/technical/{SYMBOL}.csv with
the original price columns plus indicator columns appended.

Usage:
    python -m src.features.technical
    python -m src.features.technical --asset-type stocks
    python -m src.features.technical --asset-type crypto --symbols BTC,ETH
"""
from __future__ import annotations

import argparse
import sys

import pandas as pd

from src.data.utils import DATA_RAW_DIR, PROJECT_ROOT, ensure_dir, get_logger

logger = get_logger("features.technical")

STOCKS_RAW_DIR = DATA_RAW_DIR / "stocks"
CRYPTO_RAW_DIR = DATA_RAW_DIR / "crypto"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "technical"


# --- indicator functions -----------------------------------------------------
# Each function takes a price Series (indexed by date, ascending) and
# returns a Series aligned to the same index.

def sma(series: pd.Series, window: int) -> pd.Series:
    """Simple moving average."""
    return series.rolling(window=window, min_periods=window).mean()


def ema(series: pd.Series, span: int) -> pd.Series:
    """Exponential moving average."""
    return series.ewm(span=span, adjust=False, min_periods=span).mean()


def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """
    Wilder's Relative Strength Index.

    RSI = 100 - (100 / (1 + RS)), RS = avg_gain / avg_loss over `window`
    periods, using Wilder's smoothing (equivalent to an EMA with
    alpha = 1/window).
    """
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, pd.NA)
    result = 100 - (100 / (1 + rs))
    # When avg_loss is 0 (pure uptrend), RSI is defined as 100.
    result = result.where(avg_loss != 0, 100.0)
    return result


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """
    Classic MACD: fast EMA - slow EMA, plus a signal line (EMA of MACD)
    and the histogram (MACD - signal).
    """
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    return pd.DataFrame(
        {"macd": macd_line, "macd_signal": signal_line, "macd_hist": histogram}
    )


def rolling_volatility(returns: pd.Series, window: int = 30, annualize_factor: float = 365.0) -> pd.Series:
    """Annualized rolling standard deviation of simple daily returns."""
    return returns.rolling(window=window, min_periods=window).std() * (annualize_factor ** 0.5)


def max_drawdown(series: pd.Series) -> pd.Series:
    """
    Running max drawdown: (price / running peak) - 1, expressed as a
    negative fraction (e.g. -0.35 means 35% below the all-time high so far).
    """
    running_max = series.cummax()
    return series / running_max - 1.0


# --- per-asset pipeline ------------------------------------------------------

def compute_features(df: pd.DataFrame, price_col: str) -> pd.DataFrame:
    """Add all technical indicator columns to a price dataframe, in place-safe copy."""
    out = df.copy()
    price = out[price_col]

    out["sma_20"] = sma(price, 20)
    out["sma_50"] = sma(price, 50)
    out["sma_200"] = sma(price, 200)
    out["ema_12"] = ema(price, 12)
    out["ema_26"] = ema(price, 26)
    out["rsi_14"] = rsi(price, 14)

    macd_df = macd(price)
    out = pd.concat([out, macd_df], axis=1)

    daily_return = price.pct_change()
    out["daily_return"] = daily_return
    out["volatility_30d_annualized"] = rolling_volatility(daily_return, window=30)
    out["drawdown_from_ath"] = max_drawdown(price)

    # Simple trend flags, useful for the rule-based Timing Score baseline.
    out["above_sma_50"] = price > out["sma_50"]
    out["golden_cross"] = out["sma_50"] > out["sma_200"]

    return out


def process_stock_file(csv_path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    # yfinance's "close" is unadjusted; use adj_close for indicators so
    # splits/dividends don't create fake price jumps.
    price_col = "adj_close" if "adj_close" in df.columns else "close"
    return compute_features(df, price_col)


def process_crypto_file(csv_path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return compute_features(df, "price_usd")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute technical indicators from raw price data.")
    parser.add_argument(
        "--asset-type",
        choices=["stocks", "crypto", "all"],
        default="all",
        help="Which raw data to process (default: all)",
    )
    parser.add_argument(
        "--symbols",
        default=None,
        help="Comma-separated symbol override, e.g. AAPL,NVDA or BTC,ETH. Defaults to all files found.",
    )
    args = parser.parse_args()

    symbol_filter = None
    if args.symbols:
        symbol_filter = {s.strip().upper() for s in args.symbols.split(",") if s.strip()}

    ensure_dir(PROCESSED_DIR)

    jobs = []
    if args.asset_type in ("stocks", "all") and STOCKS_RAW_DIR.exists():
        for path in sorted(STOCKS_RAW_DIR.glob("*.csv")):
            if symbol_filter and path.stem.upper() not in symbol_filter:
                continue
            jobs.append(("stock", path))
    if args.asset_type in ("crypto", "all") and CRYPTO_RAW_DIR.exists():
        for path in sorted(CRYPTO_RAW_DIR.glob("*.csv")):
            if symbol_filter and path.stem.upper() not in symbol_filter:
                continue
            jobs.append(("crypto", path))

    if not jobs:
        logger.error("No input files found. Run src/data/fetch_stocks.py or fetch_crypto.py first.")
        sys.exit(1)

    for kind, path in jobs:
        symbol = path.stem
        logger.info("Computing technical features for %s (%s)", symbol, kind)
        try:
            out_df = process_stock_file(path) if kind == "stock" else process_crypto_file(path)
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to process %s: %s", symbol, exc)
            continue
        out_path = PROCESSED_DIR / f"{symbol}.csv"
        out_df.to_csv(out_path, index=False)
        logger.info("Saved %d rows -> %s", len(out_df), out_path)

    logger.info("Done.")


if __name__ == "__main__":
    main()

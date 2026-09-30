"""
Shared helpers for the data-fetching layer of crypto-stock-signal-lab.

Kept deliberately tiny and dependency-light (stdlib + pyyaml) so each
fetcher script (fetch_stocks.py, fetch_fundamentals.py, fetch_crypto.py)
can import from here instead of duplicating boilerplate.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import yaml

# --- paths -----------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "config"
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"


def load_asset_config(path: Path | None = None) -> dict[str, Any]:
    """Load config/assets.yaml (stock/crypto universe + SEC concepts)."""
    cfg_path = path or (CONFIG_DIR / "assets.yaml")
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


# --- logging -----------------------------------------------------------------

def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


# --- polite rate limiting -----------------------------------------------------

def polite_sleep(seconds: float, logger: logging.Logger | None = None) -> None:
    """
    Sleep between API calls so we stay well under free-tier rate limits
    (SEC asks for <=10 req/s with a descriptive User-Agent; CoinGecko's
    free public API is far stricter in practice, ~10-30 req/min).
    """
    if logger:
        logger.debug("Sleeping %.2fs to respect rate limits", seconds)
    time.sleep(seconds)

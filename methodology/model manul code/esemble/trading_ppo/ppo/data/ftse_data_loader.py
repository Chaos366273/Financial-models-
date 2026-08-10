"""FTSE dissertation market data for local trading_ppo (no Pathrise / Alpaca).

Loads OHLCV from data/final data for AI:
  1) 03_technical daily parquet/csv if present
  2) else 07_panel/master_panel.parquet month-end bars (offline fallback)
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# __file__ = .../trading_ppo/ppo/data/ftse_data_loader.py
# parents: data, ppo, trading_ppo, esemble, model manul code, methodology, dessertation
_DISS_ROOT = Path(__file__).resolve().parents[6]
_TRADING_PPO = Path(__file__).resolve().parents[2]
FINAL_DATA = Path(os.environ.get(
    "DISS_FINAL_DATA",
    str(_DISS_ROOT / "data" / "final data for AI"),
))
PANEL_PATH = FINAL_DATA / "07_panel" / "master_panel.parquet"
TECH_DIR = FINAL_DATA / "03_technical"
CSV_OUT_DIR = _TRADING_PPO / "ppo" / "saved_data" / "csv_data"
FTSE_CSV_NAME = "ftse_from_final_data.csv"


def _normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Return Date, Company, Open, High, Low, Close, Volume (NewStockDataLoader schema)."""
    work = df.copy()
    colmap = {c.lower(): c for c in work.columns}
    rename = {}
    for want, options in {
        "Date": ("date", "month", "timestamp"),
        "Company": ("ticker", "tic", "company", "symbol"),
        "Open": ("open",),
        "High": ("high",),
        "Low": ("low",),
        "Close": ("close", "px_last", "price"),
        "Volume": ("volume", "px_volume"),
    }.items():
        for opt in options:
            if opt in colmap:
                rename[colmap[opt]] = want
                break
    work = work.rename(columns=rename)
    need = ["Date", "Company", "Close"]
    missing = [c for c in need if c not in work.columns]
    if missing:
        raise ValueError(f"FTSE OHLCV missing columns {missing}; have {list(work.columns)[:20]}")
    if "Open" not in work.columns:
        work["Open"] = work["Close"]
    if "High" not in work.columns:
        work["High"] = work["Close"]
    if "Low" not in work.columns:
        work["Low"] = work["Close"]
    if "Volume" not in work.columns:
        work["Volume"] = 0.0
    work["Date"] = pd.to_datetime(work["Date"])
    work["Company"] = work["Company"].astype(str)
    for c in ("Open", "High", "Low", "Close", "Volume"):
        work[c] = pd.to_numeric(work[c], errors="coerce")
    work = work.dropna(subset=["Date", "Company", "Close"])
    work = work.sort_values(["Company", "Date"]).reset_index(drop=True)
    return work[["Date", "Company", "Open", "High", "Low", "Close", "Volume"]]


def _load_daily_technical(tickers: Optional[List[str]] = None) -> Optional[pd.DataFrame]:
    candidates = [
        TECH_DIR / "daily_features.parquet",
        TECH_DIR / "trading_data_features_full.csv",
        TECH_DIR / "prices_daily_adjusted.parquet",
        TECH_DIR / "prices_daily_adjusted.csv",
    ]
    path = next((p for p in candidates if p.is_file()), None)
    if path is None:
        return None
    logger.info(f"[FTSE] Loading daily technical: {path}")
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    out = _normalize_ohlcv(df)
    if tickers:
        want = {str(t).upper() for t in tickers}
        out = out.loc[out["Company"].str.upper().isin(want)]
    return out


def _load_master_panel(tickers: Optional[List[str]] = None) -> pd.DataFrame:
    if not PANEL_PATH.is_file():
        raise FileNotFoundError(
            f"FTSE master panel not found: {PANEL_PATH}. "
            "Place final data under data/final data for AI/07_panel/"
        )
    logger.info(f"[FTSE] Loading master_panel (month bars): {PANEL_PATH}")
    cols = ["Ticker", "month", "open", "high", "low", "close", "volume"]
    df = pd.read_parquet(PANEL_PATH, columns=cols)
    if tickers:
        want = {str(t).upper() for t in tickers}
        df = df.loc[df["Ticker"].astype(str).str.upper().isin(want)]
    # Cap universe if still huge (no Arch B filter yet)
    if tickers is None and df["Ticker"].nunique() > 80:
        # Prefer liquid names by mean volume
        vol = df.groupby("Ticker")["volume"].mean().sort_values(ascending=False)
        keep = set(vol.head(40).index.astype(str))
        df = df.loc[df["Ticker"].astype(str).isin(keep)]
        logger.warning(f"[FTSE] No ticker filter — truncated to top-40 by volume ({len(keep)})")
    return _normalize_ohlcv(df)


def load_ftse_ohlcv(tickers: Optional[List[str]] = None) -> pd.DataFrame:
    """Load FTSE OHLCV; prefer daily technical, else master_panel monthly bars."""
    daily = _load_daily_technical(tickers)
    if daily is not None and len(daily) > 0:
        return daily
    return _load_master_panel(tickers)


def materialize_ftse_csv(tickers: Optional[List[str]] = None) -> Path:
    """Write NewStockDataLoader-compatible CSV under trading_ppo csv_data."""
    CSV_OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = CSV_OUT_DIR / FTSE_CSV_NAME
    df = load_ftse_ohlcv(tickers)
    df.to_csv(out, index=False)
    logger.info(f"[FTSE] Wrote {len(df)} rows / {df['Company'].nunique()} tickers -> {out}")
    return out


def to_ppo_market_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Map Date/Company/... to date/tic/open/high/low/close/volume used by PPO_fixed."""
    out = df.rename(columns={
        "Date": "date",
        "Company": "tic",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
    }).copy()
    out["Ticker"] = out["tic"]
    out["date"] = pd.to_datetime(out["date"])
    return out

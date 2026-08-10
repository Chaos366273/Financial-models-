"""Build monthly panel from Bloomberg (or pilot) CSVs with FF5 proxy factors."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from data.config_loader import load_config
from data.features import add_proxy_sentiment, winsorise_with_bounds

# Fallback if fundamentals file has no tickers (legacy pilot bundle).
PILOT_EQUITY_TICKERS = {"AAPL", "AMZN", "JNJ", "JPM", "KO", "MSFT", "NVDA"}


def _equity_universe(fund: pd.DataFrame, monthly: pd.DataFrame) -> set[str]:
    """Tickers eligible for FF5 cross-sectional factor construction."""
    if "Ticker" in fund.columns and fund["Ticker"].notna().any():
        return set(fund["Ticker"].dropna().unique())
    return set(monthly["Ticker"].unique()) or PILOT_EQUITY_TICKERS


@dataclass
class PanelData:
    panel: pd.DataFrame
    daily: pd.DataFrame
    tickers: list[str]
    fold_label: str


def _load_daily(features_path: str) -> pd.DataFrame:
    if not Path(features_path).is_file():
        raise FileNotFoundError(f"Features CSV not found: {features_path}")
    df = pd.read_csv(features_path)
    required = {"date", "Ticker", "close", "volume"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Features CSV missing required columns: {', '.join(missing)}")
    df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_localize(None)
    df = df.sort_values(["Ticker", "date"])
    return df


def _load_fundamentals(path: str) -> pd.DataFrame:
    if not Path(path).is_file():
        raise FileNotFoundError(f"Fundamentals CSV not found: {path}")
    f = pd.read_csv(path)
    f = f.rename(columns={
        "Symbol": "Ticker",
        "Price/Earnings": "pe_ratio",
        "Price/Book": "pb_ratio",
        "ROE": "roe",
        "Debt/Equity": "debt_equity",
        "Revenue Growth": "revenue_growth",
        "Earnings Growth": "earnings_growth",
        "Market Cap": "market_cap",
    })
    required = {"Ticker", "Fiscal_Year"}
    missing = sorted(required.difference(f.columns))
    if missing:
        raise ValueError(f"Fundamentals CSV missing required columns: {', '.join(missing)}")
    f["year"] = f["Fiscal_Year"].astype(int)
    return f


def _load_risk_free(path: str | None, fallback: float) -> pd.DataFrame:
    """Load monthly risk-free series; empty frame means use scalar fallback only."""
    if not path or not Path(path).is_file():
        return pd.DataFrame(columns=["month", "risk_free_monthly"])
    rf = pd.read_csv(path)
    required = {"month", "risk_free_monthly"}
    missing = sorted(required.difference(rf.columns))
    if missing:
        raise ValueError(f"Risk-free CSV missing required columns: {', '.join(missing)}")
    rf = rf.copy()
    rf["month"] = pd.to_datetime(rf["month"]).dt.to_period("M").dt.to_timestamp()
    rf["risk_free_monthly"] = pd.to_numeric(rf["risk_free_monthly"], errors="coerce")
    rf = rf.dropna(subset=["month"]).drop_duplicates(subset=["month"], keep="last")
    return rf[["month", "risk_free_monthly"]]


def _to_monthly(
    daily: pd.DataFrame,
    rf_fallback: float,
    rf_series: pd.DataFrame | None = None,
) -> pd.DataFrame:
    daily = daily.copy()
    daily["month"] = daily["date"].dt.to_period("M").dt.to_timestamp()
    monthly = (
        daily.sort_values("date")
        .groupby(["Ticker", "month"], as_index=False)
        .agg(
            close=("close", "last"),
            volume=("volume", "sum"),
            **{c: (c, "last") for c in daily.columns if c in [
                "rsi", "macd", "macd_signal", "atr", "bb_upper", "bb_middle", "bb_lower",
                "price_to_sma_20", "volume_trend", "sma_20", "ema_12",
            ]},
        )
    )
    monthly["monthly_return"] = monthly.groupby("Ticker")["close"].pct_change()
    next_ret = monthly.groupby("Ticker")["monthly_return"].shift(-1)

    rf_col = pd.Series(rf_fallback, index=monthly.index, dtype=float)
    if rf_series is not None and not rf_series.empty:
        rf_map = rf_series.set_index("month")["risk_free_monthly"]
        matched = monthly["month"].map(rf_map)
        rf_col = matched.fillna(rf_fallback)
    monthly["risk_free_monthly"] = rf_col
    monthly["target_excess_return"] = next_ret - monthly["risk_free_monthly"]
    monthly["year"] = monthly["month"].dt.year
    return monthly


def _merge_fundamentals(
    monthly: pd.DataFrame,
    fund: pd.DataFrame,
    lag_months: int = 6,
) -> pd.DataFrame:
    """Attach FY(t) fundamentals only after a reporting lag (default 6 months).

    FY2023 figures become available from 2024-07-01 onward (not Jan 2023),
    avoiding point-in-time look-ahead from calendar-year==fiscal-year merges.
    """
    cols = ["pe_ratio", "pb_ratio", "roe", "debt_equity", "revenue_growth", "earnings_growth", "market_cap"]
    cols = [c for c in cols if c in fund.columns]
    f = fund[["Ticker", "year"] + cols].copy()
    # Available starting June of the following year for lag_months=6:
    # FY(t) available_from = (t+1)-01-01 + lag_months months ≈ mid (t+1).
    f["available_from"] = pd.to_datetime(
        (f["year"] + 1).astype(str) + "-01-01"
    ) + pd.DateOffset(months=lag_months)

    m = monthly.copy()
    m["_idx"] = np.arange(len(m))
    merged = m.merge(f, on="Ticker", how="left", suffixes=("", "_fy"))
    usable = merged["month"] >= merged["available_from"]
    merged = merged.loc[usable].copy()

    # For each (Ticker, month) keep the latest available fiscal year
    merged = merged.sort_values(["Ticker", "month", "year"])
    merged = merged.groupby(["Ticker", "month"], as_index=False).last()

    # Reattach to all monthly rows (ETFs / pre-lag months stay NaN for fund cols)
    out = m.drop(columns=["_idx"]).merge(
        merged[["Ticker", "month"] + cols],
        on=["Ticker", "month"],
        how="left",
    )
    for c in cols:
        if c in out.columns:
            out[c] = out.groupby("Ticker")[c].ffill()
    return out


def _signed_factor(char: pd.Series, ret: pd.Series, high_is_long: bool) -> pd.Series:
    """Cross-sectional long/short using characteristic; NaN char → NaN factor."""
    med = char.median()
    if pd.isna(med) or char.notna().sum() < 2:
        return pd.Series(np.nan, index=char.index)
    if high_is_long:
        long = char > med
    else:
        long = char < med
    out = pd.Series(np.nan, index=char.index, dtype=float)
    valid = char.notna()
    out.loc[valid & long] = ret.loc[valid & long]
    out.loc[valid & ~long] = -ret.loc[valid & ~long]
    return out


def _ff5_factors(
    monthly: pd.DataFrame,
    train_end: pd.Timestamp,
    equity_tickers: set[str],
) -> pd.DataFrame:
    """Cross-sectional FF5 proxies for pilot (replace with Kenneth French / Bloomberg).

    Equity tickers only enter the factor cross-section; ETFs get NaN factor exposures
    (they remain in the panel for H3a portfolios). Winsorise bounds fit on train only.
    """
    m = monthly.copy()
    equity = m["Ticker"].isin(equity_tickers)

    # Market: equal-weight equity mean return that month
    mkt = m.loc[equity].groupby("month")["monthly_return"].mean()
    m["MKT_RF"] = m["month"].map(mkt)

    factor_specs = [
        ("SMB", "market_cap", False),   # small (low mcap) long
        ("HML", "pb_ratio", False),     # value (low PB) long
        ("RMW", "roe", True),           # robust (high ROE) long
        ("CMA", "revenue_growth", False),  # conservative (low growth) long
    ]

    for fname, col, high_is_long in factor_specs:
        m[fname] = np.nan
        if col not in m.columns:
            continue
        for month, g in m.loc[equity].groupby("month"):
            signed = _signed_factor(g[col], g["monthly_return"], high_is_long)
            # Store portfolio return (mean of signed) as the factor for that month
            fac_val = signed.mean()
            m.loc[m["month"] == month, fname] = fac_val

    train_mask = m["month"] <= train_end
    for c in ["MKT_RF", "SMB", "HML", "RMW", "CMA"]:
        m[c] = winsorise_with_bounds(m[c], train_mask)
    return m


def split_panel(panel: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    train_end = pd.Timestamp(cfg["splits"]["train_end"])
    val_end = pd.Timestamp(cfg["splits"]["val_end"])
    train = panel[panel["month"] <= train_end].copy()
    val = panel[(panel["month"] > train_end) & (panel["month"] <= val_end)].copy()
    test = panel[panel["month"] > val_end].copy()
    fold = cfg["splits"]["test_label"]
    return train, val, test, fold


def _cache_meta_path(cache_path: Path) -> Path:
    return cache_path.with_suffix(cache_path.suffix + ".meta.json")


def _source_mtime(path: str | None) -> float:
    if not path:
        return 0.0
    p = Path(path)
    return p.stat().st_mtime if p.is_file() else 0.0


def _cache_fingerprint(cfg: dict) -> dict:
    return {
        "features_csv": cfg["data"]["features_csv"],
        "fundamentals_csv": cfg["data"]["fundamentals_csv"],
        "risk_free_csv": cfg["data"].get("risk_free_csv"),
        "risk_free_rate": float(cfg["data"].get("risk_free_rate", 0.0)),
        "fundamentals_lag_months": int(cfg["data"].get("fundamentals_lag_months", 6)),
        "train_end": str(cfg["splits"]["train_end"]),
        "features_mtime": _source_mtime(cfg["data"]["features_csv"]),
        "fundamentals_mtime": _source_mtime(cfg["data"]["fundamentals_csv"]),
        "risk_free_mtime": _source_mtime(cfg["data"].get("risk_free_csv")),
    }


def _panel_cache_fresh(cache_path: Path, cfg: dict) -> bool:
    csv_path = cache_path.with_suffix(".csv")
    if not cache_path.is_file() and not csv_path.is_file():
        return False
    meta_path = _cache_meta_path(cache_path if cache_path.is_file() else csv_path)
    if not meta_path.is_file():
        return False
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return meta == _cache_fingerprint(cfg)


def _read_panel_cache(cache_path: Path) -> pd.DataFrame | None:
    csv_path = cache_path.with_suffix(".csv")
    try:
        if cache_path.is_file():
            monthly = pd.read_parquet(cache_path)
        elif csv_path.is_file():
            monthly = pd.read_csv(csv_path)
        else:
            return None
    except ImportError:
        if not csv_path.is_file():
            return None
        monthly = pd.read_csv(csv_path)
    if "month" in monthly.columns:
        monthly["month"] = pd.to_datetime(monthly["month"])
    return monthly


def _write_panel_cache(cache_path: Path, monthly: pd.DataFrame, cfg: dict) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    meta = json.dumps(_cache_fingerprint(cfg), indent=2)
    try:
        monthly.to_parquet(cache_path, index=False)
        _cache_meta_path(cache_path).write_text(meta, encoding="utf-8")
    except ImportError:
        csv_path = cache_path.with_suffix(".csv")
        monthly.to_csv(csv_path, index=False)
        _cache_meta_path(csv_path).write_text(meta, encoding="utf-8")


def _load_master_panel(path: str) -> pd.DataFrame:
    """Load pre-built final-data monthly panel (already has target + all indicators)."""
    p = Path(path)
    csv_fallback = p.with_suffix(".csv")
    if not p.is_file() and not csv_fallback.is_file():
        raise FileNotFoundError(f"Master panel not found: {p}")

    monthly = None
    if p.is_file() and p.suffix.lower() == ".parquet":
        try:
            monthly = pd.read_parquet(p)
        except ImportError as exc:
            if csv_fallback.is_file():
                monthly = pd.read_csv(csv_fallback)
            else:
                raise ImportError(
                    "Reading master_panel.parquet needs pyarrow (or fastparquet).\n"
                    "Fix one of:\n"
                    "  1) Prefer Python 3.11:  run_xgb.bat\n"
                    "     C:\\Users\\Faiaz\\AppData\\Local\\Programs\\Python\\Python311\\python.exe xgb_return.py\n"
                    "  2) Or install engine on this interpreter:\n"
                    "     python -m pip install pyarrow\n"
                    f"Original error: {exc}"
                ) from exc
    elif p.is_file():
        monthly = pd.read_csv(p)
    else:
        monthly = pd.read_csv(csv_fallback)

    if "month" not in monthly.columns or "Ticker" not in monthly.columns:
        raise ValueError(f"Master panel missing Ticker/month columns: {p}")
    if "target_excess_return" not in monthly.columns:
        raise ValueError(f"Master panel missing target_excess_return: {p}")
    monthly = monthly.copy()
    monthly["month"] = pd.to_datetime(monthly["month"])
    monthly = monthly.dropna(subset=["target_excess_return"])
    # Keep proxy if already present; otherwise build for H2 backward compatibility
    if "sentiment_proxy" not in monthly.columns:
        monthly = add_proxy_sentiment(monthly, train_end=None)
    return monthly


def _load_daily_optional(features_path: str | None) -> pd.DataFrame:
    """Daily features for H3b GARCH; empty frame if master-panel-only mode."""
    if not features_path or not Path(features_path).is_file():
        return pd.DataFrame()
    return _load_daily(features_path)


def build_panel() -> tuple[PanelData, dict]:
    cfg = load_config()
    rf_fallback = float(cfg["data"]["risk_free_rate"])
    lag = int(cfg["data"].get("fundamentals_lag_months", 6))
    train_end = pd.Timestamp(cfg["splits"]["train_end"])
    val_end = pd.Timestamp(cfg["splits"]["val_end"])
    cache_raw = cfg["data"].get("panel_cache")
    cache_path = Path(cache_raw) if cache_raw else None
    master_raw = cfg["data"].get("master_panel")
    master_path = Path(master_raw) if master_raw else None

    # Prefer final-data master panel when configured and present
    if master_path is not None and master_path.is_file():
        monthly = _load_master_panel(str(master_path))
        daily = _load_daily_optional(cfg["data"].get("features_csv"))
        source_label = str(master_path)
    else:
        features_csv = cfg["data"].get("features_csv")
        fundamentals_csv = cfg["data"].get("fundamentals_csv")
        if not features_csv or not fundamentals_csv:
            raise FileNotFoundError(
                "No master_panel.parquet and features_csv/fundamentals_csv not set. "
                "Point config.yaml data.master_panel at final data for AI/07_panel."
            )
        daily = _load_daily(features_csv)
        fund = _load_fundamentals(fundamentals_csv)

        monthly = None
        if cache_path is not None and _panel_cache_fresh(cache_path, cfg):
            monthly = _read_panel_cache(cache_path)
        if monthly is None:
            rf_series = _load_risk_free(cfg["data"].get("risk_free_csv"), rf_fallback)
            monthly = _to_monthly(daily, rf_fallback, rf_series=rf_series)
            monthly = _merge_fundamentals(monthly, fund, lag_months=lag)
            equity_tickers = _equity_universe(fund, monthly)
            monthly = _ff5_factors(monthly, train_end=train_end, equity_tickers=equity_tickers)
            monthly = add_proxy_sentiment(monthly, train_end=train_end)
            monthly = monthly.dropna(subset=["target_excess_return"])
            if cache_path is not None:
                _write_panel_cache(cache_path, monthly, cfg)
        source_label = str(features_csv)

    train, val, test, fold = split_panel(monthly, cfg)
    empty_splits = [
        name for name, frame in (("train", train), ("validation", val), ("test", test))
        if frame.empty
    ]
    if empty_splits:
        raise ValueError(
            "Dataset produced empty chronological split(s): "
            + ", ".join(empty_splits)
            + f". Check split dates in config.yaml (train_end={train_end.date()}, "
            + f"val_end={val_end.date()}) against the CSV date range."
        )
    tickers = sorted(monthly["Ticker"].unique().tolist())
    pdata = PanelData(panel=monthly, daily=daily, tickers=tickers, fold_label=fold)
    splits = {
        "train": train,
        "val": val,
        "test": test,
        "fold": fold,
        "train_end": train_end,
        "val_end": val_end,
        "features_csv": source_label,
        "fundamentals_csv": cfg["data"].get("fundamentals_csv"),
        "risk_free_csv": cfg["data"].get("risk_free_csv"),
        "master_panel": str(master_path) if master_path and master_path.is_file() else None,
    }
    return pdata, splits

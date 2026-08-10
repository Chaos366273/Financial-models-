"""Feature engineering: traditional / alt-data / full master-panel indicator sets."""
from __future__ import annotations

import numpy as np
import pandas as pd

# H1 baseline (unchanged for traditional vs with_alt_data comparisons)
TECH_COLS = [
    "rsi", "macd", "macd_signal", "atr", "bb_upper", "bb_middle", "bb_lower",
    "price_to_sma_20", "volume_trend", "sma_20", "ema_12",
]

FUND_COLS = [
    "pe_ratio", "pb_ratio", "roe", "debt_equity", "revenue_growth", "earnings_growth",
]

FACTOR_COLS = ["MKT_RF", "SMB", "HML", "RMW", "CMA"]

# Cross-sectional factors only (no market factor — same for all stocks in a month)
FACTOR_XS_COLS = ["SMB", "HML", "RMW", "CMA"]

# Extra indicators present on final-data master_panel.parquet
TECH_EXTRA_COLS = ["sma_15", "volatility_30d"]

LIQUIDITY_COLS = [
    "amihud_illiquidity", "bid_ask_spread", "eqy_turnover", "volume",
    "px_bid", "px_ask", "cur_mkt_cap",
]

FUND_EXTRA_COLS = ["market_cap", "log_market_cap"]

FACTOR_EXTRA_COLS = ["MOM", "RF_FF"]

SENTIMENT_COLS = [
    "news_sentiment",
    "news_publication_count",
    "twitter_sentiment",
    "twitter_publication_count",
    "sentiment_proxy",
]

MACRO_COLS = [
    "uk_bank_rate", "uk_cpi_yoy", "uk_unemployment_rate", "uk_gdp_qoq",
    "us_3m_yield", "us_fed_funds_target",
    "brent_crude", "gold_usd", "silver_usd", "gbp_usd",
    "ftse100_level", "ftse250_level",
]

MEMBERSHIP_COLS = ["is_member", "has_bloomberg_sentiment"]

MONTH_FE_COLS = [f"month_fe_{m:02d}" for m in range(2, 13)]

SECTOR_DUMMY_COLS = [
    "sector__Automobiles_and_Parts",
    "sector__Banks",
    "sector__Basic_Resources",
    "sector__Chemicals",
    "sector__Construction_and_Materials",
    "sector__Consumer_Products_and_Services",
    "sector__Energy",
    "sector__Financial_Services",
    "sector__Food__Beverage_and_Tobacco",
    "sector__Health_Care",
    "sector__Industrial_Goods_and_Services",
    "sector__Insurance",
    "sector__Media",
    "sector__Personal_Care__Drug_and_Grocery_Stores",
    "sector__Real_Estate",
    "sector__Retail",
    "sector__Technology",
    "sector__Telecommunications",
    "sector__Travel_and_Leisure",
    "sector__Unknown",
    "sector__Utilities",
]

# Clean liquidity (no raw bid/ask/mcap levels — use log_market_cap instead)
LIQUIDITY_CLEAN_COLS = [
    "amihud_illiquidity", "bid_ask_spread", "eqy_turnover",
]

# Stock-picking set: NO MKT_RF, NO sector dummies
H1_CLEAN_COLS = (
    TECH_COLS
    + ["sma_15", "volatility_30d"]
    + LIQUIDITY_CLEAN_COLS
    + FUND_COLS
    + ["log_market_cap"]
    + FACTOR_XS_COLS
    + ["MOM"]
)

H1_STOCKPICK_COLS = list(H1_CLEAN_COLS)

# Hybrid (restored): stock features + MKT_RF + sector dummies — matches earlier LS ~+0.36% setup
H1_HYBRID_COLS = (
    TECH_COLS
    + ["sma_15", "volatility_30d"]
    + LIQUIDITY_CLEAN_COLS
    + FUND_COLS
    + ["log_market_cap"]
    + FACTOR_COLS
    + ["MOM"]
    + [c for c in SECTOR_DUMMY_COLS if c != "sector__Unknown"]
)
ALL_INDICATOR_COLS = (
    TECH_COLS
    + TECH_EXTRA_COLS
    + LIQUIDITY_COLS
    + FUND_COLS
    + FUND_EXTRA_COLS
    + FACTOR_COLS
    + FACTOR_EXTRA_COLS
    + SENTIMENT_COLS
    + MACRO_COLS
    + MEMBERSHIP_COLS
    + MONTH_FE_COLS
    + SECTOR_DUMMY_COLS
)


def winsorise(s: pd.Series, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
    """Winsorise using bounds estimated from the series itself."""
    if s.dropna().empty:
        return s
    lo, hi = s.quantile(lower), s.quantile(upper)
    return s.clip(lo, hi)


def winsorise_with_bounds(
    s: pd.Series,
    train_mask: pd.Series,
    lower: float = 0.01,
    upper: float = 0.99,
) -> pd.Series:
    """Winsorise full series using quantile bounds fit on train rows only."""
    ref = s.loc[train_mask].dropna()
    if ref.empty:
        return s
    lo, hi = float(ref.quantile(lower)), float(ref.quantile(upper))
    return s.clip(lo, hi)


def add_proxy_sentiment(df: pd.DataFrame, train_end: pd.Timestamp | None = None) -> pd.DataFrame:
    """H2 pilot proxy: volume shock x return sign (fallback when Bloomberg BSI missing).

    When train_end is supplied, per-ticker z-score mean/std and winsorise bounds
    are estimated on the train window only and applied forward (no test leakage).
    """
    out = df.copy()
    if "volume_trend" not in out.columns or "monthly_return" not in out.columns:
        out["sentiment_proxy"] = 0.0
        return out

    train_mask = (
        (out["month"] <= train_end)
        if train_end is not None and "month" in out.columns
        else pd.Series(True, index=out.index)
    )

    z_parts = []
    for _, g in out.groupby("Ticker", sort=False):
        g_train = train_mask.loc[g.index]
        tr = g.loc[g_train, "volume_trend"].dropna()
        mu = float(tr.mean()) if len(tr) else float(g["volume_trend"].mean())
        sd = float(tr.std()) if len(tr) > 1 else float(g["volume_trend"].std() or 1.0)
        if not np.isfinite(sd) or sd < 1e-9:
            sd = 1.0
        if not np.isfinite(mu):
            mu = 0.0
        z_parts.append((g["volume_trend"] - mu) / (sd + 1e-9))

    vol_z = pd.concat(z_parts).reindex(out.index)
    out["sentiment_proxy"] = vol_z * np.sign(out["monthly_return"].fillna(0))
    out["sentiment_proxy"] = winsorise_with_bounds(out["sentiment_proxy"], train_mask)
    return out


def get_feature_columns(feature_set: str) -> list[str]:
    """Return ordered feature names for a named set.

    Sets:
      traditional     — H1 baseline tech + fund + FF5 (includes MKT_RF)
      with_alt_data   — traditional + Bloomberg news/twitter sentiment (+ proxy)
      h1_clean / h1_stockpick — stock-picking set (no MKT_RF, no sector dummies)
      h1_hybrid       — tech/fund/FF5(+MKT_RF)/MOM + sector dummies (restored)
      all_indicators  — full master-panel predictor set
    """
    base = TECH_COLS + FUND_COLS + FACTOR_COLS
    if feature_set == "traditional":
        return list(base)
    if feature_set == "with_alt_data":
        return list(base) + list(SENTIMENT_COLS)
    if feature_set in ("h1_clean", "h1_improved", "h1_stockpick"):
        return list(H1_CLEAN_COLS)
    if feature_set in ("h1_hybrid", "hybrid"):
        return list(H1_HYBRID_COLS)
    if feature_set in ("all_indicators", "full", "all"):
        return list(ALL_INDICATOR_COLS)
    raise ValueError(
        f"Unknown feature_set={feature_set!r}. "
        "Use traditional | with_alt_data | h1_clean | h1_hybrid | all_indicators"
    )


def prepare_xy(
    df: pd.DataFrame,
    feature_set: str = "traditional",
    target_col: str = "target_excess_return",
    feature_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Build X/y. If feature_cols is set, use that list instead of the named feature_set."""
    cols = list(feature_cols) if feature_cols is not None else get_feature_columns(feature_set)
    cols = [c for c in cols if c in df.columns]
    # Drop only on target — fund/factor NaNs (pre-lag / ETFs) are filled with 0
    # so lagged fundamentals do not wipe the training window.
    work = df.dropna(subset=[target_col]).copy()
    if not cols:
        return pd.DataFrame(), pd.Series(dtype=float), cols
    X = work[cols].replace([np.inf, -np.inf], np.nan)
    # Bool membership / flags → float
    for c in X.columns:
        if X[c].dtype == bool:
            X[c] = X[c].astype(float)
    X = X.fillna(0.0)
    y = work[target_col]
    return X, y, cols

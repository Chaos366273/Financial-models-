"""Feature columns for Bloomberg / master-panel mode (standalone copy).

Keep in sync with dissertation-models/data/features.py.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TECH_COLS = [
    "rsi", "macd", "macd_signal", "atr", "bb_upper", "bb_middle", "bb_lower",
    "price_to_sma_20", "volume_trend", "sma_20", "ema_12",
]

FUND_COLS = [
    "pe_ratio", "pb_ratio", "roe", "debt_equity", "revenue_growth", "earnings_growth",
]

FACTOR_COLS = ["MKT_RF", "SMB", "HML", "RMW", "CMA"]

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

LIQUIDITY_CLEAN_COLS = [
    "amihud_illiquidity", "bid_ask_spread", "eqy_turnover",
]

H1_CLEAN_COLS = (
    TECH_COLS
    + ["sma_15", "volatility_30d"]
    + LIQUIDITY_CLEAN_COLS
    + FUND_COLS
    + ["log_market_cap"]
    + ["SMB", "HML", "RMW", "CMA"]
    + ["MOM"]
)

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


def get_feature_columns(feature_set: str) -> list[str]:
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
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    cols = get_feature_columns(feature_set)
    cols = [c for c in cols if c in df.columns]
    work = df.dropna(subset=[target_col]).copy()
    if not cols:
        return pd.DataFrame(), pd.Series(dtype=float), cols
    X = work[cols].replace([np.inf, -np.inf], np.nan)
    for c in X.columns:
        if X[c].dtype == bool:
            X[c] = X[c].astype(float)
    X = X.fillna(0.0)
    y = work[target_col]
    return X, y, cols

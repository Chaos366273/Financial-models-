"""Shared calculation / data-lineage manifests for Lab auditability."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from data.config_loader import load_config
from data.features import FACTOR_COLS, FUND_COLS, TECH_COLS, get_feature_columns


SHARED_TRANSFORMATIONS = [
    {
        "id": "daily_to_monthly",
        "label": "Aggregate daily prices/features to month-end rows per ticker",
        "detail": "close=last, volume=sum, technical indicators=last observation in month",
    },
    {
        "id": "monthly_return",
        "label": "Compute monthly return from month-end close",
        "formula": "monthly_return = close_t / close_{t-1} - 1",
    },
    {
        "id": "target_excess_return",
        "label": "Build next-month excess return target (H1/H2)",
        "formula": "target_excess_return = lead(monthly_return, 1) - risk_free_rate",
    },
    {
        "id": "fundamentals_lag",
        "label": "Merge fundamentals with reporting lag (no look-ahead)",
        "formula": "FY(t) available from (t+1)-01-01 + lag_months",
    },
    {
        "id": "ff5_proxies",
        "label": "Attach Ken French Europe FF5 + MOM (or rebuild proxies if master panel absent)",
        "detail": "Final-data master panel uses Ken French Europe factors; CSV rebuild path still builds cross-sectional proxies",
    },
    {
        "id": "winsorise_train",
        "label": "Winsorise factor/sentiment bounds on train window only",
        "detail": "Bounds fitted on train, applied forward to validation/test",
    },
]


def csv_columns(path: str | Path, limit: int = 40) -> list[str]:
    p = Path(path)
    if not p.is_file():
        return []
    try:
        import pandas as pd

        return [str(c) for c in pd.read_csv(p, nrows=0).columns.tolist()[:limit]]
    except Exception:
        return []


def shared_manifest(
    data_context: dict[str, Any] | None = None,
    feature_set: str = "traditional",
) -> dict[str, Any]:
    cfg = load_config()
    ctx = dict(data_context or {})
    features_csv = ctx.get("features_csv") or cfg["data"]["features_csv"]
    fundamentals_csv = ctx.get("fundamentals_csv") or cfg["data"]["fundamentals_csv"]
    lag = int(cfg["data"].get("fundamentals_lag_months", 6))
    rf = float(cfg["data"].get("risk_free_rate", 0.0))

    transforms = []
    for step in SHARED_TRANSFORMATIONS:
        item = dict(step)
        if step["id"] == "fundamentals_lag":
            item["lag_months"] = lag
        if step["id"] == "target_excess_return":
            item["risk_free_rate"] = rf
        transforms.append(item)

    return {
        "inputs": {
            "dataset": ctx.get("dataset"),
            "features_csv": str(features_csv),
            "features_file": Path(str(features_csv)).name,
            "fundamentals_csv": str(fundamentals_csv),
            "fundamentals_file": Path(str(fundamentals_csv)).name,
            "raw_feature_columns": csv_columns(features_csv),
            "raw_fundamental_columns": csv_columns(fundamentals_csv),
            "technical_columns": [c for c in TECH_COLS],
            "fundamental_columns": [c for c in FUND_COLS],
            "factor_columns": [c for c in FACTOR_COLS],
            "model_feature_columns": get_feature_columns(feature_set),
            "feature_set": feature_set,
            "config": {
                "train_end": cfg["splits"]["train_end"],
                "val_end": cfg["splits"]["val_end"],
                "test_label": cfg["splits"]["test_label"],
                "fundamentals_lag_months": lag,
                "risk_free_rate": rf,
            },
            "splits": {
                "train": ctx.get("train"),
                "validation": ctx.get("validation"),
                "test": ctx.get("test"),
                "tickers": ctx.get("tickers"),
            },
        },
        "transformations": transforms,
        "target": {
            "name": "target_excess_return",
            "unit": "monthly_excess_return",
            "formula": "lead(monthly_return, 1) - risk_free_rate",
            "plain_english": (
                "Next month's stock return (percent change in month-end close) "
                "minus the configured risk-free rate. RMSE compares predicted vs "
                "actual values of this target — not stock price or volume."
            ),
            "risk_free_rate": rf,
        },
    }


def metric_definitions(bucket: str) -> dict[str, str]:
    if bucket == "h1":
        return {
            "rmse": "Root mean squared error of predicted vs actual next-month excess returns",
            "mae": "Mean absolute error of predicted vs actual next-month excess returns",
            "r2": "Fraction of variance in next-month excess returns explained out of sample",
            "n_obs": "Number of out-of-sample ticker-month (or market-month) observations scored",
        }
    if bucket == "h3a":
        return {
            "sharpe": "Annualised risk-adjusted return of the strategy on the unseen test months",
            "sortino": "Annualised downside risk-adjusted return on the unseen test months",
            "max_drawdown": "Worst peak-to-trough decline of the strategy wealth path",
            "net_return_annual": "Annualised mean monthly portfolio return on the test window",
        }
    if bucket == "h3b":
        return {
            "var_95": "Average one-day 95% Value-at-Risk forecast over the test window",
            "cvar_95": "Approximate expected shortfall (mean VaR × 1.2 on this pilot setup)",
            "violation_rate": "Share of test days where realised return breached the VaR forecast",
        }
    return {}

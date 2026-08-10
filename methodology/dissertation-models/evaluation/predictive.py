"""Predictive metrics for H1 (level error + ranking usefulness)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def predictive_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[mask], y_pred[mask]
    if len(y_true) == 0:
        return {"mse": None, "mae": None, "rmse": None, "r2": None, "n_obs": 0}
    mse = float(mean_squared_error(y_true, y_pred))
    return {
        "mse": mse,
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mse)),
        "r2": float(r2_score(y_true, y_pred)),
        "n_obs": int(len(y_true)),
    }


def ranking_metrics(
    frame: pd.DataFrame,
    y_col: str = "y_true",
    pred_col: str = "y_pred",
    month_col: str = "month",
    top_frac: float = 0.1,
    min_names: int = 10,
) -> dict:
    """Spearman IC (mean across months) + mean long-short decile spread.

    frame must contain month, y_true (actual next-month excess return), y_pred.
    Long-short uses actual returns of top vs bottom predicted decile each month.
    """
    need = {y_col, pred_col, month_col}
    if frame is None or frame.empty or not need.issubset(frame.columns):
        return {
            "ic_mean": None,
            "ic_std": None,
            "ic_ir": None,
            "long_short_mean": None,
            "long_short_std": None,
            "n_months": 0,
        }

    work = frame[[month_col, y_col, pred_col]].copy()
    work[month_col] = pd.to_datetime(work[month_col])
    work = work.replace([np.inf, -np.inf], np.nan).dropna()
    ics: list[float] = []
    spreads: list[float] = []

    for _, g in work.groupby(month_col, sort=True):
        if len(g) < min_names:
            continue
        ic = g[pred_col].corr(g[y_col], method="spearman")
        if np.isfinite(ic):
            ics.append(float(ic))
        n = len(g)
        k = max(1, int(np.floor(n * top_frac)))
        g_sorted = g.sort_values(pred_col)
        bottom = g_sorted.head(k)[y_col].mean()
        top = g_sorted.tail(k)[y_col].mean()
        if np.isfinite(top) and np.isfinite(bottom):
            spreads.append(float(top - bottom))

    if not ics:
        return {
            "ic_mean": None,
            "ic_std": None,
            "ic_ir": None,
            "long_short_mean": float(np.mean(spreads)) if spreads else None,
            "long_short_std": float(np.std(spreads, ddof=1)) if len(spreads) > 1 else None,
            "n_months": int(len(spreads)),
        }

    ic_mean = float(np.mean(ics))
    ic_std = float(np.std(ics, ddof=1)) if len(ics) > 1 else 0.0
    return {
        "ic_mean": ic_mean,
        "ic_std": ic_std,
        "ic_ir": float(ic_mean / ic_std) if ic_std > 1e-12 else None,
        "long_short_mean": float(np.mean(spreads)) if spreads else None,
        "long_short_std": float(np.std(spreads, ddof=1)) if len(spreads) > 1 else None,
        "n_months": int(len(ics)),
    }

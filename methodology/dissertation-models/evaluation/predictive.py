"""Predictive metrics for H1 (level error + ranking usefulness)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

PRED_DISP_TOL = 1e-12
TOP_K = 50
LS_TOP_FRAC = 0.10


def _finite_pair(y_true, y_pred) -> tuple[np.ndarray, np.ndarray]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    return y_true[mask], y_pred[mask]


def _has_pred_dispersion(values, tol: float = PRED_DISP_TOL) -> bool:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) < 2:
        return False
    return float(np.std(v)) > tol


def gu_r2_oos(y_true, y_pred) -> float | None:
    """Gu, Kelly and Xiu (2020): 1 - SSE / sum(y^2). Zero-return benchmark."""
    y_true, y_pred = _finite_pair(y_true, y_pred)
    if len(y_true) == 0:
        return None
    denom = float(np.sum(np.square(y_true)))
    if denom <= 0:
        return None
    return float(1.0 - np.sum(np.square(y_true - y_pred)) / denom)


def predictive_metrics(y_true, y_pred) -> dict:
    y_true, y_pred = _finite_pair(y_true, y_pred)
    if len(y_true) == 0:
        return {"mse": None, "mae": None, "rmse": None, "r2": None, "r2_oos": None, "n_obs": 0}
    mse = float(mean_squared_error(y_true, y_pred))
    return {
        "mse": mse,
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mse)),
        "r2": float(r2_score(y_true, y_pred)),
        "r2_oos": gu_r2_oos(y_true, y_pred),
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

    Months with no cross-sectional variation in y_pred are skipped (ranking not identified).
    Long-short Sharpe is annualised from monthly spreads (mean / std * sqrt(12)).
    """
    empty = {
        "ic_mean": None,
        "ic_std": None,
        "ic_ir": None,
        "long_short_mean": None,
        "long_short_std": None,
        "long_short_sharpe": None,
        "n_months": 0,
        "ranking_identified": False,
    }
    need = {y_col, pred_col, month_col}
    if frame is None or frame.empty or not need.issubset(frame.columns):
        return empty

    work = frame[[month_col, y_col, pred_col]].copy()
    work[month_col] = pd.to_datetime(work[month_col])
    work = work.replace([np.inf, -np.inf], np.nan).dropna()
    ics: list[float] = []
    spreads: list[float] = []

    for _, g in work.groupby(month_col, sort=True):
        if len(g) < min_names:
            continue
        if not _has_pred_dispersion(g[pred_col]):
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

    ls_mean = float(np.mean(spreads)) if spreads else None
    ls_std = float(np.std(spreads, ddof=1)) if len(spreads) > 1 else None
    ls_sharpe = None
    if ls_mean is not None and ls_std is not None and ls_std > 1e-12:
        ls_sharpe = float(ls_mean / ls_std * np.sqrt(12.0))

    identified = bool(ics or spreads)
    if not ics:
        return {
            "ic_mean": None,
            "ic_std": None,
            "ic_ir": None,
            "long_short_mean": ls_mean,
            "long_short_std": ls_std,
            "long_short_sharpe": ls_sharpe,
            "n_months": int(len(spreads)),
            "ranking_identified": identified,
        }

    ic_mean = float(np.mean(ics))
    ic_std = float(np.std(ics, ddof=1)) if len(ics) > 1 else 0.0
    return {
        "ic_mean": ic_mean,
        "ic_std": ic_std,
        "ic_ir": float(ic_mean / ic_std) if ic_std > 1e-12 else None,
        "long_short_mean": ls_mean,
        "long_short_std": ls_std,
        "long_short_sharpe": ls_sharpe,
        "n_months": int(len(ics)),
        "ranking_identified": True,
    }


def economic_metrics(
    frame: pd.DataFrame,
    k: int = TOP_K,
    y_col: str = "y_true",
    pred_col: str = "y_pred",
    month_col: str = "month",
    min_names: int = 10,
) -> dict:
    """Top-k mean return, edge vs equal-weight market, and hit rate (share of months edge > 0)."""
    empty = {
        "topk_avg_return": None,
        "topk_edge": None,
        "hit_rate": None,
        "n_months": 0,
        "n_picks": 0,
        "ranking_identified": False,
    }
    need = {y_col, pred_col, month_col}
    if frame is None or frame.empty or not need.issubset(frame.columns):
        return empty

    work = frame[[month_col, y_col, pred_col]].copy()
    work[month_col] = pd.to_datetime(work[month_col])
    work = work.replace([np.inf, -np.inf], np.nan).dropna()
    edges: list[float] = []
    ports: list[float] = []
    n_picks = 0

    for _, g in work.groupby(month_col, sort=True):
        if len(g) < min_names:
            continue
        if not _has_pred_dispersion(g[pred_col]):
            continue
        kk = min(int(k), len(g))
        picks = g.sort_values(pred_col, ascending=False).head(kk)
        port = float(picks[y_col].mean())
        mkt = float(g[y_col].mean())
        ports.append(port)
        edges.append(port - mkt)
        n_picks += int(len(picks))

    if not ports:
        return empty
    edge_arr = np.asarray(edges, dtype=float)
    return {
        "topk_avg_return": float(np.mean(ports)),
        "topk_edge": float(np.mean(edge_arr)),
        "hit_rate": float(np.mean(edge_arr > 0)),
        "n_months": int(len(ports)),
        "n_picks": int(n_picks),
        "ranking_identified": True,
    }

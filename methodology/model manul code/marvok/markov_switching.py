"""Markov-switching H1: regime-dependent FF5 walk-forward + market appendix.

Primary (H1, default k=2 — Hamilton bull/bear):
  1) Fit MarkovRegression on train monthly MKT_RF
  2) Soft-weight train stock-months by filtered regime probs
  3) Weighted OLS FF5 per regime (pooled FF5 fallback if a regime is thin)
  4) OOS: roll filtered probs with transition matrix; blend regime FF5 forecasts

Extension (k=4 — Maheu-style labels by sorted means):
  bear / bear_rally / bull_correction / bull
  Writes to walk_forward_k4/ and market_appendix_k4/ (does NOT overwrite k=2).

Appendix:
  Pure Hamilton market forecast of MKT_RF (not stock-month H1 unit).

Protocol matches fama5.py / xgb_walk_forward (10y train -> 1y eval, members-only).

Commands:
  python markov_switching.py
  python markov_switching.py --k-regimes 4
  python markov_switching.py --build-h1-table
  python markov_switching.py --market-only
  python markov_switching.py --market-only --k-regimes 4
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

ROOT = Path(__file__).resolve().parent
MANUAL_ROOT = ROOT.parent
DISS_MODELS = MANUAL_ROOT.parent / "dissertation-models"
FINAL_DATA = MANUAL_ROOT.parent.parent / "data" / "final data for AI"
MASTER_PANEL = FINAL_DATA / "07_panel" / "master_panel.parquet"
OUT_ROOT = ROOT
RESULTS_TRACKER = OUT_ROOT / "markov_results_tracker.csv"

# Set by configure_run()
K_REGIMES = 2
WINDOW_MODE = "rolling"  # rolling | expanding
TRAIN_ASSIGN = "soft"  # soft | hard (R1)
TRAIN_PROB_SOURCE = "filtered"  # filtered | smoothed (R2)
MARKET_SERIES = "mkt_rf"  # mkt_rf | ew_excess (R4)
ALLOW_POOLED_FALLBACK = True  # False = R5 (no pooled FF5 when regime thin/fail)
WF_DIR = OUT_ROOT / "walk_forward_baseline"
MARKET_DIR = OUT_ROOT / "market_appendix"
MIN_HIST_MONTHS = 24
MIN_REGIME_WEIGHT_SUM = 200.0
MIN_REGIME_MONTHS = 12
MS_MAXITER = 200
REGIME_LABELS: list[str] = ["bear", "bull"]
EXPAND_START_YEAR = 2002

FACTOR_COLS = ["MKT_RF", "SMB", "HML", "RMW", "CMA"]
TARGET_COL = "target_excess_return"
FILTER_MEMBERS = True
COLLAPSE_MU_EPS = 1e-6

# Frozen rolling soft+filtered+MKT_RF baselines (never overwritten by sensitivities)
WF_DIR_K2 = OUT_ROOT / "walk_forward_baseline"
MARKET_DIR_K2 = OUT_ROOT / "market_appendix"
WF_DIR_K4 = OUT_ROOT / "walk_forward_k4"
MARKET_DIR_K4 = OUT_ROOT / "market_appendix_k4"
WF_DIR_EXP_K2 = OUT_ROOT / "walk_forward_expanding_k2"
MARKET_DIR_EXP_K2 = OUT_ROOT / "market_appendix_expanding_k2"
WF_DIR_EXP_K4 = OUT_ROOT / "walk_forward_expanding_k4"
MARKET_DIR_EXP_K4 = OUT_ROOT / "market_appendix_expanding_k4"
WF_DIR_HARD_K2 = OUT_ROOT / "walk_forward_hard_k2"
MARKET_DIR_HARD_K2 = OUT_ROOT / "market_appendix_hard_k2"
WF_DIR_HARD_K4 = OUT_ROOT / "walk_forward_hard_k4"
MARKET_DIR_HARD_K4 = OUT_ROOT / "market_appendix_hard_k4"
WF_DIR_SMOOTH_K2 = OUT_ROOT / "walk_forward_smooth_k2"
MARKET_DIR_SMOOTH_K2 = OUT_ROOT / "market_appendix_smooth_k2"
WF_DIR_SMOOTH_K4 = OUT_ROOT / "walk_forward_smooth_k4"
MARKET_DIR_SMOOTH_K4 = OUT_ROOT / "market_appendix_smooth_k4"
WF_DIR_EW_K2 = OUT_ROOT / "walk_forward_ew_k2"
MARKET_DIR_EW_K2 = OUT_ROOT / "market_appendix_ew_k2"
WF_DIR_EW_K4 = OUT_ROOT / "walk_forward_ew_k4"
MARKET_DIR_EW_K4 = OUT_ROOT / "market_appendix_ew_k4"
WF_DIR_NOFB_K2 = OUT_ROOT / "walk_forward_nofallback_k2"
MARKET_DIR_NOFB_K2 = OUT_ROOT / "market_appendix_nofallback_k2"
WF_DIR_NOFB_K4 = OUT_ROOT / "walk_forward_nofallback_k4"
MARKET_DIR_NOFB_K4 = OUT_ROOT / "market_appendix_nofallback_k4"

LABELS_K2 = ["bear", "bull"]
# Ascending mean order ≈ Maheu et al. (2012, 2021) taxonomy
LABELS_K4 = ["bear", "bear_rally", "bull_correction", "bull"]

WALK_WINDOWS = [
    (2002, 2011, 2012),
    (2003, 2012, 2013),
    (2004, 2013, 2014),
    (2005, 2014, 2015),
    (2006, 2015, 2016),
    (2007, 2016, 2017),
    (2008, 2017, 2018),
    (2009, 2018, 2019),
    (2010, 2019, 2020),
    (2011, 2020, 2021),
    (2012, 2021, 2022),
    (2013, 2022, 2023),
    (2014, 2023, 2024),
    (2015, 2024, 2025),
]

FF5_SUMMARY = (
    MANUAL_ROOT / "final test results" / "ff5" / "walk_forward_baseline"
    / "walk_forward_baseline_summary.csv"
)
XGB_SUMMARY = (
    MANUAL_ROOT / "final test results" / "xgboost 2" / "walk_forward_baseline"
    / "walk_forward_baseline_summary.csv"
)
LSTM_SUMMARY = (
    MANUAL_ROOT / "saved_data" / "lstm_walk_forward" / "walk_forward_baseline"
    / "summary_metrics.csv"
)

log = logging.getLogger("markov")


def configure_run(
    k_regimes: int,
    window: str = "rolling",
    train_assign: str = "soft",
    train_prob_source: str = "filtered",
    market_series: str = "mkt_rf",
    allow_pooled_fallback: bool = True,
) -> None:
    """Set globals and output folders. One sensitivity change at a time."""
    global K_REGIMES, WINDOW_MODE, TRAIN_ASSIGN, TRAIN_PROB_SOURCE, MARKET_SERIES
    global ALLOW_POOLED_FALLBACK
    global WF_DIR, MARKET_DIR, MIN_HIST_MONTHS
    global MIN_REGIME_WEIGHT_SUM, MIN_REGIME_MONTHS, MS_MAXITER, REGIME_LABELS

    k = int(k_regimes)
    w = str(window).strip().lower()
    a = str(train_assign).strip().lower()
    ps = str(train_prob_source).strip().lower()
    ms = str(market_series).strip().lower()
    fb = bool(allow_pooled_fallback)
    if k not in (2, 4):
        raise ValueError("Only k_regimes=2 (Hamilton) or 4 (Maheu-style) are supported")
    if w not in ("rolling", "expanding"):
        raise ValueError("window must be 'rolling' or 'expanding'")
    if a not in ("soft", "hard"):
        raise ValueError("train_assign must be 'soft' or 'hard'")
    if ps not in ("filtered", "smoothed"):
        raise ValueError("train_prob_source must be 'filtered' or 'smoothed'")
    if ms not in ("mkt_rf", "ew_excess"):
        raise ValueError("market_series must be 'mkt_rf' or 'ew_excess'")
    if a == "hard" and w != "rolling":
        raise ValueError("R1 hard assignment is tested on rolling windows only")
    if ps == "smoothed" and (w != "rolling" or a != "soft" or ms != "mkt_rf"):
        raise ValueError("R2 smoothed requires rolling + soft + mkt_rf")
    if a == "hard" and (ps != "filtered" or ms != "mkt_rf"):
        raise ValueError("R1 hard requires filtered + mkt_rf")
    if ms == "ew_excess" and (w != "rolling" or a != "soft" or ps != "filtered"):
        raise ValueError("R4 ew_excess requires --window rolling --train-assign soft --train-prob-source filtered")
    if (not fb) and (w != "rolling" or a != "soft" or ps != "filtered" or ms != "mkt_rf"):
        raise ValueError(
            "R5 --no-pooled-fallback requires rolling + soft + filtered + mkt_rf"
        )

    K_REGIMES = k
    WINDOW_MODE = w
    TRAIN_ASSIGN = a
    TRAIN_PROB_SOURCE = ps
    MARKET_SERIES = ms
    ALLOW_POOLED_FALLBACK = fb

    if k == 2:
        REGIME_LABELS = list(LABELS_K2)
        MIN_HIST_MONTHS = 24
        MIN_REGIME_WEIGHT_SUM = 200.0
        MIN_REGIME_MONTHS = 12
        MS_MAXITER = 200
    else:
        REGIME_LABELS = list(LABELS_K4)
        MIN_HIST_MONTHS = 48
        MIN_REGIME_WEIGHT_SUM = 100.0
        MIN_REGIME_MONTHS = 6
        MS_MAXITER = 500

    if not fb:
        WF_DIR, MARKET_DIR = (
            (WF_DIR_NOFB_K2, MARKET_DIR_NOFB_K2) if k == 2
            else (WF_DIR_NOFB_K4, MARKET_DIR_NOFB_K4)
        )
    elif ms == "ew_excess":
        WF_DIR, MARKET_DIR = (
            (WF_DIR_EW_K2, MARKET_DIR_EW_K2) if k == 2
            else (WF_DIR_EW_K4, MARKET_DIR_EW_K4)
        )
    elif ps == "smoothed":
        WF_DIR, MARKET_DIR = (
            (WF_DIR_SMOOTH_K2, MARKET_DIR_SMOOTH_K2) if k == 2
            else (WF_DIR_SMOOTH_K4, MARKET_DIR_SMOOTH_K4)
        )
    elif a == "hard":
        WF_DIR, MARKET_DIR = (
            (WF_DIR_HARD_K2, MARKET_DIR_HARD_K2) if k == 2
            else (WF_DIR_HARD_K4, MARKET_DIR_HARD_K4)
        )
    elif w == "expanding":
        WF_DIR, MARKET_DIR = (
            (WF_DIR_EXP_K2, MARKET_DIR_EXP_K2) if k == 2
            else (WF_DIR_EXP_K4, MARKET_DIR_EXP_K4)
        )
    else:
        WF_DIR, MARKET_DIR = (
            (WF_DIR_K2, MARKET_DIR_K2) if k == 2
            else (WF_DIR_K4, MARKET_DIR_K4)
        )


def _regime_labels() -> list[str]:
    return list(REGIME_LABELS)


def _prob_row(prob: np.ndarray) -> dict:
    """Named + indexed regime probability columns for one month."""
    prob = np.asarray(prob, dtype=float).reshape(-1)
    labels = _regime_labels()
    row = {f"regime_prob_{i}": float(prob[i]) for i in range(len(prob))}
    for i, name in enumerate(labels):
        if i < len(prob):
            row[f"regime_prob_{name}"] = float(prob[i])
    # convenience aliases always present for figures
    if len(prob) >= 2:
        row["regime_prob_bear"] = float(prob[0])
        row["regime_prob_bull"] = float(prob[-1])
    return row


def _reorder_by_mean(mu: np.ndarray, P: np.ndarray, filtered: np.ndarray) -> tuple:
    """Sort regimes ascending by mean → Maheu/Hamilton label order."""
    order = list(np.argsort(mu))
    mu = mu[order]
    P = P[np.ix_(order, order)]
    filtered = filtered[:, order]
    return mu, P, filtered, order


def _regimes_collapsed(mu: np.ndarray) -> bool:
    if len(mu) < 2:
        return True
    gaps = np.diff(np.sort(mu.astype(float)))
    return bool(np.min(gaps) < COLLAPSE_MU_EPS)


# ---------------------------------------------------------------------------
# Paths / logging / panel
# ---------------------------------------------------------------------------

def _configure_paths() -> None:
    if not MASTER_PANEL.is_file():
        raise FileNotFoundError(f"Master panel not found: {MASTER_PANEL}")
    os.environ["DISSLAB_MASTER_PANEL"] = str(MASTER_PANEL.resolve())
    os.environ["DISSLAB_DATASET_LABEL"] = "final_data_master_panel"
    diss = str(DISS_MODELS.resolve())
    if diss not in sys.path:
        sys.path.insert(0, diss)


def setup_logging(out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / f"markov_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    logger = logging.getLogger("markov")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False
    fmt = logging.Formatter("%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    logger.info("Log file: %s", log_path)
    return log_path


def _filter_members(df: pd.DataFrame) -> pd.DataFrame:
    if not FILTER_MEMBERS or "is_member" not in df.columns:
        return df
    mask = df["is_member"]
    if mask.dtype != bool:
        mask = mask.astype(bool)
    return df.loc[mask].copy()


def _slice_years(df: pd.DataFrame, start_year: int, end_year: int) -> pd.DataFrame:
    m = pd.to_datetime(df["month"])
    lo = pd.Timestamp(year=start_year, month=1, day=1)
    hi = pd.Timestamp(year=end_year, month=12, day=1)
    return df.loc[(m >= lo) & (m <= hi)].copy()


def _slice_eval_year(df: pd.DataFrame, eval_year: int) -> pd.DataFrame:
    m = pd.to_datetime(df["month"])
    lo = pd.Timestamp(year=eval_year, month=1, day=1)
    hi = pd.Timestamp(year=eval_year, month=12, day=1)
    return df.loc[(m >= lo) & (m <= hi)].copy()


def load_panel_members() -> pd.DataFrame:
    _configure_paths()
    from data.build_panel import build_panel

    pdata, _ = build_panel()
    panel = _filter_members(pdata.panel.copy())
    panel["month"] = pd.to_datetime(panel["month"])
    missing = [c for c in FACTOR_COLS + [TARGET_COL, "Ticker", "month"] if c not in panel.columns]
    if missing:
        raise ValueError(f"Panel missing required columns: {missing}")
    return panel


def _month_mkt_series(df: pd.DataFrame) -> pd.Series:
    """Monthly market series used to identify regimes.

    mkt_rf: panel MKT_RF (Ken French Europe / panel factor).
    ew_excess: equal-weight member contemporaneous excess return
               (monthly_return - RF_FF), not next-month target.
    """
    if MARKET_SERIES == "ew_excess":
        need = ["month", "monthly_return"]
        missing = [c for c in need if c not in df.columns]
        if missing:
            raise ValueError(f"EW market series needs columns {missing}")
        work = df[need + (["RF_FF"] if "RF_FF" in df.columns else [])].copy()
        work["month"] = pd.to_datetime(work["month"])
        ret = work["monthly_return"].astype(float)
        if "RF_FF" in work.columns:
            rf = work["RF_FF"].astype(float).fillna(0.0)
            work["_ex"] = ret - rf
        else:
            work["_ex"] = ret
        s = work.groupby("month", sort=True)["_ex"].mean()
        return s.replace([np.inf, -np.inf], np.nan).dropna()

    work = df[["month", "MKT_RF"]].drop_duplicates(subset=["month"]).copy()
    work["month"] = pd.to_datetime(work["month"])
    s = work.set_index("month")["MKT_RF"].astype(float).sort_index()
    return s.replace([np.inf, -np.inf], np.nan).dropna()


# ---------------------------------------------------------------------------
# Markov fit helpers
# ---------------------------------------------------------------------------

def _extract_regime_means(res, mkt_values: np.ndarray) -> np.ndarray:
    params = res.params
    try:
        mu = np.array([float(params[f"const[{i}]"]) for i in range(K_REGIMES)])
        return mu
    except Exception:
        sm_prob = np.asarray(res.smoothed_marginal_probabilities)
        return np.array([
            float(np.average(mkt_values, weights=sm_prob[:, i] + 1e-12))
            for i in range(K_REGIMES)
        ])


def _extract_transition(res) -> np.ndarray:
    """Return left-stochastic P (columns sum to 1): next = P @ prob."""
    P = np.asarray(res.regime_transition, dtype=float)
    if P.ndim == 3:
        P = P[:, :, 0]
    P = P.reshape(K_REGIMES, K_REGIMES)
    col_sums = P.sum(axis=0)
    row_sums = P.sum(axis=1)
    if np.allclose(col_sums, 1.0, atol=0.05):
        return P
    if np.allclose(row_sums, 1.0, atol=0.05):
        return P.T
    # renormalise columns as fallback
    P = P / (col_sums + 1e-12)
    return P


def _last_filtered_prob(res) -> np.ndarray:
    try:
        prob = np.asarray(res.filtered_marginal_probabilities)[-1].astype(float)
    except Exception:
        prob = np.asarray(res.smoothed_marginal_probabilities)[-1].astype(float)
    if prob.ndim > 1:
        prob = prob.flatten()[:K_REGIMES]
    prob = prob[:K_REGIMES].astype(float)
    prob = prob / (prob.sum() + 1e-12)
    return prob


def _month_prob_frame(index, probs: np.ndarray) -> pd.DataFrame:
    """Build month-level regime probability table from (T, k) array."""
    month_probs = pd.DataFrame({"month": index})
    for i in range(K_REGIMES):
        month_probs[f"regime_prob_{i}"] = probs[:, i]
    labels = _regime_labels()
    for i, name in enumerate(labels):
        month_probs[f"regime_prob_{name}"] = probs[:, i]
    month_probs["regime_prob_bear"] = probs[:, 0]
    month_probs["regime_prob_bull"] = probs[:, -1]
    return month_probs


def _fit_markov_mkt(mkt_hist: pd.Series) -> dict:
    """Fit k-regime MS on monthly MKT_RF.

    Returns filtered probs for OOS roll; train month probs use TRAIN_PROB_SOURCE
    (filtered baseline or smoothed for R2).
    """
    if len(mkt_hist) < MIN_HIST_MONTHS:
        raise ValueError(f"Need >= {MIN_HIST_MONTHS} hist months, got {len(mkt_hist)}")

    y = mkt_hist.values.astype(float)
    mod = MarkovRegression(y, k_regimes=K_REGIMES, trend="c", switching_variance=True)
    res = mod.fit(disp=False, maxiter=MS_MAXITER)

    mle_retvals = getattr(res, "mle_retvals", None) or {}
    ms_converged = bool(mle_retvals.get("converged", True))
    if not ms_converged:
        log.warning(
            "Markov MLE did not converge (k=%d, n=%d months, maxiter=%d) | mle_retvals=%s",
            K_REGIMES,
            len(mkt_hist),
            MS_MAXITER,
            mle_retvals,
        )

    mu = _extract_regime_means(res, y)
    P = _extract_transition(res)
    filtered = np.asarray(res.filtered_marginal_probabilities, dtype=float)
    smoothed = np.asarray(res.smoothed_marginal_probabilities, dtype=float)
    if filtered.ndim != 2 or filtered.shape[1] != K_REGIMES:
        raise ValueError(f"Unexpected filtered probs shape: {filtered.shape}")
    if smoothed.ndim != 2 or smoothed.shape[1] != K_REGIMES:
        raise ValueError(f"Unexpected smoothed probs shape: {smoothed.shape}")

    mu, P, filtered, order = _reorder_by_mean(mu, P, filtered)
    smoothed = smoothed[:, order]

    # OOS start always from end-of-train FILTERED probs (no look-ahead)
    last_prob = filtered[-1].astype(float)
    last_prob = last_prob / (last_prob.sum() + 1e-12)

    train_arr = smoothed if TRAIN_PROB_SOURCE == "smoothed" else filtered
    month_probs = _month_prob_frame(mkt_hist.index, train_arr)

    collapsed = _regimes_collapsed(mu)
    labels = _regime_labels()
    label_map = {int(order[i]): labels[i] for i in range(K_REGIMES)}
    return {
        "mu": mu,
        "P": P,
        "last_prob": last_prob,
        "month_probs": month_probs,
        "n_hist_months": int(len(mkt_hist)),
        "labels": labels,
        "label_map_from_raw": label_map,
        "bull_index": K_REGIMES - 1,
        "bear_index": 0,
        "collapsed": collapsed,
        "train_prob_source": TRAIN_PROB_SOURCE,
        "ms_converged": ms_converged,
        "result": res,
    }


def _roll_market_forecast(
    mkt_test: pd.Series,
    mu: np.ndarray,
    P: np.ndarray,
    start_prob: np.ndarray,
) -> pd.DataFrame:
    """Hamilton OOS: advance probs with P (no updating on test actuals)."""
    prob = np.asarray(start_prob, dtype=float).copy()
    prob = prob / (prob.sum() + 1e-12)
    rows = []
    for month, actual in mkt_test.items():
        pred = float(prob @ mu)
        row = {
            "month": pd.Timestamp(month).strftime("%Y-%m-%d"),
            "mkt_actual": float(actual),
            "mkt_pred": pred,
            "y_true": float(actual),
            "y_pred": pred,
            "error": float(actual) - pred,
            "abs_error": abs(float(actual) - pred),
        }
        row.update(_prob_row(prob))
        rows.append(row)
        # left-stochastic: next = P @ prob
        prob = P @ prob
        prob = prob / (prob.sum() + 1e-12)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Regime-dependent FF5
# ---------------------------------------------------------------------------

def _prepare_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    cols = FACTOR_COLS + [TARGET_COL, "Ticker", "month"]
    work = df[cols].replace([np.inf, -np.inf], np.nan).dropna(subset=FACTOR_COLS + [TARGET_COL]).copy()
    X = work[FACTOR_COLS].astype(float)
    y = work[TARGET_COL].astype(float)
    return X, y, work


def _fit_pooled_ff5(train: pd.DataFrame):
    X_tr, y_tr, _ = _prepare_xy(train)
    if X_tr.empty:
        raise ValueError("Empty train matrix for pooled FF5")
    X_c = sm.add_constant(X_tr, has_constant="add")
    model = sm.OLS(y_tr, X_c).fit()
    return model


def _fit_weighted_ff5(train: pd.DataFrame, weight_col: str):
    """WLS on rows with finite target/factors; weights from `weight_col`."""
    cols = FACTOR_COLS + [TARGET_COL, "Ticker", "month", weight_col]
    work = train[cols].replace([np.inf, -np.inf], np.nan).dropna(
        subset=FACTOR_COLS + [TARGET_COL]
    ).copy()
    if work.empty:
        raise ValueError("Empty train matrix for weighted FF5")
    w = work[weight_col].astype(float).fillna(0.0).values
    m = np.isfinite(w) & (w > 1e-12)
    if int(m.sum()) < 10:
        raise ValueError("Too few positive weights for WLS")
    X_c = sm.add_constant(work.loc[m, FACTOR_COLS].astype(float), has_constant="add")
    y = work.loc[m, TARGET_COL].astype(float)
    model = sm.WLS(y, X_c, weights=w[m]).fit()
    return model


def _coeffs_dict(model) -> dict:
    return {str(k): float(v) for k, v in model.params.items()}


def _predict_with_model(model, eval_meta: pd.DataFrame) -> np.ndarray:
    X = eval_meta[FACTOR_COLS].astype(float)
    X_c = sm.add_constant(X, has_constant="add")
    X_c = X_c.reindex(columns=model.model.exog_names, fill_value=0.0)
    return np.asarray(model.predict(X_c), dtype=float)


def _attach_month_probs(df: pd.DataFrame, month_probs: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["month"] = pd.to_datetime(work["month"])
    mp = month_probs.copy()
    mp["month"] = pd.to_datetime(mp["month"])
    return work.merge(mp, on="month", how="left")


def _apply_train_assignment(month_probs: pd.DataFrame) -> pd.DataFrame:
    """Return month_probs with regime_weight_* columns for FF5 training.

    soft: weight_k = filtered regime_prob_k
    hard: weight_k = 1 if k = argmax(probs), else 0  (R1)
    """
    mp = month_probs.copy()
    prob_cols = [f"regime_prob_{k}" for k in range(K_REGIMES)]
    missing = [c for c in prob_cols if c not in mp.columns]
    if missing:
        raise ValueError(f"month_probs missing {missing}")
    arr = mp[prob_cols].astype(float).fillna(0.0).values
    if TRAIN_ASSIGN == "hard":
        argmax = np.argmax(arr, axis=1)
        for k in range(K_REGIMES):
            mp[f"regime_weight_{k}"] = (argmax == k).astype(float)
        mp["hard_regime"] = argmax.astype(int)
    else:
        for k in range(K_REGIMES):
            mp[f"regime_weight_{k}"] = arr[:, k]
        mp["hard_regime"] = np.argmax(arr, axis=1).astype(int)
    return mp


def _fit_regime_ff5(
    train: pd.DataFrame,
    month_probs: pd.DataFrame,
) -> tuple[list, dict, str]:
    """Fit regime FF5 with soft or hard train assignment. OOS blend stays soft."""
    _X_tr, _y_tr, meta = _prepare_xy(train)
    mp = _apply_train_assignment(month_probs)
    merged = _attach_month_probs(meta, mp)
    if merged["regime_prob_0"].isna().all():
        raise ValueError("No overlapping months between train panel and regime probs")

    models: list = [None] * K_REGIMES
    coeffs = {f"regime_{k}": None for k in range(K_REGIMES)}
    coeffs["pooled"] = None
    used = {f"regime_{k}": False for k in range(K_REGIMES)}
    fallback_reasons = []
    labels = _regime_labels()
    month_counts = {}

    for k in range(K_REGIMES):
        col = f"regime_weight_{k}"
        w = merged[col].astype(float).fillna(0.0)
        wsum = float(w.sum())
        n_months = int((merged.groupby("month")[col].first() > 0.05).sum())
        month_counts[labels[k]] = n_months
        if wsum < MIN_REGIME_WEIGHT_SUM or n_months < MIN_REGIME_MONTHS:
            fallback_reasons.append(
                f"regime_{k}({labels[k]}): weight_sum={wsum:.1f} n_months={n_months} "
                f"assign={TRAIN_ASSIGN}"
            )
            continue
        try:
            models[k] = _fit_weighted_ff5(merged, col)
            coeffs[f"regime_{k}"] = _coeffs_dict(models[k])
            used[f"regime_{k}"] = True
        except Exception as exc:
            fallback_reasons.append(
                f"regime_{k}({labels[k]}) WLS failed ({TRAIN_ASSIGN}): {exc}"
            )

    if all(m is None for m in models):
        if not ALLOW_POOLED_FALLBACK:
            raise ValueError(
                "R5: all regimes thin/failed and pooled FF5 fallback disabled | "
                + "; ".join(fallback_reasons)
            )
        pooled = _fit_pooled_ff5(train)
        coeffs["pooled"] = _coeffs_dict(pooled)
        return [pooled] * K_REGIMES, {
            "coefficients": coeffs,
            "mode": "pooled_fallback",
            "train_assign": TRAIN_ASSIGN,
            "allow_pooled_fallback": ALLOW_POOLED_FALLBACK,
            "used": used,
            "fallback_reasons": fallback_reasons,
            "labels": labels,
            "train_month_counts": month_counts,
        }, "pooled_fallback"

    if any(m is None for m in models):
        if not ALLOW_POOLED_FALLBACK:
            raise ValueError(
                "R5: partial regime failure and pooled FF5 fallback disabled | "
                + "; ".join(fallback_reasons)
            )
        pooled = _fit_pooled_ff5(train)
        coeffs["pooled"] = _coeffs_dict(pooled)
        for k in range(K_REGIMES):
            if models[k] is None:
                models[k] = pooled
        mode = "partial_fallback"
    else:
        mode = "regime_hard_ols" if TRAIN_ASSIGN == "hard" else "regime_wls"

    return models, {
        "coefficients": coeffs,
        "mode": mode,
        "train_assign": TRAIN_ASSIGN,
        "allow_pooled_fallback": ALLOW_POOLED_FALLBACK,
        "used": used,
        "fallback_reasons": fallback_reasons,
        "labels": labels,
        "train_month_counts": month_counts,
    }, mode


def _oos_regime_probs(
    eval_months: pd.DatetimeIndex | pd.Series,
    start_prob: np.ndarray,
    P: np.ndarray,
) -> pd.DataFrame:
    """Roll start_prob forward per eval month (no test updating).

    First OOS month uses end-of-train filtered probs; then prob <- P @ prob.
    Matches `_roll_market_forecast` / Hamilton stub convention.
    """
    months = pd.to_datetime(pd.Series(eval_months).drop_duplicates()).sort_values()
    prob = np.asarray(start_prob, dtype=float).copy()
    prob = prob / (prob.sum() + 1e-12)
    rows = []
    for month in months:
        row = {"month": pd.Timestamp(month)}
        row.update(_prob_row(prob))
        rows.append(row)
        prob = P @ prob
        prob = prob / (prob.sum() + 1e-12)
    return pd.DataFrame(rows)


def _fit_predict_step(train: pd.DataFrame, eval_df: pd.DataFrame) -> tuple[pd.DataFrame, dict, dict]:
    """Regime-dependent FF5 for one walk-forward step."""
    mkt_hist = _month_mkt_series(train)
    ms = _fit_markov_mkt(mkt_hist)
    if ms["collapsed"]:
        log.warning("  Regime means nearly equal — identification weak")

    models, coeff_meta, mode = _fit_regime_ff5(train, ms["month_probs"])

    X_ev, y_ev, meta = _prepare_xy(eval_df)
    if X_ev.empty:
        raise ValueError("Empty eval matrix after dropping NaNs")

    oos_probs = _oos_regime_probs(meta["month"], ms["last_prob"], ms["P"])
    meta = _attach_month_probs(meta, oos_probs)

    preds = [_predict_with_model(models[k], meta) for k in range(K_REGIMES)]
    probs = [meta[f"regime_prob_{k}"].astype(float).values for k in range(K_REGIMES)]
    y_pred = np.zeros(len(meta), dtype=float)
    for k in range(K_REGIMES):
        y_pred += probs[k] * preds[k]

    frame = pd.DataFrame({
        "Ticker": meta["Ticker"].astype(str).values,
        "month": pd.to_datetime(meta["month"]).dt.strftime("%Y-%m-%d").values,
        "y_true": y_ev.values,
        "y_pred": y_pred,
    })
    for k in range(K_REGIMES):
        frame[f"regime_prob_{k}"] = probs[k]
        frame[f"y_pred_regime_{k}"] = preds[k]
    labels = _regime_labels()
    for k, name in enumerate(labels):
        frame[f"regime_prob_{name}"] = probs[k]
    frame["regime_prob_bear"] = probs[0]
    frame["regime_prob_bull"] = probs[-1]
    for c in FACTOR_COLS:
        frame[c] = meta[c].astype(float).values
    frame["error"] = frame["y_true"] - frame["y_pred"]
    frame["abs_error"] = frame["error"].abs()

    persistence = [float(ms["P"][i, i]) for i in range(K_REGIMES)]
    duration = [float(1.0 / max(1e-12, 1.0 - persistence[i])) for i in range(K_REGIMES)]
    mu_named = {labels[i]: float(ms["mu"][i]) for i in range(K_REGIMES)}
    regime_meta = {
        "k_regimes": K_REGIMES,
        "switching_variance": True,
        "labels": labels,
        "label_scheme": (
            "ascending mean → bear/bull"
            if K_REGIMES == 2
            else "ascending mean → bear/bear_rally/bull_correction/bull (Maheu-style)"
        ),
        "regime_means": [float(x) for x in ms["mu"]],
        "regime_means_named": mu_named,
        "mu_bear": float(ms["mu"][0]),
        "mu_bull": float(ms["mu"][-1]),
        "transition_matrix": [[float(x) for x in row] for row in ms["P"]],
        "persistence": persistence,
        "typical_duration_months": duration,
        "last_filtered_prob": [float(x) for x in ms["last_prob"]],
        "n_hist_months": ms["n_hist_months"],
        "bull_index": K_REGIMES - 1,
        "bear_index": 0,
        "collapsed_regimes": ms["collapsed"],
        "ms_converged": ms.get("ms_converged", True),
        "fit_mode": mode,
        "train_prob_source": TRAIN_PROB_SOURCE,
        "train_assign": TRAIN_ASSIGN,
        "market_series": MARKET_SERIES,
        "allow_pooled_fallback": ALLOW_POOLED_FALLBACK,
    }
    return frame, coeff_meta, regime_meta


def _step_metrics(frame: pd.DataFrame) -> dict:
    from evaluation.predictive import predictive_metrics, ranking_metrics

    pm = predictive_metrics(frame["y_true"].values, frame["y_pred"].values)
    rm = ranking_metrics(frame)
    return {
        "mse": pm.get("mse"),
        "mae": pm.get("mae"),
        "rmse": pm.get("rmse"),
        "r2": pm.get("r2"),
        "n_obs": pm.get("n_obs"),
        "ic_mean": rm.get("ic_mean"),
        "ic_ir": rm.get("ic_ir"),
        "long_short_mean": rm.get("long_short_mean"),
        "n_months": rm.get("n_months"),
    }


def pooled_metrics(df: pd.DataFrame) -> dict:
    from evaluation.predictive import predictive_metrics, ranking_metrics

    if df is None or df.empty:
        return {
            "mse": None, "mae": None, "rmse": None, "r2": None, "n_obs": 0,
            "ic_mean": None, "ic_ir": None, "long_short_mean": None,
        }
    pm = predictive_metrics(df["y_true"].values, df["y_pred"].values)
    rm = ranking_metrics(df)
    return {
        **pm,
        "ic_mean": rm.get("ic_mean"),
        "ic_ir": rm.get("ic_ir"),
        "long_short_mean": rm.get("long_short_mean"),
        "n_months": rm.get("n_months"),
    }


def _append_tracker(row: dict) -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    fields = [
        "timestamp", "mode", "k_regimes", "window", "train_assign",
        "train_prob_source", "market_series", "allow_pooled_fallback",
        "rmse", "mae", "mse", "r2",
        "ic_mean", "long_short_mean", "n_obs", "n_steps_ok", "output_dir",
    ]
    write_header = not RESULTS_TRACKER.is_file()
    with RESULTS_TRACKER.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow({k: row.get(k) for k in fields})
    log.info("Appended tracker row -> %s", RESULTS_TRACKER)


def _json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return None if not np.isfinite(v) else v
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if obj is None:
        return None
    if isinstance(obj, str):
        return obj
    return str(obj)


# ---------------------------------------------------------------------------
# Walk-forward + market appendix
# ---------------------------------------------------------------------------

def run_walk_forward_markov(
    panel: pd.DataFrame | None = None,
    *,
    stock_level: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, dict, pd.DataFrame]:
    """Run walk-forward. Returns (summary, pooled_stock, pooled_metrics, market_pooled)."""
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    WF_DIR.mkdir(parents=True, exist_ok=True)
    MARKET_DIR.mkdir(parents=True, exist_ok=True)
    if panel is None:
        panel = load_panel_members()

    log.info(
        "Panel rows=%d | months %s -> %s | factors=%s | stock_level=%s | "
        "k=%d | window=%s | assign=%s | train_probs=%s | market=%s | fallback=%s",
        len(panel),
        panel["month"].min().date(),
        panel["month"].max().date(),
        FACTOR_COLS,
        stock_level,
        K_REGIMES,
        WINDOW_MODE,
        TRAIN_ASSIGN,
        TRAIN_PROB_SOURCE,
        MARKET_SERIES,
        ALLOW_POOLED_FALLBACK,
    )

    rows: list[dict] = []
    pooled_parts: list[pd.DataFrame] = []
    market_parts: list[pd.DataFrame] = []
    regime_summary_rows: list[dict] = []

    for i, (roll_tr0, roll_tr1, ev_year) in enumerate(WALK_WINDOWS, start=1):
        if WINDOW_MODE == "expanding":
            tr0, tr1 = EXPAND_START_YEAR, ev_year - 1
        else:
            tr0, tr1 = roll_tr0, roll_tr1
        step_tag = f"step_{i:02d}_{ev_year}"
        step_dir = WF_DIR / step_tag
        step_dir.mkdir(parents=True, exist_ok=True)
        mkt_step_dir = MARKET_DIR / step_tag
        mkt_step_dir.mkdir(parents=True, exist_ok=True)

        train = _slice_years(panel, tr0, tr1)
        eval_df = _slice_eval_year(panel, ev_year)
        log.info(
            "[%s] %s train %d-%d (%d rows) -> eval %d (%d rows)",
            step_tag, WINDOW_MODE, tr0, tr1, len(train), ev_year, len(eval_df),
        )

        if train.empty or eval_df.empty:
            rows.append({
                "step": i, "eval_year": ev_year, "train_start": tr0, "train_end": tr1,
                "n_train": len(train), "n_eval": 0,
                "rmse": np.nan, "mae": np.nan, "mse": np.nan, "r2": np.nan,
                "ic_mean": np.nan, "ic_ir": np.nan, "long_short_mean": np.nan,
                "fit_mode": "", "status": "skipped_empty",
            })
            continue

        # --- Market appendix (always) ---
        try:
            mkt_hist = _month_mkt_series(train)
            ms = _fit_markov_mkt(mkt_hist)
            mkt_test = _month_mkt_series(eval_df)
            mkt_frame = _roll_market_forecast(
                mkt_test, ms["mu"], ms["P"], ms["last_prob"],
            )
            mkt_frame["eval_year"] = ev_year
            market_parts.append(mkt_frame)
            mkt_frame.to_csv(mkt_step_dir / "market_forecast.csv", index=False)
            mkt_metrics = {}
            if len(mkt_frame):
                from evaluation.predictive import predictive_metrics as _pm
                mkt_metrics = _pm(mkt_frame["y_true"].values, mkt_frame["y_pred"].values)
            labels = _regime_labels()
            mu_named = {labels[j]: float(ms["mu"][j]) for j in range(K_REGIMES)}
            (mkt_step_dir / "regime.json").write_text(
                json.dumps(_json_safe({
                    "eval_year": ev_year,
                    "k_regimes": K_REGIMES,
                    "labels": labels,
                    "regime_means": [float(x) for x in ms["mu"]],
                    "regime_means_named": mu_named,
                    "mu_bear": float(ms["mu"][0]),
                    "mu_bull": float(ms["mu"][-1]),
                    "transition_matrix": [[float(x) for x in row] for row in ms["P"]],
                    "persistence": [float(ms["P"][j, j]) for j in range(K_REGIMES)],
                    "typical_duration_months": [
                        float(1.0 / max(1e-12, 1.0 - ms["P"][j, j])) for j in range(K_REGIMES)
                    ],
                    "last_filtered_prob": [float(x) for x in ms["last_prob"]],
                    "n_hist_months": ms["n_hist_months"],
                    "collapsed_regimes": ms["collapsed"],
                    "market_metrics": mkt_metrics,
                }), indent=2),
                encoding="utf-8",
            )
            reg_row = {
                "step": i,
                "eval_year": ev_year,
                "k_regimes": K_REGIMES,
                "mu_bear": float(ms["mu"][0]),
                "mu_bull": float(ms["mu"][-1]),
                "p_bear_bear": float(ms["P"][0, 0]),
                "p_bull_bull": float(ms["P"][-1, -1]),
                "duration_bear": float(1.0 / max(1e-12, 1.0 - ms["P"][0, 0])),
                "duration_bull": float(1.0 / max(1e-12, 1.0 - ms["P"][-1, -1])),
                "n_hist_months": ms["n_hist_months"],
                "market_rmse": mkt_metrics.get("rmse"),
                "collapsed": ms["collapsed"],
                "ms_converged": ms.get("ms_converged", True),
            }
            for j, name in enumerate(labels):
                reg_row[f"mu_{name}"] = float(ms["mu"][j])
                reg_row[f"persist_{name}"] = float(ms["P"][j, j])
                reg_row[f"duration_{name}"] = float(
                    1.0 / max(1e-12, 1.0 - ms["P"][j, j])
                )
            regime_summary_rows.append(reg_row)
        except Exception as exc:
            log.exception("  Market appendix FAILED: %s", exc)
            if not stock_level:
                rows.append({
                    "step": i, "eval_year": ev_year, "train_start": tr0, "train_end": tr1,
                    "n_train": len(train), "n_eval": len(eval_df),
                    "rmse": np.nan, "mae": np.nan, "mse": np.nan, "r2": np.nan,
                    "ic_mean": np.nan, "ic_ir": np.nan, "long_short_mean": np.nan,
                    "fit_mode": "", "status": f"error: {exc}",
                })
                continue
            mkt_frame = pd.DataFrame()
            mkt_metrics = {}

        if not stock_level:
            rows.append({
                "step": i, "eval_year": ev_year, "train_start": tr0, "train_end": tr1,
                "n_train": len(train), "n_eval": int(len(mkt_frame)),
                "rmse": mkt_metrics.get("rmse"),
                "mae": mkt_metrics.get("mae"),
                "mse": mkt_metrics.get("mse"),
                "r2": mkt_metrics.get("r2"),
                "ic_mean": np.nan, "ic_ir": np.nan, "long_short_mean": np.nan,
                "fit_mode": "market_only", "status": "ok",
            })
            continue

        # --- Stock-level regime FF5 ---
        try:
            frame, coeff_meta, regime_meta = _fit_predict_step(train, eval_df)
            metrics = _step_metrics(frame)
        except Exception as exc:
            log.exception("  FAILED: %s", exc)
            if not ALLOW_POOLED_FALLBACK:
                rows.append({
                    "step": i, "eval_year": ev_year, "train_start": tr0, "train_end": tr1,
                    "n_train": len(train), "n_eval": len(eval_df),
                    "rmse": np.nan, "mae": np.nan, "mse": np.nan, "r2": np.nan,
                    "ic_mean": np.nan, "ic_ir": np.nan, "long_short_mean": np.nan,
                    "fit_mode": "no_fallback", "status": f"error: {exc}",
                })
                log.warning("  R5: skipped step_pooled_fallback for %s", step_tag)
                continue
            # pooled FF5 fallback for the whole step
            try:
                X_tr, y_tr, _ = _prepare_xy(train)
                X_ev, y_ev, meta = _prepare_xy(eval_df)
                X_tr_c = sm.add_constant(X_tr, has_constant="add")
                model = sm.OLS(y_tr, X_tr_c).fit()
                X_ev_c = sm.add_constant(X_ev, has_constant="add")
                X_ev_c = X_ev_c.reindex(columns=X_tr_c.columns, fill_value=0.0)
                pred = np.asarray(model.predict(X_ev_c), dtype=float)
                frame = pd.DataFrame({
                    "Ticker": meta["Ticker"].astype(str).values,
                    "month": pd.to_datetime(meta["month"]).dt.strftime("%Y-%m-%d").values,
                    "y_true": y_ev.values,
                    "y_pred": pred,
                    "regime_prob_bear": np.nan,
                    "regime_prob_bull": np.nan,
                })
                for k in range(K_REGIMES):
                    frame[f"regime_prob_{k}"] = np.nan
                    frame[f"y_pred_regime_{k}"] = pred
                for c in FACTOR_COLS:
                    frame[c] = meta[c].astype(float).values
                frame["error"] = frame["y_true"] - frame["y_pred"]
                frame["abs_error"] = frame["error"].abs()
                metrics = _step_metrics(frame)
                coeff_meta = {
                    "coefficients": {"pooled": _coeffs_dict(model)},
                    "mode": "step_pooled_fallback",
                    "used": {},
                    "fallback_reasons": [str(exc)],
                }
                regime_meta = {
                    "fit_mode": "step_pooled_fallback",
                    "error": str(exc),
                    "k_regimes": K_REGIMES,
                }
                log.warning("  Used pooled FF5 fallback for %s", step_tag)
            except Exception as exc2:
                log.exception("  Fallback also FAILED: %s", exc2)
                rows.append({
                    "step": i, "eval_year": ev_year, "train_start": tr0, "train_end": tr1,
                    "n_train": len(train), "n_eval": len(eval_df),
                    "rmse": np.nan, "mae": np.nan, "mse": np.nan, "r2": np.nan,
                    "ic_mean": np.nan, "ic_ir": np.nan, "long_short_mean": np.nan,
                    "fit_mode": "", "status": f"error: {exc2}",
                })
                continue

        frame = frame.copy()
        frame["eval_year"] = ev_year
        pooled_parts.append(frame)

        preview_cols = [
            "Ticker", "month", "y_true", "y_pred", "error", "abs_error",
            *[f"regime_prob_{k}" for k in range(K_REGIMES)],
            "regime_prob_bear", "regime_prob_bull",
            *FACTOR_COLS,
        ]
        preview_cols = [c for c in preview_cols if c in frame.columns]
        frame[preview_cols].to_csv(step_dir / "preview.csv", index=False)
        blend_formula = " + ".join([f"p{k}*X*b{k}" for k in range(K_REGIMES)])
        (step_dir / "coefficients.json").write_text(
            json.dumps(_json_safe({
                "formula": (
                    f"target_excess_return ~ regime-weighted FF5 (k={K_REGIMES}); "
                    f"y_pred = {blend_formula}"
                ),
                "k_regimes": K_REGIMES,
                "labels": _regime_labels(),
                "n_train": int(len(train)),
                "n_eval": int(len(frame)),
                **coeff_meta,
            }), indent=2),
            encoding="utf-8",
        )
        (step_dir / "regime.json").write_text(
            json.dumps(_json_safe(regime_meta), indent=2),
            encoding="utf-8",
        )
        (step_dir / "metrics.json").write_text(
            json.dumps(_json_safe({
                "step": i,
                "eval_year": ev_year,
                "train_start": tr0,
                "train_end": tr1,
                "n_train": int(len(train)),
                "n_eval": int(len(frame)),
                **metrics,
                "fit_mode": regime_meta.get("fit_mode") or coeff_meta.get("mode"),
                "status": "ok",
            }), indent=2),
            encoding="utf-8",
        )

        rows.append({
            "step": i,
            "eval_year": ev_year,
            "train_start": tr0,
            "train_end": tr1,
            "n_train": int(len(train)),
            "n_eval": int(len(frame)),
            "rmse": metrics.get("rmse"),
            "mae": metrics.get("mae"),
            "mse": metrics.get("mse"),
            "r2": metrics.get("r2"),
            "ic_mean": metrics.get("ic_mean"),
            "ic_ir": metrics.get("ic_ir"),
            "long_short_mean": metrics.get("long_short_mean"),
            "fit_mode": regime_meta.get("fit_mode") or coeff_meta.get("mode"),
            "status": "ok",
        })
        log.info(
            "  RMSE=%.4f IC=%s n=%d mode=%s",
            metrics.get("rmse") or float("nan"),
            f"{metrics.get('ic_mean'):.4f}" if metrics.get("ic_mean") is not None else "n/a",
            metrics.get("n_obs") or 0,
            regime_meta.get("fit_mode") or coeff_meta.get("mode"),
        )

    summary = pd.DataFrame(rows)
    ok = summary[summary["status"] == "ok"] if len(summary) else summary
    avg = {
        "step": 0,
        "eval_year": "AVERAGE",
        "train_start": "",
        "train_end": "",
        "n_train": float(ok["n_train"].mean()) if len(ok) else np.nan,
        "n_eval": float(ok["n_eval"].mean()) if len(ok) else np.nan,
        "rmse": float(ok["rmse"].mean()) if len(ok) else np.nan,
        "mae": float(ok["mae"].mean()) if len(ok) else np.nan,
        "mse": float(ok["mse"].mean()) if len(ok) else np.nan,
        "r2": float(ok["r2"].mean()) if len(ok) else np.nan,
        "ic_mean": float(ok["ic_mean"].mean()) if len(ok) and ok["ic_mean"].notna().any() else np.nan,
        "ic_ir": float(ok["ic_ir"].mean()) if len(ok) and ok["ic_ir"].notna().any() else np.nan,
        "long_short_mean": (
            float(ok["long_short_mean"].mean())
            if len(ok) and ok["long_short_mean"].notna().any() else np.nan
        ),
        "fit_mode": "average",
        "status": "average",
    }
    summary = pd.concat([summary, pd.DataFrame([avg])], ignore_index=True)
    summary_path = WF_DIR / "walk_forward_baseline_summary.csv"
    summary.to_csv(summary_path, index=False)
    log.info("Saved summary -> %s", summary_path)

    if pooled_parts:
        pooled = pd.concat(pooled_parts, ignore_index=True)
    else:
        pooled = pd.DataFrame()
    pooled_path = WF_DIR / "pooled_oos_predictions.csv"
    pooled.to_csv(pooled_path, index=False)
    log.info("Saved pooled OOS predictions -> %s (%d rows)", pooled_path, len(pooled))

    if market_parts:
        market_pooled = pd.concat(market_parts, ignore_index=True)
    else:
        market_pooled = pd.DataFrame()
    market_pooled_path = MARKET_DIR / "pooled_market_forecast.csv"
    market_pooled.to_csv(market_pooled_path, index=False)

    # OOS bull-prob path for figures
    if not market_pooled.empty:
        path_cols = [
            c for c in [
                "month", "eval_year", "mkt_actual", "mkt_pred",
                "regime_prob_bear", "regime_prob_bull", "error", "abs_error",
            ] if c in market_pooled.columns
        ]
        market_pooled[path_cols].to_csv(MARKET_DIR / "regime_prob_path.csv", index=False)

    if regime_summary_rows:
        reg_df = pd.DataFrame(regime_summary_rows)
        reg_df.to_csv(MARKET_DIR / "regime_params_by_step.csv", index=False)
        labels = _regime_labels()
        avg_reg = {
            "k_regimes": K_REGIMES,
            "labels": labels,
            "mu_bear_mean": float(reg_df["mu_bear"].mean()),
            "mu_bull_mean": float(reg_df["mu_bull"].mean()),
            "persistence_bear_mean": float(reg_df["p_bear_bear"].mean()),
            "persistence_bull_mean": float(reg_df["p_bull_bull"].mean()),
            "duration_bear_mean": float(reg_df["duration_bear"].mean()),
            "duration_bull_mean": float(reg_df["duration_bull"].mean()),
            "market_rmse_mean": float(reg_df["market_rmse"].mean())
            if reg_df["market_rmse"].notna().any() else None,
            "n_steps": int(len(reg_df)),
            "n_collapsed": int(reg_df["collapsed"].sum()),
        }
        for name in labels:
            mc, pc, dc = f"mu_{name}", f"persist_{name}", f"duration_{name}"
            if mc in reg_df.columns:
                avg_reg[f"{mc}_mean"] = float(reg_df[mc].mean())
            if pc in reg_df.columns:
                avg_reg[f"{pc}_mean"] = float(reg_df[pc].mean())
            if dc in reg_df.columns:
                avg_reg[f"{dc}_mean"] = float(reg_df[dc].mean())
        (MARKET_DIR / "regime_table_summary.json").write_text(
            json.dumps(_json_safe(avg_reg), indent=2), encoding="utf-8",
        )
        # market RMSE by year
        keep_cols = ["eval_year", "market_rmse", "mu_bear", "mu_bull"] + [
            f"mu_{n}" for n in labels if f"mu_{n}" in reg_df.columns
        ]
        mkt_year = reg_df[keep_cols].copy()
        mkt_year.to_csv(MARKET_DIR / "market_rmse_by_year.csv", index=False)

    pm = pooled_metrics(pooled) if stock_level else {}
    if stock_level and pooled is not None and not pooled.empty:
        log.info(
            "POOLED OOS | RMSE=%.4f MAE=%.4f R2=%.4f IC=%s n=%s",
            pm.get("rmse") or float("nan"),
            pm.get("mae") or float("nan"),
            pm.get("r2") or float("nan"),
            f"{pm.get('ic_mean'):.4f}" if pm.get("ic_mean") is not None else "n/a",
            pm.get("n_obs"),
        )

    _fb_tag = "fallback" if ALLOW_POOLED_FALLBACK else "nofallback"
    _append_tracker({
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "mode": (
            f"market_only_k{K_REGIMES}_{WINDOW_MODE}_{TRAIN_ASSIGN}_{TRAIN_PROB_SOURCE}_{MARKET_SERIES}_{_fb_tag}"
            if not stock_level
            else f"walk_forward_regime_ff5_k{K_REGIMES}_{WINDOW_MODE}_{TRAIN_ASSIGN}_{TRAIN_PROB_SOURCE}_{MARKET_SERIES}_{_fb_tag}"
        ),
        "k_regimes": K_REGIMES,
        "window": WINDOW_MODE,
        "train_assign": TRAIN_ASSIGN,
        "train_prob_source": TRAIN_PROB_SOURCE,
        "market_series": MARKET_SERIES,
        "allow_pooled_fallback": ALLOW_POOLED_FALLBACK,
        "rmse": pm.get("rmse") if stock_level else (
            float(ok["rmse"].mean()) if len(ok) else None
        ),
        "mae": pm.get("mae") if stock_level else None,
        "mse": pm.get("mse") if stock_level else None,
        "r2": pm.get("r2") if stock_level else None,
        "ic_mean": pm.get("ic_mean") if stock_level else None,
        "long_short_mean": pm.get("long_short_mean") if stock_level else None,
        "n_obs": pm.get("n_obs") if stock_level else None,
        "n_steps_ok": int((summary["status"] == "ok").sum()),
        "output_dir": str(WF_DIR if stock_level else MARKET_DIR),
    })

    return summary, pooled, pm, market_pooled


# ---------------------------------------------------------------------------
# H1 tables
# ---------------------------------------------------------------------------

def _safe_float(x) -> float | None:
    try:
        if x is None or (isinstance(x, float) and not np.isfinite(x)):
            return None
        if pd.isna(x):
            return None
        return float(x)
    except Exception:
        return None


def _year_key(x) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return ""
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    s = str(x).strip().upper()
    if s in {"AVERAGE", "AVG", "NAN", "MEAN", "MEDIAN", ""}:
        return s
    try:
        return str(int(float(s)))
    except Exception:
        return s


def _fmt(x, nd: int = 4) -> str:
    v = _safe_float(x)
    if v is None:
        return "-"
    return f"{v:.{nd}f}"


def _avg_row(summary: pd.DataFrame) -> pd.Series | None:
    if summary is None or summary.empty:
        return None
    if "eval_year" in summary.columns:
        m = summary["eval_year"].astype(str).str.upper() == "AVERAGE"
        if m.any():
            return summary.loc[m].iloc[0]
    if "status" in summary.columns:
        m = summary["status"].astype(str).str.lower() == "average"
        if m.any():
            return summary.loc[m].iloc[0]
    return None


def _ok_steps(summary: pd.DataFrame) -> pd.DataFrame:
    if summary is None or summary.empty:
        return pd.DataFrame()
    out = summary.copy()
    if "status" in out.columns:
        out = out[out["status"].astype(str).str.lower() == "ok"]
    if "eval_year" in out.columns:
        out = out[out["eval_year"].map(_year_key).str.fullmatch(r"\d{4}", na=False)]
    return out


def _sum_n_eval(summary: pd.DataFrame) -> int | None:
    ok = _ok_steps(summary)
    if ok.empty:
        return None
    for col in ("n_eval_rows", "n_eval", "n_obs"):
        if col in ok.columns:
            return int(pd.to_numeric(ok[col], errors="coerce").fillna(0).sum())
    return None


def _load_markov_bundle(wf_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Load summary + pooled + metrics from a walk-forward folder."""
    sp = wf_dir / "walk_forward_baseline_summary.csv"
    summary = pd.read_csv(sp) if sp.is_file() else pd.DataFrame()
    pp = wf_dir / "pooled_oos_predictions.csv"
    pooled = pd.read_csv(pp) if pp.is_file() else pd.DataFrame()
    metrics = pooled_metrics(pooled) if not pooled.empty else {}
    return summary, pooled, metrics


def _markov_row_from_bundle(
    model_name: str,
    features: str,
    summary: pd.DataFrame,
    pooled: pd.DataFrame,
    metrics: dict,
) -> dict:
    ok = _ok_steps(summary)
    step_rmse = _safe_float(ok["rmse"].mean()) if len(ok) and "rmse" in ok.columns else None
    step_rmse_med = _safe_float(ok["rmse"].median()) if len(ok) and "rmse" in ok.columns else None
    n_obs = int(len(pooled)) if not pooled.empty else (
        int(metrics.get("n_obs") or 0) if metrics else None
    )
    return {
        "Model": model_name,
        "Features": features,
        "RMSE_mean_year": step_rmse,
        "RMSE_median_year": step_rmse_med,
        "RMSE_pooled": metrics.get("rmse") if metrics else None,
        "MAE": metrics.get("mae") if metrics else None,
        "MSE": metrics.get("mse") if metrics else None,
        "R2": metrics.get("r2") if metrics else None,
        "IC_Spearman": None,
        "Long_short_EW": None,
        "n_obs": n_obs,
    }


def write_h1_comparison(
    markov_pooled_metrics: dict | None = None,
    markov_summary: pd.DataFrame | None = None,
) -> tuple[Path, Path]:
    """Merge Markov k2 (+k4 if present) + FF5 + XGB + LSTM into dissertation tables.

    Always reads frozen k=2 from walk_forward_baseline/. If walk_forward_k4/ exists,
    adds a Maheu-style 4-state row. Active-run override only fills the matching k folder.
    """
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    # --- Markov k=2 (primary / frozen) ---
    mk2_summary, mk2_pooled, mk2_metrics = _load_markov_bundle(WF_DIR_K2)
    if K_REGIMES == 2 and markov_summary is not None:
        mk2_summary = markov_summary
    if K_REGIMES == 2 and markov_pooled_metrics is not None:
        mk2_metrics = markov_pooled_metrics
        if mk2_pooled.empty and (WF_DIR_K2 / "pooled_oos_predictions.csv").is_file():
            mk2_pooled = pd.read_csv(WF_DIR_K2 / "pooled_oos_predictions.csv")

    # --- Markov k=4 (extension) ---
    mk4_summary, mk4_pooled, mk4_metrics = _load_markov_bundle(WF_DIR_K4)
    if K_REGIMES == 4 and markov_summary is not None:
        mk4_summary = markov_summary
    if K_REGIMES == 4 and markov_pooled_metrics is not None:
        mk4_metrics = markov_pooled_metrics
        if mk4_pooled.empty and (WF_DIR_K4 / "pooled_oos_predictions.csv").is_file():
            mk4_pooled = pd.read_csv(WF_DIR_K4 / "pooled_oos_predictions.csv")

    ff5 = pd.read_csv(FF5_SUMMARY) if FF5_SUMMARY.is_file() else pd.DataFrame()
    xgb = pd.read_csv(XGB_SUMMARY) if XGB_SUMMARY.is_file() else pd.DataFrame()
    lstm = pd.read_csv(LSTM_SUMMARY) if LSTM_SUMMARY.is_file() else pd.DataFrame()

    mk2_ok = _ok_steps(mk2_summary)
    mk4_ok = _ok_steps(mk4_summary)
    ff5_ok = _ok_steps(ff5)
    xgb_ok = _ok_steps(xgb)
    lstm_ok = _ok_steps(lstm)
    xgb_avg = _avg_row(xgb)
    lstm_avg = _avg_row(lstm)

    ff5_step_rmse = _safe_float(ff5_ok["rmse"].mean()) if len(ff5_ok) and "rmse" in ff5_ok.columns else None
    ff5_step_rmse_med = _safe_float(ff5_ok["rmse"].median()) if len(ff5_ok) and "rmse" in ff5_ok.columns else None
    lstm_step_rmse = _safe_float(lstm_ok["rmse"].mean()) if len(lstm_ok) and "rmse" in lstm_ok.columns else None
    lstm_step_rmse_med = _safe_float(lstm_ok["rmse"].median()) if len(lstm_ok) and "rmse" in lstm_ok.columns else None
    lstm_step_r2 = _safe_float(lstm_ok["r2"].mean()) if len(lstm_ok) and "r2" in lstm_ok.columns else None

    n_ff5 = _sum_n_eval(ff5)
    n_xgb = _sum_n_eval(xgb)
    n_lstm = _sum_n_eval(lstm)

    rows: list[dict] = []
    if not mk2_summary.empty or not mk2_pooled.empty:
        rows.append(_markov_row_from_bundle(
            "Markov regime-FF5 (k=2 Hamilton)",
            "2-regime MS on MKT_RF + FF5 WLS",
            mk2_summary, mk2_pooled, mk2_metrics,
        ))
    if not mk4_summary.empty or not mk4_pooled.empty:
        rows.append(_markov_row_from_bundle(
            "Markov regime-FF5 (k=4 Maheu-style)",
            "4-regime MS (bear/rally/correction/bull) + FF5 WLS",
            mk4_summary, mk4_pooled, mk4_metrics,
        ))

    rows.extend([
        {
            "Model": "Fama-French 5 (panel OLS)",
            "Features": "5 factors only",
            "RMSE_mean_year": ff5_step_rmse,
            "RMSE_median_year": ff5_step_rmse_med,
            "RMSE_pooled": None,
            "MAE": None,
            "MSE": None,
            "R2": None,
            "IC_Spearman": None,
            "Long_short_EW": None,
            "n_obs": n_ff5,
        },
        {
            "Model": "XGBoost (B_drop_bottom9)",
            "Features": "19 features",
            "RMSE_mean_year": None,
            "RMSE_median_year": None,
            "RMSE_pooled": None,
            "MAE": None,
            "MSE": None,
            "R2": None,
            "IC_Spearman": _safe_float(xgb_avg.get("ic")) if xgb_avg is not None else None,
            "Long_short_EW": _safe_float(xgb_avg.get("long_short")) if xgb_avg is not None else None,
            "n_obs": n_xgb,
        },
        {
            "Model": "LSTM",
            "Features": "same protocol (19 features)",
            "RMSE_mean_year": lstm_step_rmse if lstm_step_rmse is not None else (
                _safe_float(lstm_avg.get("rmse")) if lstm_avg is not None else None
            ),
            "RMSE_median_year": lstm_step_rmse_med,
            "RMSE_pooled": None,
            "MAE": None,
            "MSE": None,
            "R2": lstm_step_r2 if lstm_step_r2 is not None else (
                _safe_float(lstm_avg.get("r2")) if lstm_avg is not None else None
            ),
            "IC_Spearman": _safe_float(lstm_avg.get("ic")) if lstm_avg is not None else None,
            "Long_short_EW": _safe_float(lstm_avg.get("long_short")) if lstm_avg is not None else None,
            "n_obs": n_lstm,
        },
    ])

    # Fill FF5 pooled from final-test pooled file if present
    ff5_pooled_path = (
        MANUAL_ROOT / "final test results" / "ff5" / "walk_forward_baseline"
        / "pooled_oos_predictions.csv"
    )
    if ff5_pooled_path.is_file():
        try:
            ff5_pooled = pd.read_csv(ff5_pooled_path)
            from evaluation.predictive import predictive_metrics as _pm
            fpm = _pm(ff5_pooled["y_true"].values, ff5_pooled["y_pred"].values)
            for r in rows:
                if r["Model"].startswith("Fama-French"):
                    r["RMSE_pooled"] = fpm.get("rmse")
                    r["MAE"] = fpm.get("mae")
                    r["MSE"] = fpm.get("mse")
                    r["R2"] = fpm.get("r2")
                    if r["n_obs"] is None:
                        r["n_obs"] = fpm.get("n_obs")
        except Exception as exc:
            log.warning("Could not load FF5 pooled metrics: %s", exc)

    cmp_csv = OUT_ROOT / "h1_comparison_template.csv"
    pd.DataFrame(rows).to_csv(cmp_csv, index=False)

    years = [ev for (_, _, ev) in WALK_WINDOWS]
    for df in (mk2_ok, mk4_ok, ff5_ok, xgb_ok, lstm_ok):
        if not df.empty and "eval_year" in df.columns:
            df["_yk"] = df["eval_year"].map(_year_key)

    year_rows = []
    for yr in years:
        yk = _year_key(yr)

        def _pick(df, col):
            if df is None or df.empty or "_yk" not in df.columns or col not in df.columns:
                return None
            r = df.loc[df["_yk"] == yk]
            return _safe_float(r.iloc[0][col]) if len(r) else None

        def _pick_n(df, cols=("n_eval", "n_eval_rows", "n_obs")):
            if df is None or df.empty or "_yk" not in df.columns:
                return None
            r = df.loc[df["_yk"] == yk]
            if not len(r):
                return None
            for c in cols:
                if c in r.columns:
                    return int(r.iloc[0][c])
            return None

        year_rows.append({
            "eval_year": yr,
            "Markov_k2_RMSE": _pick(mk2_ok, "rmse"),
            "Markov_k4_RMSE": _pick(mk4_ok, "rmse"),
            "FF5_RMSE": _pick(ff5_ok, "rmse"),
            "LSTM_RMSE": _pick(lstm_ok, "rmse"),
            "XGB_IC": None,
            "LSTM_IC": None,
            "Markov_k2_n": _pick_n(mk2_ok),
            "Markov_k4_n": _pick_n(mk4_ok),
            "FF5_n": _pick_n(ff5_ok),
            "XGB_n": _pick_n(xgb_ok),
            "LSTM_n": _pick_n(lstm_ok),
        })

    for i, yr in enumerate(years):
        yk = _year_key(yr)
        if len(xgb_ok) and "_yk" in xgb_ok.columns:
            r = xgb_ok.loc[xgb_ok["_yk"] == yk]
            if len(r):
                for c in ("ic", "ic_mean"):
                    if c in r.columns:
                        year_rows[i]["XGB_IC"] = _safe_float(r.iloc[0][c])
                        break
        if len(lstm_ok) and "_yk" in lstm_ok.columns:
            r = lstm_ok.loc[lstm_ok["_yk"] == yk]
            if len(r):
                for c in ("ic", "ic_mean"):
                    if c in r.columns:
                        year_rows[i]["LSTM_IC"] = _safe_float(r.iloc[0][c])
                        break

    year_df = pd.DataFrame(year_rows)
    if len(year_df):
        mean_row = {"eval_year": "Mean"}
        med_row = {"eval_year": "Median"}
        for col in year_df.columns:
            if col == "eval_year":
                continue
            if col.endswith("_n"):
                mean_row[col] = float(year_df[col].sum(skipna=True))
                med_row[col] = np.nan
            else:
                mean_row[col] = year_df[col].mean(skipna=True)
                med_row[col] = year_df[col].median(skipna=True)
        year_df = pd.concat([year_df, pd.DataFrame([mean_row, med_row])], ignore_index=True)
    year_csv = OUT_ROOT / "h1_rmse_by_year.csv"
    year_df.to_csv(year_csv, index=False)

    md_lines = [
        "# Table 4.x - Out-of-sample predictive accuracy (walk-forward 2012-2025)",
        "",
        "Protocol: 10-year train -> 1-year eval, annual steps; members-only master panel.",
        "Markov k=2: Hamilton bull/bear MS on MKT_RF + soft-weighted regime FF5 (primary classical).",
        "Markov k=4: Maheu-style labels by sorted means "
        "(bear / bear_rally / bull_correction / bull) + regime FF5 (extension).",
        "FF5: panel OLS of `target_excess_return` on MKT_RF, SMB, HML, RMW, CMA.",
        "RMSE columns: **mean / median across evaluation years** (primary).",
        "",
        "| Model | Features | RMSE (mean yr) | RMSE (median yr) | RMSE pooled | IC (Spearman) | Long-short EW | n_obs |",
        "|-------|----------|----------------|------------------|-------------|---------------|---------------|-------|",
    ]
    for r in rows:
        md_lines.append(
            f"| {r['Model']} | {r['Features']} | {_fmt(r['RMSE_mean_year'])} | {_fmt(r['RMSE_median_year'])} | "
            f"{_fmt(r['RMSE_pooled'])} | {_fmt(r['IC_Spearman'])} | {_fmt(r['Long_short_EW'])} | "
            f"{r['n_obs'] if r['n_obs'] is not None else '-'} |"
        )
    md_lines.extend([
        "",
        "Notes:",
        "- k=2 is the frozen H1 Hamilton baseline; k=4 is the ERP Maheu-motivated extension "
          "(unconstrained MS with mean-sorted labels — not the full hierarchical Maheu sampler).",
        "- Markov/FF5 use market-wide factors => within-month `y_pred` has little cross-section; "
          "compare these on RMSE. Use XGB/LSTM for ranking metrics.",
        "- Market-level Hamilton forecasts: `market_appendix/` (k=2) and `market_appendix_k4/` (k=4).",
        "- Prefer **median year RMSE**; pooled/mean can be dominated by 2014 extremes.",
        "",
        "# Table 4.y - Walk-forward metrics by evaluation year",
        "",
        "| Eval year | Markov k2 RMSE | Markov k4 RMSE | FF5 RMSE | LSTM RMSE | XGB IC | LSTM IC |",
        "|-----------|----------------|----------------|----------|-----------|--------|---------|",
    ])
    for _, r in year_df.iterrows():
        md_lines.append(
            f"| {r['eval_year']} | {_fmt(r.get('Markov_k2_RMSE'))} | {_fmt(r.get('Markov_k4_RMSE'))} | "
            f"{_fmt(r.get('FF5_RMSE'))} | {_fmt(r.get('LSTM_RMSE'))} | "
            f"{_fmt(r.get('XGB_IC'))} | {_fmt(r.get('LSTM_IC'))} |"
        )

    # Regime appendix snippets (k2 + k4 if present)
    for label, mdir in (("k=2", MARKET_DIR_K2), ("k=4", MARKET_DIR_K4)):
        reg_path = mdir / "regime_table_summary.json"
        if not reg_path.is_file():
            continue
        reg = json.loads(reg_path.read_text(encoding="utf-8"))
        md_lines.extend([
            "",
            f"# Appendix - Average regime parameters ({label})",
            "",
        ])
        if reg.get("k_regimes") == 4 and reg.get("labels"):
            names = reg["labels"]
            hdr = "| " + " | ".join([f"mu_{n}" for n in names]) + " | mkt RMSE mean |"
            sep = "|" + "|".join(["---------"] * (len(names) + 1)) + "|"
            vals = "| " + " | ".join([
                _fmt(reg.get(f"mu_{n}_mean")) for n in names
            ]) + f" | {_fmt(reg.get('market_rmse_mean'))} |"
            md_lines.extend([hdr, sep, vals])
        else:
            md_lines.extend([
                f"| mu_bear | mu_bull | persist_bear | persist_bull | dur_bear (m) | dur_bull (m) | mkt RMSE mean |",
                f"|---------|---------|--------------|--------------|--------------|--------------|---------------|",
                (
                    f"| {_fmt(reg.get('mu_bear_mean'))} | {_fmt(reg.get('mu_bull_mean'))} | "
                    f"{_fmt(reg.get('persistence_bear_mean'))} | {_fmt(reg.get('persistence_bull_mean'))} | "
                    f"{_fmt(reg.get('duration_bear_mean'))} | {_fmt(reg.get('duration_bull_mean'))} | "
                    f"{_fmt(reg.get('market_rmse_mean'))} |"
                ),
            ])
        md_lines.append(
            "Caveat: market RMSE is month-level MKT_RF forecast error — not comparable 1:1 to stock-month H1 RMSE."
        )

    md_lines.extend([
        "",
        "## Sources",
        f"- Markov k=2: `{WF_DIR_K2}`",
        f"- Markov k=4: `{WF_DIR_K4}`",
        f"- Market appendix k=2: `{MARKET_DIR_K2}`",
        f"- Market appendix k=4: `{MARKET_DIR_K4}`",
        f"- FF5 summary: `{FF5_SUMMARY}`",
        f"- XGB summary: `{XGB_SUMMARY}`",
        f"- LSTM summary: `{LSTM_SUMMARY}`",
        "",
    ])
    md_path = OUT_ROOT / "dissertation_table_h1.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    log.info("Wrote H1 tables -> %s | %s | %s", md_path, cmp_csv, year_csv)
    return md_path, cmp_csv


def write_fallback_comparison() -> Path:
    """R5: baseline (pooled FF5 fallback allowed) vs no-fallback."""
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    bundles = [
        ("k2_fallback", WF_DIR_K2),
        ("k2_nofallback", WF_DIR_NOFB_K2),
        ("k4_fallback", WF_DIR_K4),
        ("k4_nofallback", WF_DIR_NOFB_K4),
    ]
    summary_rows = []
    year_map: dict[str, dict] = {}
    for name, wf in bundles:
        sp = wf / "walk_forward_baseline_summary.csv"
        if not sp.is_file():
            continue
        summary = pd.read_csv(sp)
        ok = _ok_steps(summary)
        pooled_path = wf / "pooled_oos_predictions.csv"
        pooled = pd.read_csv(pooled_path) if pooled_path.is_file() else pd.DataFrame()
        pm = pooled_metrics(pooled) if not pooled.empty else {}
        n_fb = 0
        if "fit_mode" in summary.columns:
            n_fb = int(
                summary["fit_mode"].astype(str).str.contains(
                    "fallback", case=False, na=False
                ).sum()
            )
        summary_rows.append({
            "spec": name,
            "rmse_mean_year": _safe_float(ok["rmse"].mean()) if len(ok) else None,
            "rmse_median_year": _safe_float(ok["rmse"].median()) if len(ok) else None,
            "rmse_pooled": pm.get("rmse"),
            "mae_pooled": pm.get("mae"),
            "n_obs": int(len(pooled)) if not pooled.empty else pm.get("n_obs"),
            "n_steps_ok": int(len(ok)),
            "n_steps_fallback_mode": n_fb,
            "path": str(wf),
        })
        if len(ok) and "eval_year" in ok.columns:
            tmp = ok.copy()
            tmp["_yk"] = tmp["eval_year"].map(_year_key)
            for _, r in tmp.iterrows():
                year_map.setdefault(r["_yk"], {})[name] = _safe_float(r.get("rmse"))

    cmp = pd.DataFrame(summary_rows)
    csv_path = OUT_ROOT / "fallback_sensitivity_summary.csv"
    cmp.to_csv(csv_path, index=False)

    years = [str(ev) for (_, _, ev) in WALK_WINDOWS]
    year_rows = []
    for yk in years:
        row = {"eval_year": yk}
        for name, _ in bundles:
            row[f"{name}_RMSE"] = year_map.get(yk, {}).get(name)
        year_rows.append(row)
    year_df = pd.DataFrame(year_rows)
    if len(year_df):
        mean_row = {"eval_year": "Mean"}
        med_row = {"eval_year": "Median"}
        for col in year_df.columns:
            if col == "eval_year":
                continue
            mean_row[col] = year_df[col].mean(skipna=True)
            med_row[col] = year_df[col].median(skipna=True)
        year_df = pd.concat([year_df, pd.DataFrame([mean_row, med_row])], ignore_index=True)
    year_csv = OUT_ROOT / "fallback_sensitivity_by_year.csv"
    year_df.to_csv(year_csv, index=False)

    md = [
        "# R5 Sensitivity — Pooled FF5 fallback on/off (rolling + soft + filtered + MKT_RF)",
        "",
        "fallback (baseline): thin/failed regimes replaced by pooled FF5; step errors can "
        "also fall back to pooled FF5.",
        "nofallback (R5): if any regime is thin/failed, the step errors out (no silent FF5).",
        "Lower RMSE is better. Prefer median year RMSE among completed steps.",
        "",
        "| Spec | RMSE mean yr | RMSE median yr | RMSE pooled | n_obs | n_steps_ok | n_fallback_mode |",
        "|------|--------------|----------------|-------------|-------|------------|-----------------|",
    ]
    for _, r in cmp.iterrows():
        md.append(
            f"| {r['spec']} | {_fmt(r['rmse_mean_year'])} | {_fmt(r['rmse_median_year'])} | "
            f"{_fmt(r['rmse_pooled'])} | {r['n_obs'] if r['n_obs'] is not None else '-'} | "
            f"{r['n_steps_ok']} | {r.get('n_steps_fallback_mode', '-') } |"
        )
    md.extend([
        "",
        "## By evaluation year",
        "",
        "| Eval year | k2 fallback | k2 nofallback | k4 fallback | k4 nofallback |",
        "|-----------|-------------|---------------|-------------|---------------|",
    ])
    for _, r in year_df.iterrows():
        md.append(
            f"| {r['eval_year']} | {_fmt(r.get('k2_fallback_RMSE'))} | "
            f"{_fmt(r.get('k2_nofallback_RMSE'))} | {_fmt(r.get('k4_fallback_RMSE'))} | "
            f"{_fmt(r.get('k4_nofallback_RMSE'))} |"
        )
    md.extend(["", f"CSV: `{csv_path.name}`, `{year_csv.name}`", ""])
    md_path = OUT_ROOT / "fallback_sensitivity.md"
    md_path.write_text("\n".join(md), encoding="utf-8")
    log.info("Wrote fallback sensitivity -> %s | %s | %s", md_path, csv_path, year_csv)
    return md_path


def write_market_series_comparison() -> Path:
    """R4: MKT_RF vs EW member excess for regime identification."""
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    bundles = [
        ("k2_mkt_rf", WF_DIR_K2),
        ("k2_ew_excess", WF_DIR_EW_K2),
        ("k4_mkt_rf", WF_DIR_K4),
        ("k4_ew_excess", WF_DIR_EW_K4),
    ]
    summary_rows = []
    year_map: dict[str, dict] = {}
    for name, wf in bundles:
        sp = wf / "walk_forward_baseline_summary.csv"
        if not sp.is_file():
            continue
        summary = pd.read_csv(sp)
        ok = _ok_steps(summary)
        pooled_path = wf / "pooled_oos_predictions.csv"
        pooled = pd.read_csv(pooled_path) if pooled_path.is_file() else pd.DataFrame()
        pm = pooled_metrics(pooled) if not pooled.empty else {}
        summary_rows.append({
            "spec": name,
            "rmse_mean_year": _safe_float(ok["rmse"].mean()) if len(ok) else None,
            "rmse_median_year": _safe_float(ok["rmse"].median()) if len(ok) else None,
            "rmse_pooled": pm.get("rmse"),
            "mae_pooled": pm.get("mae"),
            "n_obs": int(len(pooled)) if not pooled.empty else pm.get("n_obs"),
            "n_steps_ok": int(len(ok)),
            "path": str(wf),
        })
        if len(ok) and "eval_year" in ok.columns:
            tmp = ok.copy()
            tmp["_yk"] = tmp["eval_year"].map(_year_key)
            for _, r in tmp.iterrows():
                year_map.setdefault(r["_yk"], {})[name] = _safe_float(r.get("rmse"))

    cmp = pd.DataFrame(summary_rows)
    csv_path = OUT_ROOT / "market_series_sensitivity_summary.csv"
    cmp.to_csv(csv_path, index=False)

    years = [str(ev) for (_, _, ev) in WALK_WINDOWS]
    year_rows = []
    for yk in years:
        row = {"eval_year": yk}
        for name, _ in bundles:
            row[f"{name}_RMSE"] = year_map.get(yk, {}).get(name)
        year_rows.append(row)
    year_df = pd.DataFrame(year_rows)
    if len(year_df):
        mean_row = {"eval_year": "Mean"}
        med_row = {"eval_year": "Median"}
        for col in year_df.columns:
            if col == "eval_year":
                continue
            mean_row[col] = year_df[col].mean(skipna=True)
            med_row[col] = year_df[col].median(skipna=True)
        year_df = pd.concat([year_df, pd.DataFrame([mean_row, med_row])], ignore_index=True)
    year_csv = OUT_ROOT / "market_series_sensitivity_by_year.csv"
    year_df.to_csv(year_csv, index=False)

    md = [
        "# R4 Sensitivity — Market series for regime identification (rolling + soft + filtered)",
        "",
        "mkt_rf (baseline): Markov fit on panel `MKT_RF` (Ken French Europe / panel factor).",
        "ew_excess (R4): Markov fit on equal-weight member contemporaneous excess "
        "(`monthly_return - RF_FF`).",
        "FF5 stock forecasts still use MKT_RF/SMB/HML/RMW/CMA as regressors; only the "
        "regime-state series changes. Lower RMSE is better.",
        "",
        "| Spec | RMSE mean yr | RMSE median yr | RMSE pooled | n_obs | n_steps |",
        "|------|--------------|----------------|-------------|-------|---------|",
    ]
    for _, r in cmp.iterrows():
        md.append(
            f"| {r['spec']} | {_fmt(r['rmse_mean_year'])} | {_fmt(r['rmse_median_year'])} | "
            f"{_fmt(r['rmse_pooled'])} | {r['n_obs'] if r['n_obs'] is not None else '-'} | "
            f"{r['n_steps_ok']} |"
        )
    md.extend([
        "",
        "## By evaluation year",
        "",
        "| Eval year | k2 MKT_RF | k2 EW excess | k4 MKT_RF | k4 EW excess |",
        "|-----------|-----------|--------------|-----------|--------------|",
    ])
    for _, r in year_df.iterrows():
        md.append(
            f"| {r['eval_year']} | {_fmt(r.get('k2_mkt_rf_RMSE'))} | {_fmt(r.get('k2_ew_excess_RMSE'))} | "
            f"{_fmt(r.get('k4_mkt_rf_RMSE'))} | {_fmt(r.get('k4_ew_excess_RMSE'))} |"
        )
    md.extend(["", f"CSV: `{csv_path.name}`, `{year_csv.name}`", ""])
    md_path = OUT_ROOT / "market_series_sensitivity.md"
    md_path.write_text("\n".join(md), encoding="utf-8")
    log.info("Wrote market-series sensitivity -> %s | %s | %s", md_path, csv_path, year_csv)
    return md_path


def write_prob_comparison() -> Path:
    """R2: filtered vs smoothed train probs (rolling + soft)."""
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    bundles = [
        ("k2_filtered", WF_DIR_K2),
        ("k2_smoothed", WF_DIR_SMOOTH_K2),
        ("k4_filtered", WF_DIR_K4),
        ("k4_smoothed", WF_DIR_SMOOTH_K4),
    ]
    summary_rows = []
    year_map: dict[str, dict] = {}
    for name, wf in bundles:
        sp = wf / "walk_forward_baseline_summary.csv"
        if not sp.is_file():
            continue
        summary = pd.read_csv(sp)
        ok = _ok_steps(summary)
        pooled_path = wf / "pooled_oos_predictions.csv"
        pooled = pd.read_csv(pooled_path) if pooled_path.is_file() else pd.DataFrame()
        pm = pooled_metrics(pooled) if not pooled.empty else {}
        summary_rows.append({
            "spec": name,
            "rmse_mean_year": _safe_float(ok["rmse"].mean()) if len(ok) else None,
            "rmse_median_year": _safe_float(ok["rmse"].median()) if len(ok) else None,
            "rmse_pooled": pm.get("rmse"),
            "mae_pooled": pm.get("mae"),
            "n_obs": int(len(pooled)) if not pooled.empty else pm.get("n_obs"),
            "n_steps_ok": int(len(ok)),
            "path": str(wf),
        })
        if len(ok) and "eval_year" in ok.columns:
            tmp = ok.copy()
            tmp["_yk"] = tmp["eval_year"].map(_year_key)
            for _, r in tmp.iterrows():
                year_map.setdefault(r["_yk"], {})[name] = _safe_float(r.get("rmse"))

    cmp = pd.DataFrame(summary_rows)
    csv_path = OUT_ROOT / "prob_sensitivity_summary.csv"
    cmp.to_csv(csv_path, index=False)

    years = [str(ev) for (_, _, ev) in WALK_WINDOWS]
    year_rows = []
    for yk in years:
        row = {"eval_year": yk}
        for name, _ in bundles:
            row[f"{name}_RMSE"] = year_map.get(yk, {}).get(name)
        year_rows.append(row)
    year_df = pd.DataFrame(year_rows)
    if len(year_df):
        mean_row = {"eval_year": "Mean"}
        med_row = {"eval_year": "Median"}
        for col in year_df.columns:
            if col == "eval_year":
                continue
            mean_row[col] = year_df[col].mean(skipna=True)
            med_row[col] = year_df[col].median(skipna=True)
        year_df = pd.concat([year_df, pd.DataFrame([mean_row, med_row])], ignore_index=True)
    year_csv = OUT_ROOT / "prob_sensitivity_by_year.csv"
    year_df.to_csv(year_csv, index=False)

    md = [
        "# R2 Sensitivity — Filtered vs Smoothed train probabilities (rolling + soft)",
        "",
        "Filtered (baseline): train FF5 weights use filtered regime probs.",
        "Smoothed (R2): train FF5 weights use smoothed regime probs.",
        "OOS always: end-of-train filtered probs rolled with P (no look-ahead).",
        "Lower RMSE is better.",
        "",
        "| Spec | RMSE mean yr | RMSE median yr | RMSE pooled | n_obs | n_steps |",
        "|------|--------------|----------------|-------------|-------|---------|",
    ]
    for _, r in cmp.iterrows():
        md.append(
            f"| {r['spec']} | {_fmt(r['rmse_mean_year'])} | {_fmt(r['rmse_median_year'])} | "
            f"{_fmt(r['rmse_pooled'])} | {r['n_obs'] if r['n_obs'] is not None else '-'} | "
            f"{r['n_steps_ok']} |"
        )
    md.extend([
        "",
        "## By evaluation year",
        "",
        "| Eval year | k2 filtered | k2 smoothed | k4 filtered | k4 smoothed |",
        "|-----------|-------------|-------------|-------------|-------------|",
    ])
    for _, r in year_df.iterrows():
        md.append(
            f"| {r['eval_year']} | {_fmt(r.get('k2_filtered_RMSE'))} | {_fmt(r.get('k2_smoothed_RMSE'))} | "
            f"{_fmt(r.get('k4_filtered_RMSE'))} | {_fmt(r.get('k4_smoothed_RMSE'))} |"
        )
    md.extend(["", f"CSV: `{csv_path.name}`, `{year_csv.name}`", ""])
    md_path = OUT_ROOT / "prob_sensitivity.md"
    md_path.write_text("\n".join(md), encoding="utf-8")
    log.info("Wrote prob sensitivity -> %s | %s | %s", md_path, csv_path, year_csv)
    return md_path


def write_assign_comparison() -> Path:
    """R1: soft vs hard train assignment on rolling windows."""
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    bundles = [
        ("k2_soft", WF_DIR_K2),
        ("k2_hard", WF_DIR_HARD_K2),
        ("k4_soft", WF_DIR_K4),
        ("k4_hard", WF_DIR_HARD_K4),
    ]
    summary_rows = []
    year_map: dict[str, dict] = {}
    for name, wf in bundles:
        sp = wf / "walk_forward_baseline_summary.csv"
        if not sp.is_file():
            continue
        summary = pd.read_csv(sp)
        ok = _ok_steps(summary)
        pooled_path = wf / "pooled_oos_predictions.csv"
        pooled = pd.read_csv(pooled_path) if pooled_path.is_file() else pd.DataFrame()
        pm = pooled_metrics(pooled) if not pooled.empty else {}
        summary_rows.append({
            "spec": name,
            "rmse_mean_year": _safe_float(ok["rmse"].mean()) if len(ok) else None,
            "rmse_median_year": _safe_float(ok["rmse"].median()) if len(ok) else None,
            "rmse_pooled": pm.get("rmse"),
            "mae_pooled": pm.get("mae"),
            "n_obs": int(len(pooled)) if not pooled.empty else pm.get("n_obs"),
            "n_steps_ok": int(len(ok)),
            "path": str(wf),
        })
        if len(ok) and "eval_year" in ok.columns:
            tmp = ok.copy()
            tmp["_yk"] = tmp["eval_year"].map(_year_key)
            for _, r in tmp.iterrows():
                year_map.setdefault(r["_yk"], {})[name] = _safe_float(r.get("rmse"))

    cmp = pd.DataFrame(summary_rows)
    csv_path = OUT_ROOT / "assign_sensitivity_summary.csv"
    cmp.to_csv(csv_path, index=False)

    years = [str(ev) for (_, _, ev) in WALK_WINDOWS]
    year_rows = []
    for yk in years:
        row = {"eval_year": yk}
        for name, _ in bundles:
            row[f"{name}_RMSE"] = year_map.get(yk, {}).get(name)
        year_rows.append(row)
    year_df = pd.DataFrame(year_rows)
    if len(year_df):
        mean_row = {"eval_year": "Mean"}
        med_row = {"eval_year": "Median"}
        for col in year_df.columns:
            if col == "eval_year":
                continue
            mean_row[col] = year_df[col].mean(skipna=True)
            med_row[col] = year_df[col].median(skipna=True)
        year_df = pd.concat([year_df, pd.DataFrame([mean_row, med_row])], ignore_index=True)
    year_csv = OUT_ROOT / "assign_sensitivity_by_year.csv"
    year_df.to_csv(year_csv, index=False)

    md = [
        "# R1 Sensitivity — Soft vs Hard regime assignment (rolling only)",
        "",
        "Soft: train months weighted by filtered regime probabilities (baseline).",
        "Hard: each train month assigned fully to argmax regime, then FF5 OLS per regime.",
        "OOS forecast blend still uses rolled soft probabilities. Lower RMSE is better.",
        "",
        "| Spec | RMSE mean yr | RMSE median yr | RMSE pooled | n_obs | n_steps |",
        "|------|--------------|----------------|-------------|-------|---------|",
    ]
    for _, r in cmp.iterrows():
        md.append(
            f"| {r['spec']} | {_fmt(r['rmse_mean_year'])} | {_fmt(r['rmse_median_year'])} | "
            f"{_fmt(r['rmse_pooled'])} | {r['n_obs'] if r['n_obs'] is not None else '-'} | "
            f"{r['n_steps_ok']} |"
        )
    md.extend([
        "",
        "## By evaluation year",
        "",
        "| Eval year | k2 soft | k2 hard | k4 soft | k4 hard |",
        "|-----------|---------|---------|---------|---------|",
    ])
    for _, r in year_df.iterrows():
        md.append(
            f"| {r['eval_year']} | {_fmt(r.get('k2_soft_RMSE'))} | {_fmt(r.get('k2_hard_RMSE'))} | "
            f"{_fmt(r.get('k4_soft_RMSE'))} | {_fmt(r.get('k4_hard_RMSE'))} |"
        )
    md.extend(["", f"CSV: `{csv_path.name}`, `{year_csv.name}`", ""])
    md_path = OUT_ROOT / "assign_sensitivity.md"
    md_path.write_text("\n".join(md), encoding="utf-8")
    log.info("Wrote assign sensitivity -> %s | %s | %s", md_path, csv_path, year_csv)
    return md_path


def write_window_comparison() -> Path:
    """Compare rolling vs expanding for k=2 and k=4 (sensitivity table)."""
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    bundles = [
        ("k2_rolling", WF_DIR_K2),
        ("k2_expanding", WF_DIR_EXP_K2),
        ("k4_rolling", WF_DIR_K4),
        ("k4_expanding", WF_DIR_EXP_K4),
    ]
    summary_rows = []
    year_map: dict[str, dict] = {}
    for name, wf in bundles:
        sp = wf / "walk_forward_baseline_summary.csv"
        if not sp.is_file():
            continue
        summary = pd.read_csv(sp)
        ok = _ok_steps(summary)
        pooled_path = wf / "pooled_oos_predictions.csv"
        pooled = pd.read_csv(pooled_path) if pooled_path.is_file() else pd.DataFrame()
        pm = pooled_metrics(pooled) if not pooled.empty else {}
        summary_rows.append({
            "spec": name,
            "rmse_mean_year": _safe_float(ok["rmse"].mean()) if len(ok) else None,
            "rmse_median_year": _safe_float(ok["rmse"].median()) if len(ok) else None,
            "rmse_pooled": pm.get("rmse"),
            "mae_pooled": pm.get("mae"),
            "n_obs": int(len(pooled)) if not pooled.empty else pm.get("n_obs"),
            "n_steps_ok": int(len(ok)),
            "path": str(wf),
        })
        if len(ok) and "eval_year" in ok.columns:
            tmp = ok.copy()
            tmp["_yk"] = tmp["eval_year"].map(_year_key)
            for _, r in tmp.iterrows():
                yk = r["_yk"]
                year_map.setdefault(yk, {})[name] = _safe_float(r.get("rmse"))

    cmp = pd.DataFrame(summary_rows)
    csv_path = OUT_ROOT / "window_sensitivity_summary.csv"
    cmp.to_csv(csv_path, index=False)

    years = [str(ev) for (_, _, ev) in WALK_WINDOWS]
    year_rows = []
    for yk in years:
        row = {"eval_year": yk}
        for name, _ in bundles:
            row[f"{name}_RMSE"] = year_map.get(yk, {}).get(name)
        year_rows.append(row)
    year_df = pd.DataFrame(year_rows)
    if len(year_df):
        mean_row = {"eval_year": "Mean"}
        med_row = {"eval_year": "Median"}
        for col in year_df.columns:
            if col == "eval_year":
                continue
            mean_row[col] = year_df[col].mean(skipna=True)
            med_row[col] = year_df[col].median(skipna=True)
        year_df = pd.concat([year_df, pd.DataFrame([mean_row, med_row])], ignore_index=True)
    year_csv = OUT_ROOT / "window_sensitivity_by_year.csv"
    year_df.to_csv(year_csv, index=False)

    md = [
        "# Sensitivity — Rolling 10y vs Expanding train window",
        "",
        "Rolling: train years = [T-9, T] → eval T+1 (parity with FF5/XGB/LSTM).",
        f"Expanding: train years = [{EXPAND_START_YEAR}, T] → eval T+1 (uses more history; still OOS).",
        "Lower RMSE is better. H1 primary remains rolling.",
        "",
        "| Spec | RMSE mean yr | RMSE median yr | RMSE pooled | n_obs | n_steps |",
        "|------|--------------|----------------|-------------|-------|---------|",
    ]
    for _, r in cmp.iterrows():
        md.append(
            f"| {r['spec']} | {_fmt(r['rmse_mean_year'])} | {_fmt(r['rmse_median_year'])} | "
            f"{_fmt(r['rmse_pooled'])} | {r['n_obs'] if r['n_obs'] is not None else '-'} | "
            f"{r['n_steps_ok']} |"
        )
    md.extend([
        "",
        "## By evaluation year",
        "",
        "| Eval year | k2 rolling | k2 expanding | k4 rolling | k4 expanding |",
        "|-----------|------------|--------------|------------|--------------|",
    ])
    for _, r in year_df.iterrows():
        md.append(
            f"| {r['eval_year']} | {_fmt(r.get('k2_rolling_RMSE'))} | {_fmt(r.get('k2_expanding_RMSE'))} | "
            f"{_fmt(r.get('k4_rolling_RMSE'))} | {_fmt(r.get('k4_expanding_RMSE'))} |"
        )
    md.extend([
        "",
        f"CSV: `{csv_path.name}`, `{year_csv.name}`",
        "",
    ])
    md_path = OUT_ROOT / "window_sensitivity.md"
    md_path.write_text("\n".join(md), encoding="utf-8")
    log.info("Wrote window sensitivity -> %s | %s | %s", md_path, csv_path, year_csv)
    return md_path


def write_readme(log_path: Path | None = None) -> Path:
    path = OUT_ROOT / "README_markov_walk_forward.txt"
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join([
            "Markov-switching — regime-dependent FF5 walk-forward (H1)",
            "=========================================================",
            f"Created: {datetime.now().isoformat(timespec='seconds')}",
            f"Log: {log_path or '(see markov_run_*.txt)'}",
            f"Active: k={K_REGIMES} window={WINDOW_MODE} assign={TRAIN_ASSIGN} "
            f"train_probs={TRAIN_PROB_SOURCE} market={MARKET_SERIES} "
            f"fallback={ALLOW_POOLED_FALLBACK}",
            "",
            "Primary (H1): soft + filtered + rolling + mkt_rf + pooled fallback",
            "R1: --train-assign hard",
            "R2: --train-prob-source smoothed",
            "R4: --market-series ew_excess",
            "R5: --no-pooled-fallback",
            "Expanding: --window expanding",
            "",
            "Commands:",
            "  python markov_switching.py --no-pooled-fallback --k-regimes 2",
            "  python markov_switching.py --no-pooled-fallback --k-regimes 4",
            "  python markov_switching.py --compare-fallback",
            "",
            "Outputs include:",
            "  walk_forward_nofallback_k2/ walk_forward_nofallback_k4/",
            "  fallback_sensitivity.md",
            "",
            f"All Markov artefacts root: {OUT_ROOT}",
            "",
        ]),
        encoding="utf-8",
    )
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Markov regime-FF5 walk-forward + market appendix",
    )
    parser.add_argument("--walk-forward", action="store_true", help="Run full H1 walk-forward (default)")
    parser.add_argument("--market-only", action="store_true", help="Hamilton market appendix only")
    parser.add_argument("--build-h1-table", action="store_true", help="Rebuild H1 tables from saved CSVs")
    parser.add_argument("--compare-windows", action="store_true")
    parser.add_argument("--compare-assign", action="store_true")
    parser.add_argument("--compare-probs", action="store_true")
    parser.add_argument("--compare-market-series", action="store_true", help="Rebuild R4 MKT_RF vs EW tables")
    parser.add_argument("--compare-fallback", action="store_true", help="Rebuild R5 fallback on/off tables")
    parser.add_argument("--k-regimes", type=int, default=2, choices=[2, 4])
    parser.add_argument("--window", default="rolling", choices=["rolling", "expanding"])
    parser.add_argument("--train-assign", default="soft", choices=["soft", "hard"])
    parser.add_argument("--train-prob-source", default="filtered", choices=["filtered", "smoothed"])
    parser.add_argument(
        "--market-series",
        default="mkt_rf",
        choices=["mkt_rf", "ew_excess"],
        help="mkt_rf=baseline; ew_excess=R4 equal-weight member excess for MS fit",
    )
    parser.add_argument(
        "--no-pooled-fallback",
        action="store_true",
        help="R5: disable pooled FF5 when a regime is thin/failed (step errors instead)",
    )
    args = parser.parse_args(argv)

    configure_run(
        args.k_regimes, args.window, args.train_assign,
        args.train_prob_source, args.market_series,
        allow_pooled_fallback=not args.no_pooled_fallback,
    )
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    log_path = setup_logging(OUT_ROOT)
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    log.info(
        "Configured k=%d window=%s assign=%s train_probs=%s market=%s fallback=%s | WF_DIR=%s",
        K_REGIMES, WINDOW_MODE, TRAIN_ASSIGN, TRAIN_PROB_SOURCE, MARKET_SERIES,
        ALLOW_POOLED_FALLBACK, WF_DIR,
    )

    if args.compare_fallback and not args.walk_forward and not args.market_only:
        _configure_paths()
        write_fallback_comparison()
        write_readme(log_path)
        return 0
    if args.compare_market_series and not args.walk_forward and not args.market_only:
        _configure_paths()
        write_market_series_comparison()
        write_readme(log_path)
        return 0
    if args.compare_probs and not args.walk_forward and not args.market_only:
        _configure_paths()
        write_prob_comparison()
        write_readme(log_path)
        return 0
    if args.compare_assign and not args.walk_forward and not args.market_only:
        _configure_paths()
        write_assign_comparison()
        write_readme(log_path)
        return 0
    if args.compare_windows and not args.walk_forward and not args.market_only:
        _configure_paths()
        write_window_comparison()
        write_readme(log_path)
        return 0
    if args.build_h1_table and not args.walk_forward and not args.market_only:
        _configure_paths()
        write_h1_comparison()
        write_window_comparison()
        write_assign_comparison()
        write_prob_comparison()
        write_market_series_comparison()
        write_fallback_comparison()
        write_readme(log_path)
        return 0

    _configure_paths()
    stock_level = not args.market_only
    summary, pooled, pm, _market = run_walk_forward_markov(stock_level=stock_level)
    if (
        stock_level and WINDOW_MODE == "rolling"
        and TRAIN_ASSIGN == "soft" and TRAIN_PROB_SOURCE == "filtered"
        and MARKET_SERIES == "mkt_rf" and ALLOW_POOLED_FALLBACK
    ):
        write_h1_comparison(markov_pooled_metrics=pm, markov_summary=summary)
    write_window_comparison()
    write_assign_comparison()
    write_prob_comparison()
    write_market_series_comparison()
    write_fallback_comparison()
    write_readme(log_path)
    log.info("Done. Outputs under %s (active WF: %s)", OUT_ROOT, WF_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

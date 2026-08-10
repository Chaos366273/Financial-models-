"""Manual LSTM — XGB-parity golden rule / h1_clean / ranking metrics.

Aligned with xgb_return.py:
  - Default FEATURE_SET = h1_clean with B_drop_bottom9 columns (same 19 features)
  - Members-only, sector-neutral + winsorised training target
  - Default validation: rolling walk-forward (10y train → 1y eval, step 1y)
    baseline + sentiment H2 (19 feats vs 19+sentiment) — honest OOS
  - Preferred recipe (R3a): liquidity drop 20%, seq=9 hidden=64 epochs=35 lr=1e-3
  - Mirror copy for ensemble: esemble/lstm/ (after each default WF run)
  - Legacy single-year val:  python lstm_return.py --single-val
  - Unlock TEST once:  python lstm_return.py --final-test
  - Metrics: RMSE/R2 + Spearman IC + long-short + top-K / sector books

Output routing (same golden-rule pattern as xgb_return.py):
  - Validation / walk-forward / tune → saved_data/ + logs/
  - --final-test (golden rule unlocked once) → final test results/LSTM/

Bloomberg (default): data/final data for AI/07_panel/master_panel.parquet
Yahoo (legacy): set DATA_SOURCE = "yahoo" (no golden-rule val split).
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

ROOT = Path(__file__).resolve().parent
DISS_MODELS = ROOT.parent / "dissertation-models"
FINAL_DATA = ROOT.parent.parent / "data" / "final data for AI"
MASTER_PANEL = FINAL_DATA / "07_panel" / "master_panel.parquet"
SAVE_DIR = ROOT / "saved_data"
LOG_DIR = ROOT / "logs"
# One-shot final test only — activated by --final-test
FINAL_TEST_DIR = ROOT / "final test results" / "LSTM"
RESULTS_TRACKER = SAVE_DIR / "lstm_results_tracker.csv"
TEST_LOCK_FILE = SAVE_DIR / "GOLDEN_RULE_lstm_test_already_scored.txt"
LEGACY_TEST_LOCK_FILE = SAVE_DIR / "GOLDEN_RULE_lstm_test_already_scored.txt"
# Validation walk-forward lives under saved_data (NOT final test results)
WF_OUT_ROOT = SAVE_DIR / "lstm_walk_forward"
WALK_FORWARD_DIR = WF_OUT_ROOT / "walk_forward_baseline"
ROBUSTNESS_DIR = SAVE_DIR / "lstm_robustness"
ROBUSTNESS_LEDGER = ROBUSTNESS_DIR / "robustness_ledger.csv"
# Mirror copy for ensemble stacking (does not replace saved_data/ primary outputs)
ENSEMBLE_LSTM_COPY_DIR = ROOT / "esemble" / "lstm"

# H2: run baseline + sentiment arms on default walk-forward (use --no-sentiment to disable)
H2_INCLUDE_SENTIMENT = True

# Robustness ladder toggles (CLI overrides; defaults = preferred R3a recipe)
REQUIRE_CONTIGUOUS = False
WARM_START = True
LIQUIDITY_DROP = 0.20  # R3a: drop bottom 20% within each month (eqy_turnover)
DROPOUT = 0.0
OUT_TAG = ""  # if set, WF writes under saved_data/lstm_robustness/<OUT_TAG>/

# Pre-R3a honest WF reference (2026-08-02) — robustness ledger deltas only
FROZEN_BASELINE_AVG = {
    "ic": 0.0069,
    "long_short": 0.0034,
    "topk_edge": -0.0017,
    "hit_rate": 0.49,
}

# Adopted preferred recipe R3a_liq20 (honest WF, 2026-08-03)
PREFERRED_RECIPE_AVG = {
    "ic": 0.0106,
    "long_short": 0.0050,
    "topk_edge": 0.0019,
    "hit_rate": 0.51,
}

SENTIMENT_CANDIDATES = [
    "news_sentiment",
    "news_publication_count",
    "twitter_sentiment",
    "twitter_publication_count",
    "sentiment_proxy",
]

# Rolling walk-forward: 10y train, 1y eval, step 1y (eval years 2012–2025)
WF_WINDOWS = [
    # step, train_start_year, train_end_year, eval_year, note
    (1, 2002, 2011, 2012, "Post-GFC recovery"),
    (2, 2003, 2012, 2013, ""),
    (3, 2004, 2013, 2014, ""),
    (4, 2005, 2014, 2015, ""),
    (5, 2006, 2015, 2016, "Brexit referendum year"),
    (6, 2007, 2016, 2017, ""),
    (7, 2008, 2017, 2018, ""),
    (8, 2009, 2018, 2019, ""),
    (9, 2010, 2019, 2020, "COVID crash"),
    (10, 2011, 2020, 2021, "COVID recovery"),
    (11, 2012, 2021, 2022, "Inflation / rate-rise year"),
    (12, 2013, 2022, 2023, "Previous validation year"),
    (13, 2014, 2023, 2024, "Recent period"),
    (14, 2015, 2024, 2025, "Most recent (partial year if needed)"),
]

# Expanding window: train from EXPAND_START_YEAR through eval_year - 1 (same eval years as rolling)
EXPAND_START_YEAR = 2002
WINDOW_MODE = "rolling"  # rolling | expanding

# Rolling R3a reference for --compare-windows
R3A_ROLLING_SUMMARY = (
    ROBUSTNESS_DIR / "R3a_liq20" / "walk_forward_baseline" / "summary_metrics.csv"
)

log = logging.getLogger("lstm_return")

DATA_SOURCE = "bloomberg"  # "bloomberg" | "yahoo"

TICKERS: str | list[str] | None = None

# Match XGB primary frozen ML spec (B_drop_bottom9)
FEATURE_SET = "h1_clean"
LOCKED_FEATURE_COLS = [
    "log_market_cap", "sma_20", "SMB", "MOM", "bb_lower", "debt_equity", "CMA",
    "sma_15", "HML", "macd_signal", "volatility_30d", "price_to_sma_20", "atr",
    "volume_trend", "RMW", "bb_middle", "eqy_turnover", "revenue_growth", "macd",
]
EXPERIMENT_LABEL = "B_drop_bottom9"

FILTER_MEMBERS = True
DEMEAN_WITHIN_MONTH = True
SECTOR_NEUTRAL_TARGET = True
WINSORISE_Y = True
WINSOR_LIMITS = (0.01, 0.99)
EXCLUDE_SECTORS = {"Unknown", ""}

TOP_K = 50
TOP_FRAC = 0.10

# Golden rule default: False = report validation only
EVALUATE_TEST = False

# Architecture frozen after seq-length experiment (2026-08-02, seed=42); R3a adds liq filter
# Single-val 2023 ref: seq=9 IC 0.0447; honest WF R3a avg IC 0.0106
SEQ_LEN = 9
HIDDEN_SIZE = 64
EPOCHS = 35
BATCH_SIZE = 64
PATIENCE = 5
LR = 1e-3
SEED = 42  # torch + numpy — required for reproducible Exp grids

RUN_STABILITY = True
STABILITY_ROLLING_MONTHS = 6

# Experiment 1: sequence-length sensitivity (hidden/epochs frozen at champion)
TUNE_GRID = [
    {"seq_len": 9, "hidden_size": 64, "epochs": 35},
    {"seq_len": 12, "hidden_size": 64, "epochs": 35},
    {"seq_len": 15, "hidden_size": 64, "epochs": 35},
    {"seq_len": 18, "hidden_size": 64, "epochs": 35},
]

# Yahoo legacy
YAHOO_TRAIN_END = "2020-12-31"
YAHOO_TEST_START = "2021-01-01"
YAHOO_TEST_END = "2025-12-31"
INCLUDE_ALT_DATA = True


def _activate_final_test_output_dir() -> None:
    """Route all outputs to final test results/LSTM (only when --final-test)."""
    global SAVE_DIR, LOG_DIR, RESULTS_TRACKER, TEST_LOCK_FILE, WF_OUT_ROOT, WALK_FORWARD_DIR
    FINAL_TEST_DIR.mkdir(parents=True, exist_ok=True)
    SAVE_DIR = FINAL_TEST_DIR
    LOG_DIR = FINAL_TEST_DIR
    RESULTS_TRACKER = SAVE_DIR / "lstm_results_tracker.csv"
    TEST_LOCK_FILE = SAVE_DIR / "GOLDEN_RULE_lstm_test_already_scored.txt"
    WF_OUT_ROOT = FINAL_TEST_DIR
    WALK_FORWARD_DIR = WF_OUT_ROOT / "walk_forward_baseline"


def _activate_out_tag(tag: str) -> None:
    """Route walk-forward outputs under saved_data/lstm_robustness/<tag>/."""
    global WF_OUT_ROOT, WALK_FORWARD_DIR, OUT_TAG
    OUT_TAG = str(tag).strip()
    if not OUT_TAG:
        return
    WF_OUT_ROOT = ROBUSTNESS_DIR / OUT_TAG
    WALK_FORWARD_DIR = WF_OUT_ROOT / "walk_forward_baseline"
    WF_OUT_ROOT.mkdir(parents=True, exist_ok=True)


def _apply_liquidity_filter(df: pd.DataFrame, drop_q: float) -> pd.DataFrame:
    """Drop bottom drop_q fraction within each month by turnover (fallback mcap)."""
    if drop_q is None or float(drop_q) <= 0 or df is None or df.empty:
        return df
    col = "eqy_turnover" if "eqy_turnover" in df.columns else (
        "log_market_cap" if "log_market_cap" in df.columns else None
    )
    if col is None:
        log.warning("Liquidity filter skipped — no eqy_turnover/log_market_cap column")
        return df
    q = float(drop_q)

    def _keep(g: pd.DataFrame) -> pd.DataFrame:
        s = pd.to_numeric(g[col], errors="coerce")
        if s.notna().sum() < 5:
            return g
        thr = s.quantile(q)
        return g.loc[s >= thr]

    out = df.groupby("month", group_keys=False).apply(_keep)
    log.info(
        "Liquidity filter drop_q=%.2f via %s | rows %d -> %d",
        q, col, len(df), len(out),
    )
    return out


def _append_robustness_ledger(
    experiment_id: str,
    result: dict,
    decision: str = "pending",
    notes: str = "",
    pct_seq_dropped_gaps: float | None = None,
) -> None:
    """Append one comparison row vs frozen baseline AVERAGE."""
    ROBUSTNESS_DIR.mkdir(parents=True, exist_ok=True)
    ts = result.get("topk_summary") or {}
    mean_ic = result.get("ic_mean")
    mean_ls = result.get("long_short_mean")
    mean_edge = ts.get("mean_edge")
    mean_hit = ts.get("hit_rate")
    avg = result.get("average") or {}
    row = {
        "experiment_id": experiment_id,
        "run_id": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "n_ok_steps": result.get("n_ok"),
        "n_steps": result.get("n_steps"),
        "mean_ic": mean_ic,
        "mean_ls": mean_ls,
        "mean_topk_edge": mean_edge,
        "mean_hit": mean_hit,
        "mean_n_stocks": avg.get("n_stocks"),
        "mean_n_eval_rows": avg.get("n_eval_rows"),
        "pct_seq_dropped_gaps": pct_seq_dropped_gaps,
        "delta_ic": (None if mean_ic is None else mean_ic - FROZEN_BASELINE_AVG["ic"]),
        "delta_ls": (None if mean_ls is None else mean_ls - FROZEN_BASELINE_AVG["long_short"]),
        "delta_topk_edge": (
            None if mean_edge is None else mean_edge - FROZEN_BASELINE_AVG["topk_edge"]
        ),
        "delta_hit": (None if mean_hit is None else mean_hit - FROZEN_BASELINE_AVG["hit_rate"]),
        "require_contiguous": REQUIRE_CONTIGUOUS,
        "warm_start": WARM_START,
        "liquidity_drop": LIQUIDITY_DROP,
        "sector_neutral": SECTOR_NEUTRAL_TARGET,
        "winsor": (list(WINSOR_LIMITS) if WINSORISE_Y else "none"),
        "filter_members": FILTER_MEMBERS,
        "hidden": HIDDEN_SIZE,
        "dropout": DROPOUT,
        "lr": LR,
        "patience": PATIENCE,
        "decision": decision,
        "notes": notes,
        "out_dir": result.get("out_dir") or str(WF_OUT_ROOT),
    }
    write_header = not ROBUSTNESS_LEDGER.is_file()
    with ROBUSTNESS_LEDGER.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    log.info("Appended robustness ledger -> %s", ROBUSTNESS_LEDGER)


def setup_logging() -> Path:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"lstm_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    logger = logging.getLogger("lstm_return")
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


def _configure_paths() -> None:
    if not MASTER_PANEL.is_file():
        raise FileNotFoundError(f"Master panel not found: {MASTER_PANEL}")
    os.environ["DISSLAB_MASTER_PANEL"] = str(MASTER_PANEL.resolve())
    os.environ["DISSLAB_DATASET_LABEL"] = "final_data_master_panel"
    diss = str(DISS_MODELS.resolve())
    if diss not in sys.path:
        sys.path.insert(0, diss)
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))


def _existing_test_lock_path() -> Path | None:
    for path in (
        TEST_LOCK_FILE,
        FINAL_TEST_DIR / "GOLDEN_RULE_lstm_test_already_scored.txt",
        LEGACY_TEST_LOCK_FILE,
    ):
        if path.is_file():
            return path
    return None


def _assert_test_unlock_allowed(force_retest: bool) -> None:
    lock_path = _existing_test_lock_path()
    if lock_path is None:
        return
    if force_retest:
        log.warning(
            "GOLDEN RULE OVERRIDE: --force-retest used. LSTM test was already scored.\n%s",
            lock_path.read_text(encoding="utf-8")[:800],
        )
        return
    raise RuntimeError(
        "GOLDEN RULE: LSTM test set was already scored once.\n"
        f"Lock file: {lock_path}\n"
        "Do not re-check 2024–2025 after tuning. "
        "If you truly need a re-run (bugfix only), pass --force-retest."
    )


def _write_test_lock(result: dict, log_path: Path | None) -> None:
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    lines = [
        f"scored_at={datetime.now().isoformat(timespec='seconds')}",
        f"feature_set={result.get('feature_set')}",
        f"experiment={result.get('experiment_label')}",
        f"output_dir={SAVE_DIR}",
        f"test_start={result.get('test_start')} test_end={result.get('test_end')}",
        f"rmse={result.get('rmse')} ic_mean={result.get('ic_mean')} "
        f"long_short_mean={result.get('long_short_mean')}",
        f"topk_edge={(result.get('topk_summary') or {}).get('mean_edge')}",
        f"best_params={json.dumps(result.get('best_params') or {}, default=str)}",
        f"log_file={log_path or ''}",
        "",
        "This file means the LSTM dissertation TEST window was unlocked. "
        "Do not tune further against these numbers.",
    ]
    text = "\n".join(lines)
    TEST_LOCK_FILE.write_text(text, encoding="utf-8")
    log.info("Wrote golden-rule lock: %s", TEST_LOCK_FILE)
    LEGACY_TEST_LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
    LEGACY_TEST_LOCK_FILE.write_text(
        text + f"\ncanonical_output_dir={FINAL_TEST_DIR}\n",
        encoding="utf-8",
    )
    readme = SAVE_DIR / "README_final_test_outputs.txt"
    readme.write_text(
        "\n".join([
            "LSTM FINAL TEST outputs (dissertation)",
            "=====================================",
            f"Folder: {SAVE_DIR}",
            "",
            "Written only when --final-test unlocks the golden rule.",
            "Validation / walk-forward runs stay in saved_data/ and logs/.",
            "",
        ]),
        encoding="utf-8",
    )
    log.info("Wrote output folder guide: %s", readme)


def _append_results_tracker(result: dict, notes: str = "") -> None:
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    bp = result.get("best_params") or {}
    ts = result.get("topk_summary") or {}
    on_test = result.get("evaluated_on") == "test"
    row = {
        "run_id": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "date": datetime.now().strftime("%d/%m/%Y"),
        "model": "LSTM",
        "feature_set": result.get("feature_set"),
        "experiment": result.get("experiment_label"),
        "seq_len": bp.get("seq_len", result.get("seq_len")),
        "hidden": bp.get("hidden_size", result.get("hidden_size")),
        "epochs": bp.get("epochs", result.get("epochs")),
        "val_rmse": result.get("val_rmse"),
        "val_ic": result.get("val_ic_mean"),
        "val_ls": result.get("val_long_short_mean"),
        "val_topk_edge": result.get("val_topk_edge"),
        "test_rmse": result.get("rmse") if on_test else "",
        "test_ic": result.get("ic_mean") if on_test else "",
        "test_ls": result.get("long_short_mean") if on_test else "",
        "test_topk_edge": ts.get("mean_edge") if on_test else "",
        "evaluated_on": result.get("evaluated_on"),
        "notes": notes,
    }
    write_header = not RESULTS_TRACKER.is_file()
    with RESULTS_TRACKER.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    log.info("Appended results tracker row -> %s", RESULTS_TRACKER)


def _xgb_helpers():
    """Reuse prep/ranking helpers from xgb_return (keep LSTM + XGB aligned)."""
    import xgb_return as xgb

    # Mirror LSTM flags into xgb helpers for this process
    xgb.FILTER_MEMBERS = FILTER_MEMBERS
    xgb.DEMEAN_WITHIN_MONTH = DEMEAN_WITHIN_MONTH
    xgb.SECTOR_NEUTRAL_TARGET = SECTOR_NEUTRAL_TARGET
    xgb.WINSORISE_Y = WINSORISE_Y
    xgb.WINSOR_LIMITS = WINSOR_LIMITS
    xgb.TOP_K = TOP_K
    xgb.TOP_FRAC = TOP_FRAC
    xgb.EXCLUDE_SECTORS = EXCLUDE_SECTORS
    return xgb


def _normalize_tickers(tickers: str | list[str] | None) -> list[str] | None:
    return _xgb_helpers()._normalize_tickers(tickers)


def _filter_members(df: pd.DataFrame) -> pd.DataFrame:
    return _xgb_helpers()._filter_members(df)


def _sector_series(df: pd.DataFrame) -> pd.Series:
    return _xgb_helpers()._sector_series(df)


def _add_model_target(train: pd.DataFrame, *others: pd.DataFrame) -> tuple[pd.DataFrame, ...]:
    return _xgb_helpers()._add_model_target(train, *others)


def _pct(x) -> str:
    return _xgb_helpers()._pct(x)


def _ic_plain_english(ic) -> str:
    return _xgb_helpers()._ic_plain_english(ic)


def _ls_plain_english(ls) -> str:
    return _xgb_helpers()._ls_plain_english(ls)


def _rank_score(frame: pd.DataFrame) -> float:
    return _xgb_helpers()._rank_score(frame)


def build_topk_book(preview: pd.DataFrame, k: int = TOP_K) -> tuple[pd.DataFrame, dict]:
    return _xgb_helpers().build_topk_book(preview, k=k)


def build_sector_diversified_book(preview: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    return _xgb_helpers().build_sector_diversified_book(preview)


def _monthly_ic_ls(preview: pd.DataFrame, min_names: int = 10) -> pd.DataFrame:
    return _xgb_helpers()._monthly_ic_ls(preview, min_names=min_names)


def compute_stability_diagnostics(
    preview: pd.DataFrame,
    rolling: int = STABILITY_ROLLING_MONTHS,
) -> dict:
    """Validation-only stability: rolling IC / positive-month rates (no tree importances)."""
    monthly = _monthly_ic_ls(preview)
    if monthly.empty:
        return {"ok": False, "reason": "no monthly IC rows"}

    monthly = monthly.sort_values("month").reset_index(drop=True)
    monthly["rolling_ic"] = monthly["ic"].rolling(rolling, min_periods=rolling).mean()
    monthly["rolling_ls"] = monthly["long_short"].rolling(rolling, min_periods=rolling).mean()
    n = len(monthly)
    rolling_vals = monthly["rolling_ic"].dropna()
    return {
        "ok": True,
        "n_months": n,
        "rolling_window": rolling,
        "ic_mean": float(monthly["ic"].mean()),
        "ic_std": float(monthly["ic"].std(ddof=1)) if n > 1 else 0.0,
        "frac_months_ic_positive": float((monthly["ic"] > 0).mean()) if n else None,
        "frac_months_ls_positive": float((monthly["long_short"] > 0).mean()) if n else None,
        "frac_months_topk_edge_positive": float((monthly["topk_edge"] > 0).mean()) if n else None,
        "rolling_ic_last": float(rolling_vals.iloc[-1]) if len(rolling_vals) else None,
        "frac_rolling_ic_positive": float((rolling_vals > 0).mean()) if len(rolling_vals) else None,
        "monthly": monthly,
    }


def print_stability_report(stab: dict, feature_set: str) -> None:
    if not stab or not stab.get("ok"):
        log.info("Stability diagnostics unavailable: %s", (stab or {}).get("reason", "n/a"))
        return
    log.info("")
    log.info("=" * 72)
    log.info("LSTM STABILITY | %s | validation only", feature_set)
    log.info("=" * 72)
    log.info("Monthly IC mean=%.4f  std=%.4f  months=%d", stab["ic_mean"], stab["ic_std"], stab["n_months"])
    log.info(
        "Frac months IC>0     : %.1f%%",
        100.0 * stab["frac_months_ic_positive"] if stab["frac_months_ic_positive"] is not None else float("nan"),
    )
    log.info(
        "Frac months LS>0     : %.1f%%",
        100.0 * stab["frac_months_ls_positive"] if stab["frac_months_ls_positive"] is not None else float("nan"),
    )
    log.info(
        "Frac months TopK>mkt : %.1f%%",
        100.0 * stab["frac_months_topk_edge_positive"]
        if stab["frac_months_topk_edge_positive"] is not None else float("nan"),
    )
    rw = stab["rolling_window"]
    log.info(
        "Rolling %dm IC (last window)=%s | frac rolling IC>0=%s",
        rw,
        ("%.4f" % stab["rolling_ic_last"]) if stab["rolling_ic_last"] is not None else "n/a",
        ("%.1f%%" % (100.0 * stab["frac_rolling_ic_positive"]))
        if stab["frac_rolling_ic_positive"] is not None else "n/a",
    )
    log.info("")
    log.info("Month-by-month IC / LS / TopK-edge:")
    log.info("%-12s %8s %10s %10s", "month", "IC", "LS/mo", "TopK-edge")
    for _, row in stab["monthly"].iterrows():
        log.info(
            "%-12s %8.4f %10s %10s",
            row["month"],
            row["ic"],
            _pct(row["long_short"]),
            _pct(row["topk_edge"]),
        )
    ok_ic = (stab["frac_months_ic_positive"] or 0) >= 0.60
    ok_roll = stab["rolling_ic_last"] is not None and stab["rolling_ic_last"] > 0
    log.info("")
    log.info("Stability gate (rules of thumb for val-only):")
    if ok_ic and ok_roll:
        log.info("  PASS: edge is positive in most months; rolling IC>0.")
    else:
        log.info("  CAUTION: check weak months / rolling IC before unlocking test.")


def _preview_from_keys(keys, pred, panel: pd.DataFrame) -> pd.DataFrame:
    """Map sequence keys + predictions to raw excess returns and Sector for ranking metrics."""
    tickers_out, months_out = zip(*keys) if keys else ([], [])
    key_df = pd.DataFrame({
        "Ticker": list(tickers_out),
        "month": list(months_out),
        "y_pred": np.asarray(pred, dtype=float),
    })
    meta = panel[["Ticker", "month", "target_excess_return"]].copy()
    meta["Ticker"] = meta["Ticker"].astype(str)
    meta["month"] = pd.to_datetime(meta["month"]).dt.strftime("%Y-%m-%d")
    meta["Sector"] = _sector_series(panel).values
    meta = meta.rename(columns={"target_excess_return": "y_true"})
    out = key_df.merge(meta, on=["Ticker", "month"], how="left")
    out["error"] = out["y_true"] - out["y_pred"]
    return out.sort_values(["Ticker", "month"]).reset_index(drop=True)


def _metrics_from_preview(preview: pd.DataFrame) -> dict:
    from evaluation.predictive import ranking_metrics, predictive_metrics

    y_true = preview["y_true"].astype(float).values
    y_pred = preview["y_pred"].astype(float).values
    pm = predictive_metrics(y_true, y_pred)
    rank = ranking_metrics(preview)
    topk_picks, topk_summary = build_topk_book(preview, k=TOP_K)
    sector_picks, sector_summary = build_sector_diversified_book(preview)
    return {
        "mse": pm["mse"], "mae": pm["mae"], "rmse": pm["rmse"], "r2": pm["r2"],
        "ic_mean": rank["ic_mean"], "ic_std": rank["ic_std"], "ic_ir": rank["ic_ir"],
        "long_short_mean": rank["long_short_mean"], "long_short_std": rank["long_short_std"],
        "rank_n_months": rank["n_months"],
        "preview": preview,
        "topk_picks": topk_picks,
        "topk_summary": topk_summary,
        "sector_picks": sector_picks,
        "sector_summary": sector_summary,
    }


# In-process cache: rebuild sequences only when the window / flags change
# (helps tune grids / P4 hyperparam pilots on the same train fold).
_SEQ_CACHE: dict[tuple, tuple] = {}
_SEQ_CACHE_STATS = {"hits": 0, "misses": 0}


def clear_seq_cache() -> None:
    _SEQ_CACHE.clear()
    _SEQ_CACHE_STATS["hits"] = 0
    _SEQ_CACHE_STATS["misses"] = 0


def _panel_fingerprint(df: pd.DataFrame | None) -> tuple:
    if df is None or len(df) == 0:
        return ("empty",)
    months = pd.to_datetime(df["month"])
    return (
        int(months.min().value),
        int(months.max().value),
        int(len(df)),
        int(df["Ticker"].nunique()) if "Ticker" in df.columns else 0,
    )


def _build_sequences_cached(
    df: pd.DataFrame,
    feature_set: str,
    seq_len: int,
    target_col: str = "target_model",
    feature_cols: list[str] | None = None,
    history: pd.DataFrame | None = None,
    require_contiguous: bool = False,
):
    """Cache wrapper around models.lstm_model._build_sequences."""
    from models.lstm_model import _build_sequences

    key = (
        _panel_fingerprint(df),
        _panel_fingerprint(history),
        feature_set,
        int(seq_len),
        target_col,
        tuple(feature_cols or ()),
        bool(require_contiguous),
    )
    hit = _SEQ_CACHE.get(key)
    if hit is not None:
        _SEQ_CACHE_STATS["hits"] += 1
        return hit
    _SEQ_CACHE_STATS["misses"] += 1
    out = _build_sequences(
        df,
        feature_set,
        seq_len,
        target_col=target_col,
        feature_cols=feature_cols,
        history=history,
        require_contiguous=require_contiguous,
    )
    _SEQ_CACHE[key] = out
    return out


def _fit_one(
    params: dict,
    train: pd.DataFrame,
    val: pd.DataFrame,
    feature_set: str,
    feature_cols: list[str] | None,
    early_stop_on_eval: bool = True,
):
    """Fit LSTM on train; predict on val (optionally warm-started from train history).

    early_stop_on_eval=True  — OK for tune / single-val (monitor the val fold).
    early_stop_on_eval=False — required for walk-forward OOS (do not peek at eval year).
    """
    from models.lstm_model import fit_predict_lstm

    seq_len = int(params["seq_len"])
    hidden = int(params["hidden_size"])
    epochs = int(params["epochs"])
    hist = train if WARM_START else None

    # Train: sequences inside train only (cached per window)
    X_tr, y_tr, cols, _, stats_tr = _build_sequences_cached(
        train,
        feature_set,
        seq_len,
        target_col="target_model",
        feature_cols=feature_cols,
        require_contiguous=REQUIRE_CONTIGUOUS,
    )
    # Val/eval: optional warm-start with train history
    X_va, y_va, _, keys_va, stats_va = _build_sequences_cached(
        val,
        feature_set,
        seq_len,
        target_col="target_model",
        feature_cols=feature_cols,
        history=hist,
        require_contiguous=REQUIRE_CONTIGUOUS,
    )
    if X_tr is None or X_va is None or len(y_va) == 0:
        return None, None, None, cols

    if early_stop_on_eval:
        pred_va = fit_predict_lstm(
            X_tr, y_tr, X_va, hidden, epochs, BATCH_SIZE,
            X_va=X_va, y_va=y_va, patience=PATIENCE, lr=LR, seed=SEED,
            dropout=DROPOUT,
        )
    else:
        # Honest OOS: fixed epochs, no eval-year early stopping
        pred_va = fit_predict_lstm(
            X_tr, y_tr, X_va, hidden, epochs, BATCH_SIZE,
            lr=LR, seed=SEED, dropout=DROPOUT,
        )
    frame = _preview_from_keys(keys_va, pred_va, val)
    pack = {
        "X_tr": X_tr,
        "y_tr": y_tr,
        "cols": cols,
        "seq_len": seq_len,
        "stats_train": stats_tr,
        "stats_eval": stats_va,
    }
    return params, frame, pack, cols


def run_lstm_forecast(
    tickers: str | list[str] | None = TICKERS,
    feature_set: str = FEATURE_SET,
    evaluate_test: bool = False,
    feature_cols: list[str] | None = None,
    experiment_label: str | None = None,
    run_stability: bool | None = None,
    tune: bool = False,
) -> dict:
    _configure_paths()
    warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)

    from data.build_panel import build_panel
    from evaluation.predictive import ranking_metrics, predictive_metrics
    from models.lstm_model import fit_predict_lstm

    do_stability = RUN_STABILITY if run_stability is None else bool(run_stability)
    cols_override = list(feature_cols) if feature_cols is not None else list(LOCKED_FEATURE_COLS)
    label = experiment_label or EXPERIMENT_LABEL
    clear_seq_cache()

    selected = _normalize_tickers(tickers)
    _, splits = build_panel()
    train = _filter_members(splits["train"].copy())
    val = _filter_members(splits["val"].copy())
    test = _filter_members(splits["test"].copy())

    if selected:
        train = train[train["Ticker"].isin(selected)]
        val = val[val["Ticker"].isin(selected)]
        test = test[test["Ticker"].isin(selected)]
        if train.empty or val.empty:
            raise ValueError(f"No train/val rows for tickers {selected} after filters.")
        if evaluate_test and test.empty:
            raise ValueError(f"No test rows for tickers {selected} after filters.")

    if LIQUIDITY_DROP and LIQUIDITY_DROP > 0:
        train = _apply_liquidity_filter(train, LIQUIDITY_DROP)
        val = _apply_liquidity_filter(val, LIQUIDITY_DROP)
        test = _apply_liquidity_filter(test, LIQUIDITY_DROP)

    train, val, test = _add_model_target(train, val, test)

    grid = list(TUNE_GRID) if tune else [{
        "seq_len": SEQ_LEN,
        "hidden_size": HIDDEN_SIZE,
        "epochs": EPOCHS,
    }]
    mode = "FINAL TEST (unlocked once)" if evaluate_test else "TUNE (validation only - test locked)"
    log.info("")
    log.info(
        "--- LSTM %s | base=%s (%d cols) | mode=%s | grid=%d ---",
        label, feature_set, len(cols_override), mode, len(grid),
    )
    log.info(
        "Warm-start=%s | contiguous=%s | liquidity_drop=%s | dropout=%s",
        WARM_START, REQUIRE_CONTIGUOUS, LIQUIDITY_DROP, DROPOUT,
    )
    log.info("RNG seed=%s (torch + numpy)", SEED)

    best_score = -1e18
    best_params: dict = {}
    best_va_frame = None
    best_pack = None
    best_cols: list[str] = []
    grid_rows: list[dict] = []

    for params in grid:
        used, va_frame, pack, cols = _fit_one(params, train, val, feature_set, cols_override)
        if va_frame is None or pack is None:
            log.warning("  skip seq=%s hidden=%s — not enough sequences", params["seq_len"], params["hidden_size"])
            continue
        score = _rank_score(va_frame)
        rm = ranking_metrics(va_frame)
        _, topk_sum = build_topk_book(va_frame, k=TOP_K)
        stab = compute_stability_diagnostics(va_frame, rolling=STABILITY_ROLLING_MONTHS)
        topk_edge = topk_sum.get("mean_edge")
        frac_ic = stab.get("frac_months_ic_positive") if stab.get("ok") else None
        roll_ic = stab.get("rolling_ic_last") if stab.get("ok") else None
        log.info(
            "  seq=%s hidden=%s epochs=%s val_IC=%s val_LS=%s topk_edge=%s "
            "frac_IC>0=%s roll6_IC=%s score=%.4f",
            used["seq_len"],
            used["hidden_size"],
            used["epochs"],
            ("%.4f" % rm["ic_mean"]) if rm["ic_mean"] is not None else "n/a",
            ("%.4f" % rm["long_short_mean"]) if rm["long_short_mean"] is not None else "n/a",
            _pct(topk_edge),
            ("%.0f%%" % (100.0 * frac_ic)) if frac_ic is not None else "n/a",
            ("%.4f" % roll_ic) if roll_ic is not None else "n/a",
            score,
        )
        grid_rows.append({
            "seq_len": used["seq_len"],
            "hidden_size": used["hidden_size"],
            "epochs": used["epochs"],
            "val_ic": rm["ic_mean"],
            "val_ls": rm["long_short_mean"],
            "topk_edge": topk_edge,
            "topk_hit_rate": topk_sum.get("hit_rate"),
            "frac_months_ic_positive": frac_ic,
            "frac_months_ls_positive": stab.get("frac_months_ls_positive") if stab.get("ok") else None,
            "rolling_ic_last": roll_ic,
            "score": score,
            "n_val_months": stab.get("n_months") if stab.get("ok") else None,
        })
        if score > best_score:
            best_score = score
            best_params = dict(used)
            best_va_frame = va_frame
            best_pack = pack
            best_cols = cols

    if grid_rows:
        SAVE_DIR.mkdir(parents=True, exist_ok=True)
        grid_path = SAVE_DIR / f"lstm_tune_grid_{label.replace(' ', '_')}_val.csv"
        pd.DataFrame(grid_rows).to_csv(grid_path, index=False)
        log.info("Saved per-config tune grid -> %s", grid_path)
        log.info("")
        log.info("%-8s %8s %10s %10s %10s %10s", "seq", "IC", "LS/mo", "TopK-edge", "IC>0 mo", "roll6 IC")
        for row in grid_rows:
            log.info(
                "%-8s %8.4f %10s %10s %10s %10s",
                row["seq_len"],
                row["val_ic"] if row["val_ic"] is not None else float("nan"),
                _pct(row["val_ls"]),
                _pct(row["topk_edge"]),
                ("%.0f%%" % (100.0 * row["frac_months_ic_positive"]))
                if row["frac_months_ic_positive"] is not None else "n/a",
                ("%.4f" % row["rolling_ic_last"]) if row["rolling_ic_last"] is not None else "n/a",
            )

    if best_va_frame is None or best_pack is None:
        raise ValueError(
            f"Not enough sequences (need > {SEQ_LEN} months per ticker after filters). "
            "Try reducing SEQ_LEN or relaxing filters."
        )

    vp = best_va_frame.copy()
    vp["month"] = pd.to_datetime(vp["month"]).dt.strftime("%Y-%m-%d")
    val_pm = predictive_metrics(vp["y_true"].values, vp["y_pred"].values)
    val_rank = ranking_metrics(vp)
    val_topk_picks, val_topk_summary = build_topk_book(vp, k=TOP_K)
    val_sector_picks, val_sector_summary = build_sector_diversified_book(vp)

    result = {
        "feature_set": feature_set,
        "experiment_label": label,
        "n_features": len(best_cols),
        "feature_columns": best_cols,
        "best_params": best_params,
        "seq_len": best_params["seq_len"],
        "hidden_size": best_params["hidden_size"],
        "epochs": best_params["epochs"],
        "batch_size": BATCH_SIZE,
        "train_rows": int(len(best_pack["y_tr"])),
        "val_rows": int(len(vp)),
        "test_rows": 0,
        "train_start": pd.to_datetime(train["month"]).min().date().isoformat(),
        "train_end": pd.to_datetime(train["month"]).max().date().isoformat(),
        "val_start": pd.to_datetime(val["month"]).min().date().isoformat(),
        "val_end": pd.to_datetime(val["month"]).max().date().isoformat(),
        "test_start": pd.to_datetime(test["month"]).min().date().isoformat(),
        "test_end": pd.to_datetime(test["month"]).max().date().isoformat(),
        "evaluated_on": "validation",
        "data_source": "bloomberg",
        "features_csv": str(MASTER_PANEL),
        "val_rmse": val_pm["rmse"],
        "val_mae": val_pm["mae"],
        "val_r2": val_pm["r2"],
        "val_ic_mean": val_rank["ic_mean"],
        "val_ic_std": val_rank["ic_std"],
        "val_long_short_mean": val_rank["long_short_mean"],
        "val_topk_edge": val_topk_summary.get("mean_edge"),
        "mse": val_pm["mse"],
        "mae": val_pm["mae"],
        "rmse": val_pm["rmse"],
        "r2": val_pm["r2"],
        "ic_mean": val_rank["ic_mean"],
        "ic_std": val_rank["ic_std"],
        "ic_ir": val_rank["ic_ir"],
        "long_short_mean": val_rank["long_short_mean"],
        "long_short_std": val_rank["long_short_std"],
        "rank_n_months": val_rank["n_months"],
        "preview": vp,
        "topk_picks": val_topk_picks,
        "topk_summary": val_topk_summary,
        "sector_picks": val_sector_picks,
        "sector_summary": val_sector_summary,
        "topk_path": None,
        "sector_path": None,
    }

    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    suffix_tag = "test" if evaluate_test else "val"
    file_tag = label.replace(" ", "_")
    topk_path = SAVE_DIR / f"lstm_topk_picks_{file_tag}_{suffix_tag}.csv"
    sector_path = SAVE_DIR / f"lstm_sector_picks_{file_tag}_{suffix_tag}.csv"

    if not evaluate_test:
        log.info(
            "GOLDEN RULE: test set locked. Reporting VALIDATION only "
            "(%s -> %s). Re-run with --final-test when tuning is finished.",
            result["test_start"],
            result["test_end"],
        )
        if len(val_topk_picks):
            val_topk_picks.to_csv(topk_path, index=False)
            result["topk_path"] = str(topk_path)
        if len(val_sector_picks):
            val_sector_picks.to_csv(sector_path, index=False)
            result["sector_path"] = str(sector_path)

        if do_stability:
            stab = compute_stability_diagnostics(vp, rolling=STABILITY_ROLLING_MONTHS)
            result["stability"] = stab
            print_stability_report(stab, feature_set)
            if stab.get("ok") and stab.get("monthly") is not None:
                stab_path = SAVE_DIR / f"lstm_stability_monthly_{file_tag}_val.csv"
                stab["monthly"].to_csv(stab_path, index=False)
                result["stability_path"] = str(stab_path)
                summary_path = SAVE_DIR / f"lstm_stability_summary_{file_tag}_val.json"
                summary = {k: v for k, v in stab.items() if k != "monthly"}
                summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
                log.info("Saved monthly stability -> %s", stab_path)
        return result

    # Final test: refit on train+val sequences, score test once
    log.warning("GOLDEN RULE UNLOCK: scoring TEST %s -> %s", result["test_start"], result["test_end"])
    seq_len = int(best_params["seq_len"])
    hidden = int(best_params["hidden_size"])
    epochs = int(best_params["epochs"])

    fit_panel = pd.concat([train, val], axis=0, ignore_index=True)
    hist_fit = fit_panel if WARM_START else None
    hist_tr = train if WARM_START else None
    X_fit, y_fit, cols, _, _ = _build_sequences_cached(
        fit_panel,
        feature_set,
        seq_len,
        target_col="target_model",
        feature_cols=cols_override,
        require_contiguous=REQUIRE_CONTIGUOUS,
    )
    # Test warm-start from train+val; val monitor warm-start from train
    X_te, _, _, keys_te, _ = _build_sequences_cached(
        test,
        feature_set,
        seq_len,
        target_col="target_model",
        feature_cols=cols_override,
        history=hist_fit,
        require_contiguous=REQUIRE_CONTIGUOUS,
    )
    if X_fit is None or X_te is None or len(keys_te) == 0:
        raise ValueError("Not enough sequences for final test refit.")

    X_va, y_va, _, _, _ = _build_sequences_cached(
        val,
        feature_set,
        seq_len,
        target_col="target_model",
        feature_cols=cols_override,
        history=hist_tr,
        require_contiguous=REQUIRE_CONTIGUOUS,
    )
    pred = fit_predict_lstm(
        X_fit, y_fit, X_te, hidden, epochs, BATCH_SIZE,
        X_va=X_va, y_va=y_va, patience=PATIENCE, lr=LR, seed=SEED,
        dropout=DROPOUT,
    )
    preview = _preview_from_keys(keys_te, pred, test)
    scored = _metrics_from_preview(preview)
    if len(scored["topk_picks"]):
        scored["topk_picks"].to_csv(topk_path, index=False)
    if len(scored["sector_picks"]):
        scored["sector_picks"].to_csv(sector_path, index=False)

    result.update(scored)
    result["evaluated_on"] = "test"
    result["test_rows"] = int(len(pred))
    result["feature_columns"] = cols
    result["n_features"] = len(cols)
    result["topk_path"] = str(topk_path) if len(scored["topk_picks"]) else None
    result["sector_path"] = str(sector_path) if len(scored["sector_picks"]) else None
    return result


def _run_yahoo_lstm(tickers: list[str] | None, include_alt: bool) -> dict:
    """Legacy Yahoo path (no golden-rule val split)."""
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    diss = str(DISS_MODELS.resolve())
    if diss not in sys.path:
        sys.path.insert(0, diss)

    from yahoo_panel import (
        DEFAULT_TICKERS,
        build_sequences,
        build_yahoo_panel,
        split_yahoo_panel,
        yahoo_feature_columns,
    )
    from models.lstm_model import fit_predict_lstm
    from evaluation.predictive import predictive_metrics

    selected = tickers or DEFAULT_TICKERS
    panel = build_yahoo_panel(selected, refresh=False)
    panel = panel[panel["Ticker"].isin(selected)].copy()
    train, test = split_yahoo_panel(
        panel,
        train_end=YAHOO_TRAIN_END,
        test_start=YAHOO_TEST_START,
        test_end=YAHOO_TEST_END,
    )
    if train.empty or test.empty:
        raise ValueError(
            f"Empty Yahoo split. train={len(train)} test={len(test)}. "
            "Run fetch_yahoo_data.py first or widen date range."
        )

    feature_cols = yahoo_feature_columns(include_alt=include_alt, panel=panel)
    X_tr, y_tr, cols, _ = build_sequences(train, feature_cols, SEQ_LEN)
    X_te, y_te, _, keys_te = build_sequences(test, feature_cols, SEQ_LEN)
    if X_tr is None or X_te is None or len(y_te) == 0:
        raise ValueError(f"Not enough Yahoo sequences (need > {SEQ_LEN} months per ticker).")

    pred = fit_predict_lstm(X_tr, y_tr, X_te, HIDDEN_SIZE, EPOCHS, BATCH_SIZE)
    pm = predictive_metrics(y_te, pred)
    tickers_out, months_out = zip(*keys_te) if keys_te else ([], [])
    preview = pd.DataFrame({
        "Ticker": list(tickers_out),
        "month": list(months_out),
        "y_true": y_te,
        "y_pred": pred,
        "error": y_te - pred,
    })
    return {
        "feature_set": "yahoo_full" + ("+alt" if include_alt else ""),
        "experiment_label": "yahoo_legacy",
        "n_features": len(cols),
        "feature_columns": cols,
        "best_params": {"seq_len": SEQ_LEN, "hidden_size": HIDDEN_SIZE, "epochs": EPOCHS},
        "seq_len": SEQ_LEN,
        "hidden_size": HIDDEN_SIZE,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "train_rows": len(y_tr),
        "val_rows": 0,
        "test_rows": len(y_te),
        "train_start": pd.to_datetime(train["month"]).min().date().isoformat(),
        "train_end": pd.to_datetime(train["month"]).max().date().isoformat(),
        "val_start": "",
        "val_end": "",
        "test_start": pd.to_datetime(test["month"]).min().date().isoformat(),
        "test_end": pd.to_datetime(test["month"]).max().date().isoformat(),
        "evaluated_on": "test",
        "data_source": "yahoo",
        "features_csv": str(ROOT / "saved_data" / "yahoo" / "panel_monthly.csv"),
        "mse": pm["mse"], "mae": pm["mae"], "rmse": pm["rmse"], "r2": pm["r2"],
        "ic_mean": None, "ic_std": None, "ic_ir": None,
        "long_short_mean": None, "long_short_std": None, "rank_n_months": 0,
        "preview": preview,
        "topk_picks": pd.DataFrame(),
        "topk_summary": {"n_picks": 0, "n_months": 0, "mean_edge": None, "hit_rate": None, "k": TOP_K,
                         "mean_pick": None, "mean_market": None},
        "sector_picks": pd.DataFrame(),
        "sector_summary": {},
        "topk_path": None,
        "sector_path": None,
    }


def print_report(result: dict, title: str | None = None) -> None:
    fs = result["feature_set"]
    on = result.get("evaluated_on", "unknown")
    banner = "VALIDATION (test locked)" if on == "validation" else "TEST (final unlock)"
    log.info("")
    log.info("=" * 72)
    log.info(title or ("RESULTS | LSTM %s | %s" % (fs, banner)))
    log.info("=" * 72)
    log.info(
        "Features: %s / %s (%s cols) | sector_neutral + winsor | seq=%s hidden=%s epochs=%s",
        result.get("experiment_label"), fs, result["n_features"],
        result.get("seq_len"), result.get("hidden_size"), result.get("epochs"),
    )
    log.info(
        "Split: train %s..%s (%s seq) | val %s..%s (%s) | test %s..%s (%s)",
        result["train_start"], result["train_end"], result["train_rows"],
        result["val_start"], result["val_end"], result["val_rows"],
        result["test_start"], result["test_end"], result["test_rows"],
    )
    if on == "validation":
        log.info("Window reported below: VALIDATION only. Test metrics intentionally hidden.")
    else:
        log.info("Window reported below: TEST. Do not retune after seeing these numbers.")

    log.info("")
    log.info("H1 LEVEL (exact number?)")
    log.info("  RMSE=%.6f  R2=%.4f  MAE=%.6f", result["rmse"], result["r2"], result["mae"])

    if result.get("ic_mean") is not None:
        log.info("")
        log.info("SPEARMAN IC (right order?)")
        log.info(
            "  IC mean=%.4f  std=%.4f  months=%s",
            result["ic_mean"], result["ic_std"], result["rank_n_months"],
        )
        log.info("  %s", _ic_plain_english(result["ic_mean"]))

        log.info("")
        log.info("LONG-SHORT (top 10%% - bottom 10%%)")
        log.info("  Mean=%.4f (%s/mo)", result["long_short_mean"], _pct(result["long_short_mean"]))
        log.info("  %s", _ls_plain_english(result["long_short_mean"]))

        ts = result["topk_summary"]
        log.info("")
        log.info("TOP-%s BOOK (primary — best predicted names, any sector)", ts.get("k", TOP_K))
        log.info("  months=%s picks=%s", ts["n_months"], ts["n_picks"])
        log.info("  Avg top-K return : %s", _pct(ts["mean_pick"]))
        log.info("  Avg market return: %s", _pct(ts["mean_market"]))
        log.info("  Avg edge         : %s", _pct(ts["mean_edge"]))
        if ts.get("hit_rate") is not None:
            log.info("  Months beat mkt  : %.1f%%", 100.0 * ts["hit_rate"])
        if result.get("topk_path"):
            log.info("  CSV: %s", result["topk_path"])

        ss = result.get("sector_summary") or {}
        log.info("")
        log.info("SECTOR BOOK (secondary — 1 pick per sector)")
        log.info(
            "  pick-sector edge=%s hit=%s",
            _pct(ss.get("mean_stock_minus_sector")),
            ss.get("hit_rate_beat_sector"),
        )
    else:
        log.info("(Ranking metrics unavailable for this run.)")

    log.info("")
    log.info("Note: first %s months per ticker excluded (sequence warm-up).", result.get("seq_len"))




def _write_wf_comparison(base: pd.DataFrame, sent: pd.DataFrame, path: Path) -> None:
    """Year-by-year baseline vs sentiment deltas (same schema as xgb_walk_forward)."""
    b = base[base["status"].isin(["ok", "average"])].copy()
    s = sent[sent["status"].isin(["ok", "average"])].copy()
    # Normalise legacy "AVERAGE" step label -> 0 so sort never mixes str/int
    for df in (b, s):
        step = df["step"].astype(str)
        df.loc[step.str.upper().eq("AVERAGE"), "step"] = 0
        df["step"] = pd.to_numeric(df["step"], errors="coerce")
        df["eval_year"] = df["eval_year"].astype(object)
        ey = df["eval_year"].astype(str)
        df.loc[df["status"].eq("average") & ey.isin(["", "nan", "None"]), "eval_year"] = "AVERAGE"
    b = b.rename(columns={
        "ic": "baseline_ic",
        "long_short": "baseline_ls",
        "topk_edge": "baseline_topk_edge",
        "hit_rate": "baseline_hit",
    })
    s = s.rename(columns={
        "ic": "sentiment_ic",
        "long_short": "sentiment_ls",
        "topk_edge": "sentiment_topk_edge",
        "hit_rate": "sentiment_hit",
    })
    keys = ["step", "eval_year"]
    m = b[keys + ["baseline_ic", "baseline_ls", "baseline_topk_edge", "baseline_hit"]].merge(
        s[keys + ["sentiment_ic", "sentiment_ls", "sentiment_topk_edge", "sentiment_hit"]],
        on=keys,
        how="outer",
    )
    m["delta_ic"] = m["sentiment_ic"] - m["baseline_ic"]
    m["delta_ls"] = m["sentiment_ls"] - m["baseline_ls"]
    m["delta_topk_edge"] = m["sentiment_topk_edge"] - m["baseline_topk_edge"]
    m["delta_hit"] = m["sentiment_hit"] - m["baseline_hit"]
    m = m.sort_values(["step"], na_position="last")
    path.parent.mkdir(parents=True, exist_ok=True)
    m.to_csv(path, index=False)
    log.info("Saved comparison -> %s", path)


def _mirror_copy_to_ensemble_lstm(
    src_root: Path, dst_root: Path, log_path: Path | None = None
) -> None:
    """Copy walk-forward artefacts into esemble/lstm for ensemble use."""
    import shutil

    dst_root.mkdir(parents=True, exist_ok=True)
    for name in ("walk_forward_baseline", "walk_forward_sentiment"):
        src = src_root / name
        dst = dst_root / name
        if not src.exists():
            continue
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        log.info("Mirrored %s -> %s", src, dst)

    for fname in (
        "walk_forward_baseline_summary.csv",
        "walk_forward_sentiment_summary.csv",
        "walk_forward_comparison.csv",
        "README_walk_forward.txt",
    ):
        src = src_root / fname
        if src.is_file():
            shutil.copy2(src, dst_root / fname)

    if log_path is not None and log_path.is_file():
        log_dir = dst_root / "log"
        log_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(log_path, log_dir / log_path.name)

    (dst_root / "README_ENSEMBLE_COPY.txt").write_text(
        "\n".join([
            "Copy of LSTM walk-forward outputs for ensemble use.",
            f"Source: {src_root}",
            f"Copied: {datetime.now().isoformat(timespec='seconds')}",
            "",
            "Recipe: R3a (liquidity drop 20%), baseline + sentiment H2.",
            "Primary outputs remain under saved_data/lstm_walk_forward/.",
            "",
        ]),
        encoding="utf-8",
    )
    log.info("Ensemble mirror copy -> %s", dst_root)


def _write_rolling_expanding_comparison(
    rolling: pd.DataFrame, expanding: pd.DataFrame, path: Path
) -> pd.DataFrame:
    """Year-by-year rolling (10y fixed) vs expanding train window — same eval years."""
    r = rolling[rolling["status"].isin(["ok", "average"])].copy()
    e = expanding[expanding["status"].isin(["ok", "average"])].copy()
    for df in (r, e):
        step = df["step"].astype(str)
        df.loc[step.str.upper().eq("AVERAGE"), "step"] = 0
        df["step"] = pd.to_numeric(df["step"], errors="coerce")
        df["eval_year"] = df["eval_year"].astype(object)
        ey = df["eval_year"].astype(str)
        df.loc[df["status"].eq("average") & ey.isin(["", "nan", "None"]), "eval_year"] = "AVERAGE"
    r = r.rename(columns={
        "ic": "rolling_ic",
        "long_short": "rolling_ls",
        "topk_edge": "rolling_topk_edge",
        "hit_rate": "rolling_hit",
        "train_start": "rolling_train_start",
        "train_end": "rolling_train_end",
    })
    e = e.rename(columns={
        "ic": "expanding_ic",
        "long_short": "expanding_ls",
        "topk_edge": "expanding_topk_edge",
        "hit_rate": "expanding_hit",
        "train_start": "expanding_train_start",
        "train_end": "expanding_train_end",
    })
    keys = ["step", "eval_year"]
    m = r[keys + [
        "rolling_ic", "rolling_ls", "rolling_topk_edge", "rolling_hit",
        "rolling_train_start", "rolling_train_end",
    ]].merge(
        e[keys + [
            "expanding_ic", "expanding_ls", "expanding_topk_edge", "expanding_hit",
            "expanding_train_start", "expanding_train_end",
        ]],
        on=keys,
        how="outer",
    )
    m["delta_ic"] = m["expanding_ic"] - m["rolling_ic"]
    m["delta_ls"] = m["expanding_ls"] - m["rolling_ls"]
    m["delta_topk_edge"] = m["expanding_topk_edge"] - m["rolling_topk_edge"]
    m["delta_hit"] = m["expanding_hit"] - m["rolling_hit"]
    m = m.sort_values(["step"], na_position="last")
    path.parent.mkdir(parents=True, exist_ok=True)
    m.to_csv(path, index=False)
    log.info("Saved rolling vs expanding comparison -> %s", path)
    return m


def _resolve_train_years(
    roll_tr_start: int, roll_tr_end: int, ev_year: int, window_mode: str
) -> tuple[int, int]:
    if window_mode == "expanding":
        return EXPAND_START_YEAR, ev_year - 1
    if window_mode == "rolling":
        return roll_tr_start, roll_tr_end
    raise ValueError(f"window_mode must be 'rolling' or 'expanding', got {window_mode!r}")


def run_walk_forward(
    tickers: str | list[str] | None = TICKERS,
    feature_set: str = FEATURE_SET,
    feature_cols: list[str] | None = None,
    experiment_label: str | None = None,
    out_dir: Path | None = None,
    panel: pd.DataFrame | None = None,
    window_mode: str | None = None,
) -> dict:
    """Walk-forward validation: rolling (10y train) or expanding train window.

    Honest OOS: no early-stopping on the evaluation year (fixed epochs only).
    """
    wm = window_mode or WINDOW_MODE
    _configure_paths()
    warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)

    from data.build_panel import build_panel

    cols_override = list(feature_cols) if feature_cols is not None else list(LOCKED_FEATURE_COLS)
    label = experiment_label or EXPERIMENT_LABEL
    selected = _normalize_tickers(tickers)
    out_root = Path(out_dir) if out_dir is not None else WALK_FORWARD_DIR
    out_root.mkdir(parents=True, exist_ok=True)
    clear_seq_cache()

    if panel is None:
        pdata, _ = build_panel()
        panel = _filter_members(pdata.panel.copy())
        panel["month"] = pd.to_datetime(panel["month"])
        panel["year"] = panel["month"].dt.year
        if selected:
            panel = panel[panel["Ticker"].isin(selected)].copy()
    else:
        panel = panel.copy()
        if "year" not in panel.columns:
            panel["month"] = pd.to_datetime(panel["month"])
            panel["year"] = panel["month"].dt.year

    if panel.empty:
        raise ValueError("Empty panel after filters — cannot run walk-forward.")

    params = {"seq_len": SEQ_LEN, "hidden_size": HIDDEN_SIZE, "epochs": EPOCHS}
    log.info("")
    log.info("=" * 72)
    log.info(
        "LSTM WALK-FORWARD | %s | window=%s | seq=%s hidden=%s epochs=%s | seed=%s | steps=%d | honest_oos=True",
        label, wm, SEQ_LEN, HIDDEN_SIZE, EPOCHS, SEED, len(WF_WINDOWS),
    )
    log.info(
        "Warm-start=%s | contiguous=%s | liquidity_drop=%s | sector_neutral=%s | members=%s",
        WARM_START, REQUIRE_CONTIGUOUS, LIQUIDITY_DROP, SECTOR_NEUTRAL_TARGET, FILTER_MEMBERS,
    )
    try:
        import torch
        log.info(
            "Torch device: %s | AMP=%s",
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            bool(torch.cuda.is_available()),
        )
    except Exception:
        pass
    log.info("Features (%d): %s", len(cols_override), ", ".join(cols_override))
    log.info("Output: %s", out_root)
    log.info("=" * 72)

    summary_rows: list[dict] = []
    total_gap_dropped = 0
    total_candidates = 0

    for step, roll_tr_start, roll_tr_end, ev_year, note in WF_WINDOWS:
        tr_start, tr_end = _resolve_train_years(roll_tr_start, roll_tr_end, ev_year, wm)
        step_dir = out_root / f"step_{step:02d}_{ev_year}"
        step_dir.mkdir(parents=True, exist_ok=True)

        train = panel[(panel["year"] >= tr_start) & (panel["year"] <= tr_end)].copy()
        eval_df = panel[panel["year"] == ev_year].copy()
        if LIQUIDITY_DROP and LIQUIDITY_DROP > 0:
            train = _apply_liquidity_filter(train, LIQUIDITY_DROP)
            eval_df = _apply_liquidity_filter(eval_df, LIQUIDITY_DROP)
        if train.empty or eval_df.empty:
            log.warning(
                "Step %02d | train %d-%d -> eval %d | SKIP empty (train=%d eval=%d)",
                step, tr_start, tr_end, ev_year, len(train), len(eval_df),
            )
            row = {
                "step": step,
                "train_start": tr_start,
                "train_end": tr_end,
                "eval_year": ev_year,
                "note": note,
                "status": "skipped_empty",
                "ic": None,
                "long_short": None,
                "topk_avg_return": None,
                "topk_edge": None,
                "hit_rate": None,
                "n_stocks": 0,
                "n_eval_rows": 0,
                "n_months": 0,
            }
            summary_rows.append(row)
            (step_dir / "metrics.json").write_text(
                json.dumps(row, indent=2, default=str), encoding="utf-8"
            )
            continue

        train, eval_df = _add_model_target(train, eval_df)
        used, frame, pack, cols = _fit_one(
            params, train, eval_df, feature_set, cols_override,
            early_stop_on_eval=False,
        )
        if frame is None or pack is None or frame.empty:
            log.warning(
                "Step %02d | train %d-%d -> eval %d | SKIP not enough sequences",
                step, tr_start, tr_end, ev_year,
            )
            row = {
                "step": step,
                "train_start": tr_start,
                "train_end": tr_end,
                "eval_year": ev_year,
                "note": note,
                "status": "skipped_no_sequences",
                "ic": None,
                "long_short": None,
                "topk_avg_return": None,
                "topk_edge": None,
                "hit_rate": None,
                "n_stocks": int(eval_df["Ticker"].nunique()),
                "n_eval_rows": 0,
                "n_months": 0,
            }
            summary_rows.append(row)
            (step_dir / "metrics.json").write_text(
                json.dumps(row, indent=2, default=str), encoding="utf-8"
            )
            continue

        scored = _metrics_from_preview(frame)
        ts = scored["topk_summary"]
        n_stocks = int(frame["Ticker"].nunique())
        st_eval = pack.get("stats_eval") or {}
        n_cand = int(st_eval.get("n_candidates") or 0)
        n_gap = int(st_eval.get("n_dropped_gap") or 0)
        total_candidates += n_cand
        total_gap_dropped += n_gap
        row = {
            "step": step,
            "train_start": tr_start,
            "train_end": tr_end,
            "eval_year": ev_year,
            "note": note,
            "status": "ok",
            "ic": scored["ic_mean"],
            "long_short": scored["long_short_mean"],
            "topk_avg_return": ts.get("mean_pick"),
            "topk_edge": ts.get("mean_edge"),
            "hit_rate": ts.get("hit_rate"),
            "n_stocks": n_stocks,
            "n_eval_rows": int(len(frame)),
            "n_months": scored["rank_n_months"],
            "rmse": scored["rmse"],
            "r2": scored["r2"],
            "train_rows": int(len(pack["y_tr"])),
            "seq_len": SEQ_LEN,
            "hidden_size": HIDDEN_SIZE,
            "epochs": EPOCHS,
            "seed": SEED,
            "feature_set": feature_set,
            "experiment_label": label,
            "n_features": len(cols),
            "feature_importance_top10": None,
            "honest_oos": True,
            "n_seq_candidates": n_cand,
            "n_seq_dropped_gap": n_gap,
            "require_contiguous": REQUIRE_CONTIGUOUS,
            "warm_start": WARM_START,
            "liquidity_drop": LIQUIDITY_DROP,
        }
        summary_rows.append(row)

        metrics_payload = dict(row)
        metrics_payload["ic_std"] = scored["ic_std"]
        (step_dir / "metrics.json").write_text(
            json.dumps(metrics_payload, indent=2, default=str), encoding="utf-8"
        )
        if len(scored["topk_picks"]):
            scored["topk_picks"].to_csv(step_dir / "top50_picks.csv", index=False)
        frame.to_csv(step_dir / "preview.csv", index=False)

        log.info(
            "Step %02d | train %d-%d -> eval %d | IC=%s LS=%s TopK-edge=%s hit=%s stocks=%d months=%s | %s",
            step, tr_start, tr_end, ev_year,
            ("%.4f" % row["ic"]) if row["ic"] is not None else "n/a",
            _pct(row["long_short"]),
            _pct(row["topk_edge"]),
            ("%.0f%%" % (100.0 * row["hit_rate"])) if row["hit_rate"] is not None else "n/a",
            n_stocks,
            row["n_months"],
            note or "",
        )

    summary = pd.DataFrame(summary_rows)
    ok = summary[summary["status"] == "ok"]
    avg = {
        "step": 0,  # numeric so sort/merge matches xgb_walk_forward
        "train_start": "",
        "train_end": "",
        "eval_year": "AVERAGE",
        "note": f"mean over {len(ok)} successful steps",
        "status": "average",
        "ic": float(ok["ic"].mean()) if len(ok) else None,
        "long_short": float(ok["long_short"].mean()) if len(ok) else None,
        "topk_avg_return": float(ok["topk_avg_return"].mean()) if len(ok) else None,
        "topk_edge": float(ok["topk_edge"].mean()) if len(ok) else None,
        "hit_rate": float(ok["hit_rate"].mean()) if len(ok) else None,
        "n_stocks": float(ok["n_stocks"].mean()) if len(ok) else None,
        "n_eval_rows": float(ok["n_eval_rows"].mean()) if len(ok) else None,
        "n_months": float(ok["n_months"].mean()) if len(ok) else None,
    }
    summary = pd.concat([summary, pd.DataFrame([avg])], ignore_index=True)
    summary_path = out_root / "summary_metrics.csv"
    summary.to_csv(summary_path, index=False)
    WF_OUT_ROOT.mkdir(parents=True, exist_ok=True)
    summary.to_csv(WF_OUT_ROOT / f"walk_forward_{label}_summary.csv", index=False)

    pct_gap = (
        (100.0 * total_gap_dropped / total_candidates) if total_candidates > 0 else None
    )
    log.info("")
    log.info("=" * 72)
    log.info("WALK-FORWARD SUMMARY | %s (successful steps only)", label)
    if pct_gap is not None:
        log.info(
            "Contiguous gap drops (eval): %d / %d candidates (%.1f%%)",
            total_gap_dropped, total_candidates, pct_gap,
        )
    log.info("=" * 72)
    log.info(
        "%-6s %10s %8s %10s %10s %8s %8s",
        "step", "eval", "IC", "LS/mo", "TopK-edge", "hit", "stocks",
    )
    for _, r in summary.iterrows():
        if r["status"] not in ("ok", "average"):
            continue
        stocks_disp = "n/a"
        if r["n_stocks"] is not None and pd.notna(r["n_stocks"]):
            stocks_disp = (
                ("%.0f" % r["n_stocks"])
                if r.get("status") == "average" or r.get("eval_year") == "AVERAGE"
                else str(int(r["n_stocks"]))
            )
        log.info(
            "%-6s %10s %8s %10s %10s %8s %8s",
            r["step"],
            r["eval_year"] if r["eval_year"] != "" else "AVG",
            ("%.4f" % r["ic"]) if r["ic"] is not None and pd.notna(r["ic"]) else "n/a",
            _pct(r["long_short"]),
            _pct(r["topk_edge"]),
            ("%.0f%%" % (100.0 * r["hit_rate"]))
            if r["hit_rate"] is not None and pd.notna(r["hit_rate"])
            else "n/a",
            stocks_disp,
        )
    log.info("Saved summary -> %s", summary_path)
    log.info(
        "Sequence cache: hits=%d misses=%d size=%d",
        _SEQ_CACHE_STATS["hits"], _SEQ_CACHE_STATS["misses"], len(_SEQ_CACHE),
    )

    return {
        "evaluated_on": "walk_forward",
        "window_mode": wm,
        "feature_set": feature_set,
        "experiment_label": label,
        "n_features": len(cols_override),
        "seq_len": SEQ_LEN,
        "hidden_size": HIDDEN_SIZE,
        "epochs": EPOCHS,
        "seed": SEED,
        "n_steps": len(WF_WINDOWS),
        "n_ok": int(len(ok)),
        "summary_path": str(summary_path),
        "out_dir": str(out_root),
        "average": avg,
        "summary": summary,
        "best_params": params,
        "train_rows": None,
        "val_rows": None,
        "test_rows": None,
        "train_start": str(WF_WINDOWS[0][1]),
        "train_end": str(WF_WINDOWS[-1][2]),
        "val_start": str(WF_WINDOWS[0][3]),
        "val_end": str(WF_WINDOWS[-1][3]),
        "test_start": "",
        "test_end": "",
        "mse": None,
        "mae": None,
        "rmse": None,
        "r2": None,
        "ic_mean": avg["ic"],
        "ic_std": float(ok["ic"].std(ddof=1)) if len(ok) > 1 else None,
        "ic_ir": None,
        "long_short_mean": avg["long_short"],
        "long_short_std": float(ok["long_short"].std(ddof=1)) if len(ok) > 1 else None,
        "rank_n_months": avg["n_months"],
        "val_rmse": None,
        "val_ic_mean": avg["ic"],
        "val_long_short_mean": avg["long_short"],
        "val_topk_edge": avg["topk_edge"],
        "topk_summary": {
            "k": TOP_K,
            "n_months": avg["n_months"],
            "n_picks": None,
            "mean_pick": avg["topk_avg_return"],
            "mean_market": None,
            "mean_edge": avg["topk_edge"],
            "hit_rate": avg["hit_rate"],
        },
        "sector_summary": {},
        "preview": pd.DataFrame(),
        "topk_path": None,
        "sector_path": None,
        "pct_seq_dropped_gaps": pct_gap,
        "n_seq_dropped_gap": total_gap_dropped,
        "n_seq_candidates": total_candidates,
    }


def run_walk_forward_h2(
    tickers: str | list[str] | None = TICKERS,
    baseline_only: bool = False,
    sentiment_only: bool = False,
) -> dict:
    """Run baseline (19 feats) and/or +sentiment walk-forward; write comparison CSV."""
    _configure_paths()
    from data.build_panel import build_panel

    WF_OUT_ROOT.mkdir(parents=True, exist_ok=True)
    selected = _normalize_tickers(tickers)
    pdata, _ = build_panel()
    panel = _filter_members(pdata.panel.copy())
    panel["month"] = pd.to_datetime(panel["month"])
    panel["year"] = panel["month"].dt.year
    if selected:
        panel = panel[panel["Ticker"].isin(selected)].copy()

    log.info(
        "Panel rows=%d | months %s -> %s | members_filter=%s",
        len(panel),
        panel["month"].min().date(),
        panel["month"].max().date(),
        FILTER_MEMBERS,
    )
    sent_cols = [c for c in SENTIMENT_CANDIDATES if c in panel.columns]
    for c in SENTIMENT_CANDIDATES:
        if c in panel.columns:
            log.info("Sentiment col %-28s non-null=%.1f%%", c, 100.0 * panel[c].notna().mean())
        else:
            log.info("Sentiment col %-28s MISSING", c)

    baseline_cols = list(LOCKED_FEATURE_COLS)
    sentiment_cols = list(LOCKED_FEATURE_COLS) + sent_cols

    base_result = None
    sent_result = None
    if not sentiment_only:
        base_result = run_walk_forward(
            tickers,
            FEATURE_SET,
            feature_cols=baseline_cols,
            experiment_label="baseline",
            out_dir=WF_OUT_ROOT / "walk_forward_baseline",
            panel=panel,
        )
    if not baseline_only:
        sent_result = run_walk_forward(
            tickers,
            FEATURE_SET,
            feature_cols=sentiment_cols,
            experiment_label="sentiment",
            out_dir=WF_OUT_ROOT / "walk_forward_sentiment",
            panel=panel,
        )

    if base_result is not None and sent_result is not None:
        _write_wf_comparison(
            base_result["summary"],
            sent_result["summary"],
            WF_OUT_ROOT / "walk_forward_comparison.csv",
        )

    readme = WF_OUT_ROOT / "README_walk_forward.txt"
    readme.write_text(
        "\n".join([
            "LSTM rolling walk-forward",
            "========================",
            f"Created: {datetime.now().isoformat(timespec='seconds')}",
            "",
            "Settings:",
            "  Rolling 10y train / 1y eval / 1y step",
            f"  seq={SEQ_LEN} hidden={HIDDEN_SIZE} epochs={EPOCHS} seed={SEED}",
            "  Honest OOS: NO early-stop on eval year",
            "  Target: sector-neutral, winsor 1%/99%",
            f"  Liquidity drop: bottom {LIQUIDITY_DROP:.0%} within month (R3a)" if LIQUIDITY_DROP else "  Liquidity drop: none",
            "  Warm-start: eval sequences prepend train history",
            "  Default arms: baseline + sentiment (H2); use --no-sentiment for baseline only",
            "",
            "Folders:",
            "  walk_forward_baseline/   — B_drop_bottom9 (19 features)",
            "  walk_forward_sentiment/  — 19 + sentiment columns",
            "  walk_forward_comparison.csv — year-by-year deltas",
            "",
        ]),
        encoding="utf-8",
    )

    primary = base_result or sent_result
    assert primary is not None
    primary["h2_baseline"] = base_result
    primary["h2_sentiment"] = sent_result
    primary["comparison_path"] = str(WF_OUT_ROOT / "walk_forward_comparison.csv")
    return primary



def _parse_winsor(s: str):
    """Parse --winsor '0.01,0.99' | 'none' -> (winsorise_bool, limits_tuple)."""
    raw = (s or "").strip().lower()
    if raw in ("none", "off", "0", "false"):
        return False, (0.01, 0.99)
    parts = [p.strip() for p in raw.split(",")]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("--winsor expects 'lo,hi' or 'none'")
    lo, hi = float(parts[0]), float(parts[1])
    if not (0.0 <= lo < hi <= 1.0):
        raise argparse.ArgumentTypeError("--winsor limits must satisfy 0 <= lo < hi <= 1")
    return True, (lo, hi)


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="Manual LSTM with golden-rule test lock")
    p.add_argument(
        "--final-test",
        action="store_true",
        help="Unlock and score the held-out TEST set once (use only after tuning is finished).",
    )
    p.add_argument(
        "--force-retest",
        action="store_true",
        help="Allow a second TEST unlock (bug fix only).",
    )
    p.add_argument(
        "--walk-forward",
        action="store_true",
        help="Run rolling walk-forward (baseline + sentiment). Default when no other mode.",
    )
    p.add_argument(
        "--window",
        default="rolling",
        choices=["rolling", "expanding"],
        help="Train window: rolling 10y fixed vs expanding from 2002 to eval_year-1.",
    )
    p.add_argument(
        "--compare-windows",
        action="store_true",
        help="Run expanding WF (R3a) and compare vs saved rolling R3a_liq20 summary.",
    )
    p.add_argument(
        "--with-sentiment",
        action="store_true",
        help="Walk-forward: run baseline AND +sentiment (H2 dual arm). Default when H2 on.",
    )
    p.add_argument(
        "--no-sentiment",
        action="store_true",
        help="Walk-forward: baseline 19 features only (skip sentiment arm).",
    )
    p.add_argument(
        "--baseline-only",
        action="store_true",
        help="Same as --no-sentiment (baseline 19 features only).",
    )
    p.add_argument(
        "--sentiment-only",
        action="store_true",
        help="Walk-forward: 19 + sentiment features only (no baseline arm).",
    )
    p.add_argument(
        "--skip-ensemble-copy",
        action="store_true",
        help="Do not mirror WF outputs into esemble/lstm/.",
    )
    p.add_argument(
        "--single-val",
        action="store_true",
        help="Legacy single 2023 validation window (old static split).",
    )
    p.add_argument(
        "--tune",
        action="store_true",
        help="Small seq/hidden grid on VALIDATION only (pick by IC/LS).",
    )
    p.add_argument(
        "--no-stability",
        action="store_true",
        help="Skip validation stability diagnostics (single-val mode only).",
    )
    p.add_argument(
        "--full-h1-clean",
        action="store_true",
        help="Use full h1_clean columns instead of B_drop_bottom9 (19 feats).",
    )
    p.add_argument(
        "--require-contiguous",
        action="store_true",
        help="Keep only sequences with calendar-contiguous seq_len lookback months.",
    )
    p.add_argument(
        "--no-warm-start",
        action="store_true",
        help="Do not prepend train history when building eval sequences.",
    )
    p.add_argument(
        "--no-sector-neutral",
        action="store_true",
        help="Turn SECTOR_NEUTRAL_TARGET off (monthly demean still applied).",
    )
    p.add_argument(
        "--winsor",
        default="",
        help="Winsor limits 'lo,hi' (default 0.01,0.99) or 'none'.",
    )
    p.add_argument(
        "--liquidity-drop",
        type=float,
        default=None,
        help="Drop bottom quantile within month by eqy_turnover (default 0.20 R3a; use 0 to disable).",
    )
    p.add_argument(
        "--no-members-filter",
        action="store_true",
        help="Disable members-only filter (expanded universe).",
    )
    p.add_argument(
        "--hidden",
        type=int,
        default=None,
        help="Override HIDDEN_SIZE (e.g. 48/64/96).",
    )
    p.add_argument(
        "--dropout",
        type=float,
        default=None,
        help="Post-LSTM dropout rate (e.g. 0.0/0.1/0.2).",
    )
    p.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Override learning rate (default 1e-3).",
    )
    p.add_argument(
        "--patience",
        type=int,
        default=None,
        help="Override early-stop patience (single-val/tune only).",
    )
    p.add_argument(
        "--out-tag",
        default="",
        help="Write WF under saved_data/lstm_robustness/<tag>/.",
    )
    p.add_argument(
        "--ledger-decision",
        default="pending",
        help="Decision label written to robustness_ledger.csv (adopt/reject/diagnostic-only).",
    )
    p.add_argument(
        "--notes",
        default="",
        help="Short note stored in lstm_results_tracker.csv (saved_data or final-test folder).",
    )
    return p.parse_args(argv)


def _apply_cli_overrides(args) -> None:
    """Apply robustness CLI flags to module globals for this process."""
    global REQUIRE_CONTIGUOUS, WARM_START, LIQUIDITY_DROP, DROPOUT, WINDOW_MODE
    global SECTOR_NEUTRAL_TARGET, WINSORISE_Y, WINSOR_LIMITS, FILTER_MEMBERS
    global HIDDEN_SIZE, LR, PATIENCE

    if args.require_contiguous:
        REQUIRE_CONTIGUOUS = True
    if args.no_warm_start:
        WARM_START = False
    if args.no_sector_neutral:
        SECTOR_NEUTRAL_TARGET = False
    if args.no_members_filter:
        FILTER_MEMBERS = False
    if args.liquidity_drop is not None:
        LIQUIDITY_DROP = float(args.liquidity_drop)
    if args.winsor:
        WINSORISE_Y, WINSOR_LIMITS = _parse_winsor(args.winsor)
    if args.hidden is not None:
        HIDDEN_SIZE = int(args.hidden)
    if args.dropout is not None:
        DROPOUT = float(args.dropout)
    if args.lr is not None:
        LR = float(args.lr)
    if args.patience is not None:
        PATIENCE = int(args.patience)
    if args.window:
        WINDOW_MODE = str(args.window)
    if args.compare_windows:
        WINDOW_MODE = "expanding"
        if not args.out_tag:
            _activate_out_tag("R3a_expanding_liq20")
    if args.out_tag:
        _activate_out_tag(args.out_tag)


if __name__ == "__main__":
    args = _parse_args()
    _configure_paths()
    _apply_cli_overrides(args)

    evaluate_test = bool(args.final_test or EVALUATE_TEST)
    if args.no_stability:
        globals()["RUN_STABILITY"] = False

    if evaluate_test:
        _activate_final_test_output_dir()

    log_path = setup_logging()

    if DATA_SOURCE == "yahoo":
        log.warning("Yahoo legacy mode — no golden-rule val split.")
        result = _run_yahoo_lstm(_normalize_tickers(TICKERS), INCLUDE_ALT_DATA)
        print_report(result, title="RESULTS | LSTM yahoo legacy")
        _append_results_tracker(result, notes=args.notes or "yahoo legacy")
        log.info("Done. Full log saved to %s", log_path)
        raise SystemExit(0)

    feat_cols = None if args.full_h1_clean else list(LOCKED_FEATURE_COLS)
    exp_label = "h1_clean_full" if args.full_h1_clean else EXPERIMENT_LABEL
    if args.out_tag:
        exp_label = args.out_tag

    # Default = walk-forward unless single-val / tune / final-test requested
    use_walk_forward = bool(args.walk_forward) or (
        not args.single_val and not args.tune and not evaluate_test
    )

    if evaluate_test:
        _assert_test_unlock_allowed(force_retest=args.force_retest)
        log.warning("Running with TEST unlocked (--final-test).")
        log.warning("All final-test artefacts will be saved under: %s", SAVE_DIR)
        use_walk_forward = False
    elif use_walk_forward:
        log.info(
            "Running ROLLING WALK-FORWARD validation (default). "
            "Use --single-val for old 2023-only val."
        )
        log.info("Validation outputs: SAVE_DIR=%s | WF=%s | LOG=%s", SAVE_DIR, WF_OUT_ROOT, LOG_DIR)
    else:
        log.info("Running single-val / tune mode. Pass --walk-forward for rolling windows.")
        log.info("Validation outputs: SAVE_DIR=%s | LOG_DIR=%s", SAVE_DIR, LOG_DIR)

    log.info(
        "PRIMARY LSTM: %s | seq=%s hidden=%s epochs=%s | walk_forward=%s tune=%s",
        exp_label, SEQ_LEN, HIDDEN_SIZE, EPOCHS, use_walk_forward, bool(args.tune),
    )
    log.info(
        "Robustness: contiguous=%s warm_start=%s liq_drop=%s sector_neutral=%s "
        "members=%s winsor=%s dropout=%s lr=%s",
        REQUIRE_CONTIGUOUS, WARM_START, LIQUIDITY_DROP, SECTOR_NEUTRAL_TARGET,
        FILTER_MEMBERS, (WINSOR_LIMITS if WINSORISE_Y else "none"), DROPOUT, LR,
    )
    log.info(
        "Preferred R3a WF ref: IC=%.4f LS=%+.2f%% TopK-edge=%+.2f%% hit=%.0f%%",
        PREFERRED_RECIPE_AVG["ic"],
        100.0 * PREFERRED_RECIPE_AVG["long_short"],
        100.0 * PREFERRED_RECIPE_AVG["topk_edge"],
        100.0 * PREFERRED_RECIPE_AVG["hit_rate"],
    )

    if use_walk_forward:
        # H2 dual arm (baseline + sentiment) unless opted out
        if args.sentiment_only:
            baseline_only, sentiment_only = False, True
        elif args.no_sentiment or args.baseline_only or args.compare_windows:
            baseline_only, sentiment_only = True, False
        elif args.with_sentiment or H2_INCLUDE_SENTIMENT:
            baseline_only, sentiment_only = False, False
        else:
            baseline_only, sentiment_only = True, False
        result = run_walk_forward_h2(
            TICKERS,
            baseline_only=baseline_only,
            sentiment_only=sentiment_only,
        )
        if args.compare_windows:
            expanding_path = Path(result["summary_path"])
            rolling_path = R3A_ROLLING_SUMMARY
            if not rolling_path.is_file():
                rolling_path = WALK_FORWARD_DIR / "summary_metrics.csv"
            if not rolling_path.is_file():
                log.error(
                    "Rolling reference missing. Expected %s or %s",
                    R3A_ROLLING_SUMMARY,
                    WALK_FORWARD_DIR / "summary_metrics.csv",
                )
            else:
                rolling_df = pd.read_csv(rolling_path)
                expanding_df = pd.read_csv(expanding_path)
                cmp_path = ROBUSTNESS_DIR / "rolling_vs_expanding_R3a_comparison.csv"
                cmp = _write_rolling_expanding_comparison(rolling_df, expanding_df, cmp_path)
                r_avg = rolling_df[rolling_df["status"] == "average"].iloc[0]
                e_avg = expanding_df[expanding_df["status"] == "average"].iloc[0]
                log.info("")
                log.info("=" * 72)
                log.info("ROLLING vs EXPANDING (R3a liquidity 20%%, baseline-only)")
                log.info("Rolling ref: %s", rolling_path)
                log.info("Expanding:   %s", expanding_path)
                log.info(
                    "AVERAGE | rolling IC=%.4f LS=%s TopK=%s hit=%s",
                    float(r_avg["ic"]),
                    _pct(r_avg["long_short"]),
                    _pct(r_avg["topk_edge"]),
                    ("%.0f%%" % (100.0 * r_avg["hit_rate"])),
                )
                log.info(
                    "AVERAGE | expanding IC=%.4f LS=%s TopK=%s hit=%s",
                    float(e_avg["ic"]),
                    _pct(e_avg["long_short"]),
                    _pct(e_avg["topk_edge"]),
                    ("%.0f%%" % (100.0 * e_avg["hit_rate"])),
                )
                log.info(
                    "AVERAGE | delta   IC=%+.4f LS=%s TopK=%s",
                    float(e_avg["ic"]) - float(r_avg["ic"]),
                    _pct(float(e_avg["long_short"]) - float(r_avg["long_short"])),
                    _pct(float(e_avg["topk_edge"]) - float(r_avg["topk_edge"])),
                )
                log.info("Comparison CSV: %s", cmp_path)
                log.info("=" * 72)
        for tag, block in (
            ("baseline", result.get("h2_baseline")),
            ("sentiment", result.get("h2_sentiment")),
        ):
            if not block:
                continue
            log.info(
                "WALK-FORWARD AVG [%s] | IC=%s LS=%s TopK-edge=%s hit=%s | steps_ok=%s/%s",
                tag,
                ("%.4f" % block["ic_mean"]) if block["ic_mean"] is not None else "n/a",
                _pct(block["long_short_mean"]),
                _pct((block.get("topk_summary") or {}).get("mean_edge")),
                (
                    "%.0f%%" % (100.0 * (block.get("topk_summary") or {}).get("hit_rate"))
                    if (block.get("topk_summary") or {}).get("hit_rate") is not None
                    else "n/a"
                ),
                block["n_ok"],
                block["n_steps"],
            )
            _append_results_tracker(
                block,
                notes=args.notes or f"walk-forward honest OOS {tag}",
            )
            if args.out_tag:
                _append_robustness_ledger(
                    experiment_id=args.out_tag,
                    result=block,
                    decision=args.ledger_decision,
                    notes=args.notes or tag,
                    pct_seq_dropped_gaps=block.get("pct_seq_dropped_gaps"),
                )
        if result.get("comparison_path"):
            log.info("Comparison CSV: %s", result["comparison_path"])
        if (
            not args.skip_ensemble_copy
            and not args.out_tag
            and not args.compare_windows
        ):
            _mirror_copy_to_ensemble_lstm(WF_OUT_ROOT, ENSEMBLE_LSTM_COPY_DIR, log_path)
    else:
        result = run_lstm_forecast(
            TICKERS,
            FEATURE_SET,
            evaluate_test=evaluate_test,
            feature_cols=feat_cols,
            experiment_label=exp_label,
            tune=bool(args.tune),
        )
        print_report(result, title="PRIMARY | LSTM " + exp_label)
        _append_results_tracker(
            result,
            notes=args.notes or (
                "final test unlock"
                if evaluate_test
                else ("tune grid" if args.tune else "single-val frozen")
            ),
        )
        if args.out_tag:
            # Single-val pilots still get a ledger row (n_steps blank)
            ledger_result = dict(result)
            ledger_result.setdefault("n_ok", 1)
            ledger_result.setdefault("n_steps", 1)
            ledger_result.setdefault(
                "average",
                {
                    "n_stocks": result.get("val_rows"),
                    "n_eval_rows": result.get("val_rows"),
                },
            )
            ledger_result.setdefault("out_dir", str(SAVE_DIR))
            _append_robustness_ledger(
                experiment_id=args.out_tag,
                result=ledger_result,
                decision=args.ledger_decision,
                notes=args.notes or ("single-val" if not evaluate_test else "final-test"),
            )
        if evaluate_test:
            _write_test_lock(result, log_path)

    log.info("Done. Full log saved to %s", log_path)

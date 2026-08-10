"""Manual XGBoost — hybrid restored (full universe + sector features).

Restored (after stock-picking experiment underperformed):
  - No liquidity filter (full FTSE member panel ~109k train rows)
  - h1_hybrid: sector dummies + MKT_RF/MOM/tech/fund + sector-neutral target

Kept from later work:
  - IC / long-short metrics, top-K book, sector book report, val tuning, min trees

Golden rule:
  Default runs tune + report on VALIDATION only.
  Unlock TEST once with:  python xgb_return.py --final-test
  After a successful final test, further --final-test is blocked unless --force-retest.
  Final-test outputs (logs, picks, tracker, lock) go to:
    final test results/xgboost/

Sector diagnosis (validation only):
  python xgb_return.py --diagnose-sector

Step 2 (done): widened grid -> winner h1_clean depth=4 lr=0.03 mcw=8 trees=178.

Primary frozen ML spec (default Play / no flags):
  B_drop_bottom9 — 19 features from h1_clean
  depth=4, lr=0.03, mcw=8, trees=178
  Validation = rolling walk-forward (10y train → 1y eval, step 1y, 14 steps 2012–2025)
  Outputs: final test results/xgboost/walk_forward/

Legacy single-year 2023 val:
  python xgb_return.py --single-val

Secondary parsimony check (do not retune):
  python xgb_return.py --parsimony-a   # A_top16 (16 features)

Feature robustness archive:
  python xgb_return.py --feature-robustness

Logging (Option A + B):
  Every message goes to the console AND logs/xgb_run_YYYYMMDD_HHMMSS.txt
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
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parent
DISS_MODELS = ROOT.parent / "dissertation-models"
FINAL_DATA = ROOT.parent.parent / "data" / "final data for AI"
MASTER_PANEL = FINAL_DATA / "07_panel" / "master_panel.parquet"
# Default (validation / tuning) outputs
SAVE_DIR = ROOT / "saved_data"
LOG_DIR = ROOT / "logs"
# One-shot final test: everything for the dissertation goes here
FINAL_TEST_DIR = ROOT / "final test results" / "xgboost"
RESULTS_TRACKER = SAVE_DIR / "xgb_results_tracker.csv"
TEST_LOCK_FILE = SAVE_DIR / "GOLDEN_RULE_test_already_scored.txt"
LEGACY_TEST_LOCK_FILE = ROOT / "saved_data" / "GOLDEN_RULE_test_already_scored.txt"

log = logging.getLogger("xgb_return")

TICKERS: str | list[str] | None = None
# Step 2 archive (re-enable with --compare-sets if needed)
COMPARE_FEATURE_SETS = False
COMPARE_SETS = ["traditional", "h1_clean"]
DIAGNOSE_SECTOR = False
DIAGNOSE_SECTOR_SETS = ["traditional", "h1_clean", "h1_hybrid"]

# --- Primary frozen ML spec (from xgb_run_20260801_155445 B_drop_bottom9) ---
FEATURE_SET = "h1_clean"  # named parent set; columns overridden by LOCKED_FEATURE_COLS
LOCKED_RECIPE = True
LOCKED_N_TREES = 178
LOCKED_PARAMS = {
    "max_depth": 4,
    "learning_rate": 0.03,
    "min_child_weight": 8,
    "subsample": 0.80,
    "colsample_bytree": 0.80,
}
# Main machine-learning specification (19 features)
LOCKED_FEATURE_COLS = [
    "log_market_cap", "sma_20", "SMB", "MOM", "bb_lower", "debt_equity", "CMA",
    "sma_15", "HML", "macd_signal", "volatility_30d", "price_to_sma_20", "atr",
    "volume_trend", "RMW", "bb_middle", "eqy_turnover", "revenue_growth", "macd",
]
# Secondary robustness / parsimony check only (16 features) — do not retune
SECONDARY_A_TOP16_COLS = [
    "log_market_cap", "sma_20", "SMB", "MOM", "bb_lower", "debt_equity", "CMA",
    "sma_15", "HML", "macd_signal", "volatility_30d", "price_to_sma_20", "atr",
    "volume_trend", "RMW", "bb_middle",
]
EXPERIMENT_LABEL = "B_drop_bottom9"

# After frozen fit, compute val stability diagnostics
RUN_STABILITY = True
STABILITY_ROLLING_MONTHS = 6

# Feature robustness archive (re-run with --feature-robustness)
ROBUST_TOP_N = 16
ROBUST_DROP_BOTTOM_N = 9
ROBUST_DEDUPE_DROP = ["sma_15", "bb_middle", "bb_upper", "ema_12"]

# Golden rule default: False = tune on validation only; do not score test.
EVALUATE_TEST = False

FILTER_MEMBERS = True
LIQUIDITY_FILTER = False  # frozen primary: no liquidity cut (10%/20% = robustness only)
LIQUIDITY_MIN_PCT = 0.20  # used only when LIQUIDITY_FILTER=True
DEMEAN_WITHIN_MONTH = True  # frozen target
SECTOR_NEUTRAL_TARGET = True  # frozen: demean within month x Sector
WINSORISE_Y = True
WINSOR_LIMITS = (0.01, 0.99)  # frozen winsor
EXCLUDE_SECTORS = {"Unknown", ""}

TOP_K = 50
TOP_FRAC = 0.10
# Extra columns attached to preview / Top-50 CSV for post-hoc diagnostics (not used in fit).
DIAG_ATTACH_COLS = [
    "log_market_cap", "eqy_turnover", "cur_mkt_cap",
    "MOM", "SMB", "HML", "RMW", "CMA",
]

N_ESTIMATORS_MAX = 800
EARLY_STOPPING_ROUNDS = 80
MIN_TREES = 50
# Full Step-2 grid (used only if LOCKED_RECIPE=False / --compare-sets).
STEP2_PARAM_GRID = [
    {"max_depth": 3, "learning_rate": 0.03, "min_child_weight": 8, "subsample": 0.80, "colsample_bytree": 0.80},
    {"max_depth": 3, "learning_rate": 0.05, "min_child_weight": 10, "subsample": 0.85, "colsample_bytree": 0.85},
    {"max_depth": 4, "learning_rate": 0.03, "min_child_weight": 8, "subsample": 0.80, "colsample_bytree": 0.80},
    {"max_depth": 4, "learning_rate": 0.05, "min_child_weight": 10, "subsample": 0.85, "colsample_bytree": 0.85},
    {"max_depth": 4, "learning_rate": 0.03, "min_child_weight": 12, "subsample": 0.75, "colsample_bytree": 0.75},
    {"max_depth": 5, "learning_rate": 0.03, "min_child_weight": 10, "subsample": 0.80, "colsample_bytree": 0.80},
    {"max_depth": 5, "learning_rate": 0.05, "min_child_weight": 12, "subsample": 0.80, "colsample_bytree": 0.80},
    {"max_depth": 5, "learning_rate": 0.02, "min_child_weight": 8, "subsample": 0.85, "colsample_bytree": 0.85},
]
# Frozen to Step-2 winner when LOCKED_RECIPE=True (single candidate, fixed trees).
PARAM_GRID = [dict(LOCKED_PARAMS)]


def _activate_final_test_output_dir() -> None:
    """Route all SAVE_DIR / LOG_DIR / tracker / lock outputs to final test results/xgboost."""
    global SAVE_DIR, LOG_DIR, RESULTS_TRACKER, TEST_LOCK_FILE
    FINAL_TEST_DIR.mkdir(parents=True, exist_ok=True)
    SAVE_DIR = FINAL_TEST_DIR
    LOG_DIR = FINAL_TEST_DIR
    RESULTS_TRACKER = SAVE_DIR / "xgb_results_tracker.csv"
    TEST_LOCK_FILE = SAVE_DIR / "GOLDEN_RULE_test_already_scored.txt"
    # Snapshot frozen spec next to test artefacts (for the results chapter)
    src_spec = ROOT / "saved_data" / "xgb_frozen_spec.json"
    if src_spec.is_file():
        dst_spec = SAVE_DIR / "xgb_frozen_spec.json"
        if not dst_spec.is_file() or src_spec.read_bytes() != dst_spec.read_bytes():
            dst_spec.write_bytes(src_spec.read_bytes())


def setup_logging() -> Path:
    """Option A + B: same messages to console and a timestamped log file."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"xgb_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    logger = logging.getLogger("xgb_return")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False
    fmt = logging.Formatter("%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    # Windows consoles are often cp1252 — avoid UnicodeEncodeError on arrows/deltas
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


def _existing_test_lock_path() -> Path | None:
    """Prefer final-test folder lock; fall back to legacy saved_data lock."""
    for path in (TEST_LOCK_FILE, FINAL_TEST_DIR / "GOLDEN_RULE_test_already_scored.txt", LEGACY_TEST_LOCK_FILE):
        if path.is_file():
            return path
    return None


def _assert_test_unlock_allowed(force_retest: bool) -> None:
    """Block a second look at the test set unless explicitly forced."""
    lock_path = _existing_test_lock_path()
    if lock_path is None:
        return
    if force_retest:
        log.warning(
            "GOLDEN RULE OVERRIDE: --force-retest used. Test was already scored. "
            "Prior lock:\n%s",
            lock_path.read_text(encoding="utf-8")[:800],
        )
        return
    raise RuntimeError(
        "GOLDEN RULE: test set was already scored once.\n"
        f"Lock file: {lock_path}\n"
        "Do not re-check 2024–2025 after tuning. "
        "If you truly need a re-run (bugfix only), pass --force-retest."
    )


def _write_test_lock(result: dict, log_path: Path | None) -> None:
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    lines = [
        f"scored_at={datetime.now().isoformat(timespec='seconds')}",
        f"feature_set={result.get('feature_set')}",
        f"experiment_label={result.get('experiment_label')}",
        f"output_dir={SAVE_DIR}",
        f"test_start={result.get('test_start')} test_end={result.get('test_end')}",
        f"rmse={result.get('rmse')} ic_mean={result.get('ic_mean')} "
        f"long_short_mean={result.get('long_short_mean')}",
        f"topk_edge={(result.get('topk_summary') or {}).get('mean_edge')}",
        f"best_params={json.dumps(result.get('best_params') or {}, default=str)}",
        f"log_file={log_path or ''}",
        "",
        "This file means the dissertation TEST window was unlocked. "
        "Do not tune further against these numbers.",
    ]
    text = "\n".join(lines)
    TEST_LOCK_FILE.write_text(text, encoding="utf-8")
    log.info("Wrote golden-rule lock: %s", TEST_LOCK_FILE)
    # Pointer in saved_data so a later accidental --final-test is still blocked
    LEGACY_TEST_LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
    LEGACY_TEST_LOCK_FILE.write_text(
        text + f"\ncanonical_output_dir={FINAL_TEST_DIR}\n",
        encoding="utf-8",
    )
    readme = SAVE_DIR / "README_final_test_outputs.txt"
    readme.write_text(
        "\n".join([
            "XGBoost FINAL TEST outputs (dissertation)",
            "========================================",
            f"Folder: {SAVE_DIR}",
            "",
            "Key files:",
            "  xgb_run_YYYYMMDD_HHMMSS.txt     - full console log",
            "  xgb_results_tracker.csv         - metrics row (test columns filled)",
            "  GOLDEN_RULE_test_already_scored.txt - one-shot lock",
            "  xgb_topk_picks_B_drop_bottom9_test.csv",
            "  xgb_sector_picks_B_drop_bottom9_test.csv",
            "  xgb_preview_B_drop_bottom9_test.csv",
            "  xgb_frozen_spec.json            - frozen model recipe snapshot",
            "",
            "Do not retune the model using these test numbers.",
            "",
        ]),
        encoding="utf-8",
    )
    log.info("Wrote output folder guide: %s", readme)


def _append_results_tracker(result: dict, notes: str = "") -> None:
    """One row per run: test metrics blank while golden rule is locked."""
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    bp = result.get("best_params") or {}
    ts = result.get("topk_summary") or {}
    on_test = result.get("evaluated_on") == "test"
    row = {
        "run_id": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "date": datetime.now().strftime("%d/%m/%Y"),
        "model": "XGB",
        "feature_set": result.get("feature_set"),
        "depth": bp.get("max_depth"),
        "lr": bp.get("learning_rate"),
        "trees": bp.get("final_n_estimators", bp.get("best_n_estimators")),
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


def _normalize_tickers(tickers: str | list[str] | None) -> list[str] | None:
    if tickers is None:
        return None
    if isinstance(tickers, str):
        return [tickers.upper()]
    return [str(t).upper() for t in tickers]


def _filter_members(df: pd.DataFrame) -> pd.DataFrame:
    if not FILTER_MEMBERS or "is_member" not in df.columns:
        return df
    mask = df["is_member"]
    if mask.dtype != bool:
        mask = mask.astype(bool)
    return df.loc[mask].copy()


def _liquidity_thresholds(train: pd.DataFrame) -> dict[str, float]:
    """Fit size/turnover floors on train only."""
    thr: dict[str, float] = {}
    for col in ("log_market_cap", "eqy_turnover", "cur_mkt_cap"):
        if col in train.columns and train[col].notna().any():
            thr[col] = float(train[col].quantile(LIQUIDITY_MIN_PCT))
    return thr


def _apply_liquidity(df: pd.DataFrame, thr: dict[str, float]) -> pd.DataFrame:
    if not LIQUIDITY_FILTER or not thr:
        return df
    out = df.copy()
    mask = pd.Series(True, index=out.index)
    # Prefer log_market_cap; else cur_mkt_cap; always AND turnover if present
    if "log_market_cap" in thr and "log_market_cap" in out.columns:
        mask &= out["log_market_cap"].fillna(-np.inf) >= thr["log_market_cap"]
    elif "cur_mkt_cap" in thr and "cur_mkt_cap" in out.columns:
        mask &= out["cur_mkt_cap"].fillna(-np.inf) >= thr["cur_mkt_cap"]
    if "eqy_turnover" in thr and "eqy_turnover" in out.columns:
        mask &= out["eqy_turnover"].fillna(-np.inf) >= thr["eqy_turnover"]
    return out.loc[mask].copy()


def _sector_series(df: pd.DataFrame) -> pd.Series:
    if "Sector" not in df.columns:
        return pd.Series("Unknown", index=df.index, dtype=object)
    return df["Sector"].fillna("Unknown").astype(str).replace({"": "Unknown"})


def _add_model_target(train: pd.DataFrame, *others: pd.DataFrame) -> tuple[pd.DataFrame, ...]:
    frames = [train] + list(others)
    out: list[pd.DataFrame] = []
    for df in frames:
        d = df.copy()
        d["_sector"] = _sector_series(d)
        y = d["target_excess_return"].astype(float)
        if SECTOR_NEUTRAL_TARGET and "month" in d.columns:
            y = y - d.groupby(["month", "_sector"])["target_excess_return"].transform("mean")
        elif DEMEAN_WITHIN_MONTH and "month" in d.columns:
            y = y - d.groupby("month")["target_excess_return"].transform("mean")
        d["target_model"] = y
        out.append(d)
    if WINSORISE_Y:
        lo, hi = out[0]["target_model"].quantile(list(WINSOR_LIMITS))
        for d in out:
            d["target_model"] = d["target_model"].clip(float(lo), float(hi))
    return tuple(out)


def _rank_score(frame: pd.DataFrame) -> float:
    """Validation objective: IC first, then long-short, then RMSE penalty."""
    from evaluation.predictive import ranking_metrics, predictive_metrics

    rm = ranking_metrics(frame)
    pm = predictive_metrics(frame["y_true"].values, frame["y_pred"].values)
    ic = rm["ic_mean"] if rm["ic_mean"] is not None else -1.0
    ls = rm["long_short_mean"] if rm["long_short_mean"] is not None else -1.0
    rmse = pm["rmse"] if pm["rmse"] is not None else 9.0
    return float(ic) + 0.5 * float(ls) - 0.01 * float(rmse)


def _fit_one(params, X_tr, y_tr, X_va, y_va, val_meta):
    if LOCKED_RECIPE:
        model = XGBRegressor(
            n_estimators=LOCKED_N_TREES,
            max_depth=params["max_depth"],
            learning_rate=params["learning_rate"],
            min_child_weight=params["min_child_weight"],
            subsample=params["subsample"],
            colsample_bytree=params["colsample_bytree"],
            random_state=42,
            n_jobs=2,
            verbosity=0,
        )
        model.fit(X_tr, y_tr)
    else:
        model = XGBRegressor(
            n_estimators=N_ESTIMATORS_MAX,
            max_depth=params["max_depth"],
            learning_rate=params["learning_rate"],
            min_child_weight=params["min_child_weight"],
            subsample=params["subsample"],
            colsample_bytree=params["colsample_bytree"],
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            random_state=42,
            n_jobs=2,
            verbosity=0,
        )
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    pred_va = model.predict(X_va)
    frame = pd.DataFrame({
        "Ticker": val_meta["Ticker"].astype(str).values,
        "month": pd.to_datetime(val_meta["month"]).values,
        "y_true": val_meta["target_excess_return"].astype(float).values,
        "y_pred": pred_va,
    })
    return model, params, frame


def _monthly_ic_ls(preview: pd.DataFrame, min_names: int = 10) -> pd.DataFrame:
    """Per-month Spearman IC and long-short spread on a prediction frame."""
    work = preview.copy()
    work["month"] = pd.to_datetime(work["month"])
    rows = []
    for month, g in work.groupby("month", sort=True):
        if len(g) < min_names:
            continue
        ic = g["y_pred"].corr(g["y_true"], method="spearman")
        n = len(g)
        k = max(1, int(np.floor(n * TOP_FRAC)))
        g_sorted = g.sort_values("y_pred")
        bottom = g_sorted.head(k)["y_true"].mean()
        top = g_sorted.tail(k)["y_true"].mean()
        ls = float(top - bottom) if np.isfinite(top) and np.isfinite(bottom) else np.nan
        # Top-K edge vs equal-weight market that month
        market = float(g["y_true"].mean())
        topk = float(g.nlargest(TOP_K, "y_pred")["y_true"].mean()) if len(g) >= TOP_K else np.nan
        edge = topk - market if np.isfinite(topk) else np.nan
        rows.append({
            "month": pd.Timestamp(month).strftime("%Y-%m-%d"),
            "n_names": int(n),
            "ic": float(ic) if np.isfinite(ic) else np.nan,
            "long_short": ls,
            "topk_edge": edge,
        })
    return pd.DataFrame(rows)


def compute_stability_diagnostics(
    preview: pd.DataFrame,
    train: pd.DataFrame,
    feature_set: str,
    params: dict,
    n_trees: int,
    rolling: int = STABILITY_ROLLING_MONTHS,
    feature_cols: list[str] | None = None,
) -> dict:
    """Validation-only stability: rolling IC, positive-month rates, importance stability."""
    from data.features import prepare_xy

    monthly = _monthly_ic_ls(preview)
    if monthly.empty:
        return {"ok": False, "reason": "no monthly IC rows"}

    monthly = monthly.sort_values("month").reset_index(drop=True)
    monthly["rolling_ic"] = monthly["ic"].rolling(rolling, min_periods=rolling).mean()
    monthly["rolling_ls"] = monthly["long_short"].rolling(rolling, min_periods=rolling).mean()

    n = len(monthly)
    frac_ic_pos = float((monthly["ic"] > 0).mean()) if n else None
    frac_ls_pos = float((monthly["long_short"] > 0).mean()) if n else None
    frac_edge_pos = float((monthly["topk_edge"] > 0).mean()) if n else None
    rolling_vals = monthly["rolling_ic"].dropna()
    frac_roll_ic_pos = float((rolling_vals > 0).mean()) if len(rolling_vals) else None

    # Importance stability: refit locked recipe on early / mid / late train slices
    train_m = train.copy()
    train_m["month"] = pd.to_datetime(train_m["month"])
    months_sorted = sorted(train_m["month"].unique())
    imp_vectors: list[pd.Series] = []
    slice_labels: list[str] = []
    if len(months_sorted) >= 36:
        cuts = [
            ("early_train", months_sorted[: int(len(months_sorted) * 0.70)]),
            ("mid_train", months_sorted[int(len(months_sorted) * 0.15): int(len(months_sorted) * 0.85)]),
            ("late_train", months_sorted[int(len(months_sorted) * 0.30):]),
        ]
        for label, month_list in cuts:
            sub = train_m[train_m["month"].isin(month_list)]
            X, y, cols = prepare_xy(
                sub, feature_set, target_col="target_model", feature_cols=feature_cols
            )
            if X.empty or len(X) < 500:
                continue
            model = XGBRegressor(
                n_estimators=n_trees,
                max_depth=params["max_depth"],
                learning_rate=params["learning_rate"],
                min_child_weight=params["min_child_weight"],
                subsample=params["subsample"],
                colsample_bytree=params["colsample_bytree"],
                random_state=42,
                n_jobs=2,
                verbosity=0,
            )
            model.fit(X, y)
            imp_vectors.append(pd.Series(model.feature_importances_, index=cols, name=label))
            slice_labels.append(label)

    top10_jaccard = None
    imp_rank_corr = None
    top10_tables = {}
    if len(imp_vectors) >= 2:
        aligned = pd.concat(imp_vectors, axis=1).fillna(0.0)
        # Mean pairwise Spearman of importance ranks
        corrs = []
        jacs = []
        cols_list = list(aligned.columns)
        for i in range(len(cols_list)):
            for j in range(i + 1, len(cols_list)):
                a = aligned[cols_list[i]]
                b = aligned[cols_list[j]]
                c = a.rank(ascending=False).corr(b.rank(ascending=False), method="spearman")
                if np.isfinite(c):
                    corrs.append(float(c))
                top_a = set(a.nlargest(10).index)
                top_b = set(b.nlargest(10).index)
                union = top_a | top_b
                jacs.append(len(top_a & top_b) / len(union) if union else 0.0)
                top10_tables[cols_list[i]] = list(a.nlargest(10).index)
                top10_tables[cols_list[j]] = list(b.nlargest(10).index)
        imp_rank_corr = float(np.mean(corrs)) if corrs else None
        top10_jaccard = float(np.mean(jacs)) if jacs else None

    return {
        "ok": True,
        "n_months": n,
        "rolling_window": rolling,
        "ic_mean": float(monthly["ic"].mean()),
        "ic_std": float(monthly["ic"].std(ddof=1)) if n > 1 else 0.0,
        "frac_months_ic_positive": frac_ic_pos,
        "frac_months_ls_positive": frac_ls_pos,
        "frac_months_topk_edge_positive": frac_edge_pos,
        "rolling_ic_last": float(rolling_vals.iloc[-1]) if len(rolling_vals) else None,
        "frac_rolling_ic_positive": frac_roll_ic_pos,
        "importance_slices": slice_labels,
        "importance_rank_corr": imp_rank_corr,
        "top10_jaccard": top10_jaccard,
        "top10_by_slice": top10_tables,
        "monthly": monthly,
    }


def print_stability_report(stab: dict, feature_set: str) -> None:
    if not stab or not stab.get("ok"):
        log.info("Stability diagnostics unavailable: %s", (stab or {}).get("reason", "n/a"))
        return
    log.info("")
    log.info("=" * 72)
    log.info("STEP 3 STABILITY | %s | validation only", feature_set)
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
    if stab.get("rolling_ic_last") is not None:
        log.info("")
        log.info("Rolling %dm IC series (first fully-populated windows):", rw)
        m = stab["monthly"]
        for _, row in m.dropna(subset=["rolling_ic"]).iterrows():
            log.info("  %s  rolling_ic=%.4f", row["month"], row["rolling_ic"])

    log.info("")
    log.info(
        "Feature-importance stability (refits on early/mid/late train slices): "
        "rank_corr=%s  top10_jaccard=%s",
        ("%.3f" % stab["importance_rank_corr"]) if stab["importance_rank_corr"] is not None else "n/a",
        ("%.3f" % stab["top10_jaccard"]) if stab["top10_jaccard"] is not None else "n/a",
    )
    for label, names in (stab.get("top10_by_slice") or {}).items():
        log.info("  %s top10: %s", label, ", ".join(names[:10]))

    # Plain-English gate
    log.info("")
    log.info("Stability gate (rules of thumb for val-only):")
    ok_ic = (stab["frac_months_ic_positive"] or 0) >= 0.60
    ok_roll = stab["rolling_ic_last"] is not None and stab["rolling_ic_last"] > 0
    ok_imp = stab["top10_jaccard"] is None or stab["top10_jaccard"] >= 0.40
    if ok_ic and ok_roll and ok_imp:
        log.info("  PASS: edge is positive in most months; rolling IC>0; importances reasonably stable.")
    else:
        log.info("  CAUTION: check weak months / rolling IC / importance drift before unlocking test.")
        if not ok_ic:
            log.info("  - IC positive in fewer than 60%% of val months.")
        if not ok_roll:
            log.info("  - Latest rolling %dm IC is missing or <= 0.", rw)
        if not ok_imp:
            log.info("  - Top-10 feature Jaccard across train slices < 0.40.")


def _attach_diag_cols(frame: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    """Copy size / liquidity / factor columns from the panel meta onto a prediction frame."""
    if frame is None or frame.empty or meta is None or meta.empty:
        return frame
    out = frame.copy()
    cols = [c for c in DIAG_ATTACH_COLS if c in meta.columns]
    if not cols:
        return out
    if len(out) == len(meta):
        for c in cols:
            out[c] = meta[c].to_numpy()
        return out
    # Fallback: merge on Ticker + month
    keys = ["Ticker", "month"]
    right = meta[keys + cols].copy()
    right["Ticker"] = right["Ticker"].astype(str)
    right["month"] = pd.to_datetime(right["month"]).dt.strftime("%Y-%m-%d")
    out["Ticker"] = out["Ticker"].astype(str)
    out["month"] = pd.to_datetime(out["month"]).dt.strftime("%Y-%m-%d")
    # Drop any stale diag cols before merge
    out = out.drop(columns=[c for c in cols if c in out.columns], errors="ignore")
    return out.merge(right, on=keys, how="left")


def build_topk_book(preview: pd.DataFrame, k: int = TOP_K) -> tuple[pd.DataFrame, dict]:
    """Each month take top-K by y_pred; report vs equal-weight market that month."""
    if preview is None or preview.empty:
        return pd.DataFrame(), {"n_picks": 0, "n_months": 0, "mean_pick": None, "mean_market": None, "mean_edge": None, "hit_rate": None}

    work = preview.copy()
    work["month"] = pd.to_datetime(work["month"])
    market = work.groupby("month", as_index=False)["y_true"].mean().rename(columns={"y_true": "market_avg_return"})
    work = work.merge(market, on="month", how="left")

    picks = (
        work.sort_values(["month", "y_pred"], ascending=[True, False])
        .groupby("month", as_index=False)
        .head(k)
        .copy()
    )
    picks["pick_minus_market"] = picks["y_true"] - picks["market_avg_return"]
    picks["beat_market"] = picks["pick_minus_market"] > 0

    # Monthly portfolio = equal-weight of top-K
    monthly = picks.groupby("month", as_index=False).agg(
        port_return=("y_true", "mean"),
        market_return=("market_avg_return", "first"),
    )
    monthly["edge"] = monthly["port_return"] - monthly["market_return"]

    summary = {
        "n_picks": int(len(picks)),
        "n_months": int(monthly["month"].nunique()) if len(monthly) else 0,
        "k": int(k),
        "mean_pick": float(monthly["port_return"].mean()) if len(monthly) else None,
        "mean_market": float(monthly["market_return"].mean()) if len(monthly) else None,
        "mean_edge": float(monthly["edge"].mean()) if len(monthly) else None,
        "hit_rate": float((monthly["edge"] > 0).mean()) if len(monthly) else None,
        "monthly": monthly,
    }
    return picks, summary


def build_sector_diversified_book(preview: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    need = {"Ticker", "month", "Sector", "y_true", "y_pred"}
    empty = {
        "n_picks": 0, "n_months": 0, "n_sectors": 0,
        "mean_stock_return": None, "mean_sector_return": None, "mean_market_return": None,
        "mean_stock_minus_sector": None, "hit_rate_beat_sector": None,
    }
    if preview is None or preview.empty or not need.issubset(preview.columns):
        return pd.DataFrame(), empty

    work = preview.copy()
    work["month"] = pd.to_datetime(work["month"])
    work["Sector"] = work["Sector"].fillna("Unknown").astype(str)
    work = work[~work["Sector"].isin(EXCLUDE_SECTORS)].copy()

    market_by_month = work.groupby("month", as_index=False)["y_true"].mean().rename(columns={"y_true": "market_avg_return"})
    sector_avg = work.groupby(["month", "Sector"], as_index=False)["y_true"].mean().rename(columns={"y_true": "sector_avg_return"})
    work = work.merge(sector_avg, on=["month", "Sector"], how="left").merge(market_by_month, on="month", how="left")

    picks = (
        work.sort_values(["month", "Sector", "y_pred"], ascending=[True, True, False])
        .groupby(["month", "Sector"], as_index=False).head(1).copy()
    )
    picks["stock_minus_sector"] = picks["y_true"] - picks["sector_avg_return"]
    picks["beat_sector"] = picks["stock_minus_sector"] > 0

    return picks, {
        "n_picks": int(len(picks)),
        "n_months": int(picks["month"].nunique()) if len(picks) else 0,
        "n_sectors": int(picks["Sector"].nunique()) if len(picks) else 0,
        "mean_stock_return": float(picks["y_true"].mean()) if len(picks) else None,
        "mean_sector_return": float(picks["sector_avg_return"].mean()) if len(picks) else None,
        "mean_market_return": float(picks["market_avg_return"].mean()) if len(picks) else None,
        "mean_stock_minus_sector": float(picks["stock_minus_sector"].mean()) if len(picks) else None,
        "hit_rate_beat_sector": float(picks["beat_sector"].mean()) if len(picks) else None,
    }


def _pct(x) -> str:
    if x is None:
        return "n/a"
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "n/a"
    if not np.isfinite(v):
        return "n/a"
    return f"{100.0 * v:+.2f}%"


def _ic_plain_english(ic) -> str:
    if ic is None or not np.isfinite(ic):
        return "IC unavailable."
    if ic > 0.02:
        return "IC > 0: model ranking tended to match actual ranking (useful edge)."
    if ic > 0:
        return "IC slightly > 0: ranking only a little better than random."
    if ic > -0.02:
        return "IC ~ 0: ranking is basically random."
    return "IC < 0: model ranked stocks backwards vs what actually happened."


def _ls_plain_english(ls) -> str:
    if ls is None or not np.isfinite(ls):
        return "Long-short unavailable."
    if ls > 0:
        return "Positive spread: top 10% predicted names beat bottom 10% on average."
    if ls > -0.002:
        return "Spread ~ 0: top vs bottom picks were about equal."
    return "Negative spread: top 10% predicted names underperformed the bottom 10%."


def _metrics_from_preview(preview, cols, model) -> dict:
    from evaluation.predictive import ranking_metrics, predictive_metrics

    y_true = preview["y_true"].astype(float).values
    y_pred = preview["y_pred"].astype(float).values
    pm = predictive_metrics(y_true, y_pred)
    rank = ranking_metrics(preview)
    topk_picks, topk_summary = build_topk_book(preview, k=TOP_K)
    sector_picks, sector_summary = build_sector_diversified_book(preview)
    imp = (
        sorted(zip(cols, model.feature_importances_), key=lambda x: -x[1])
        if model is not None else []
    )
    return {
        "mse": pm["mse"], "mae": pm["mae"], "rmse": pm["rmse"], "r2": pm["r2"],
        "ic_mean": rank["ic_mean"], "ic_std": rank["ic_std"], "ic_ir": rank["ic_ir"],
        "long_short_mean": rank["long_short_mean"], "long_short_std": rank["long_short_std"],
        "rank_n_months": rank["n_months"],
        "feature_importance": imp,
        "preview": preview,
        "topk_picks": topk_picks,
        "topk_summary": topk_summary,
        "sector_picks": sector_picks,
        "sector_summary": sector_summary,
    }


def run_xgboost_forecast(
    tickers=TICKERS,
    feature_set=FEATURE_SET,
    evaluate_test: bool = False,
    feature_cols: list[str] | None = None,
    experiment_label: str | None = None,
    run_stability: bool | None = None,
) -> dict:
    _configure_paths()
    warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)

    from data.build_panel import build_panel
    from data.features import prepare_xy
    from evaluation.predictive import ranking_metrics, predictive_metrics

    do_stability = RUN_STABILITY if run_stability is None else bool(run_stability)

    selected = _normalize_tickers(tickers)
    _, splits = build_panel()
    train = _filter_members(splits["train"].copy())
    val = _filter_members(splits["val"].copy())
    test = _filter_members(splits["test"].copy())

    thr = _liquidity_thresholds(train)
    train = _apply_liquidity(train, thr)
    val = _apply_liquidity(val, thr)
    test = _apply_liquidity(test, thr)

    if selected:
        train = train[train["Ticker"].isin(selected)]
        val = val[val["Ticker"].isin(selected)]
        test = test[test["Ticker"].isin(selected)]
        if train.empty or val.empty:
            raise ValueError(f"No train/val rows for tickers {selected} after filters.")
        if evaluate_test and test.empty:
            raise ValueError(f"No test rows for tickers {selected} after filters.")

    train, val, test = _add_model_target(train, val, test)
    X_tr, y_tr, cols = prepare_xy(
        train, feature_set, target_col="target_model", feature_cols=feature_cols
    )
    X_va, y_va, _ = prepare_xy(
        val, feature_set, target_col="target_model", feature_cols=feature_cols
    )
    if X_tr.empty or X_va.empty:
        raise ValueError("Empty train/val feature matrix — check feature_set / filters / splits.")

    va_meta = val.loc[X_va.index]
    X_te = None
    te_meta = None
    if evaluate_test:
        X_te, _, _ = prepare_xy(
            test, feature_set, target_col="target_model", feature_cols=feature_cols
        )
        if X_te.empty:
            raise ValueError("Empty test feature matrix — check feature_set / filters / splits.")
        te_meta = test.loc[X_te.index]

    best_score = -1e18
    best_params: dict = {}
    best_va_frame = None
    best_model = None

    mode = "FINAL TEST (unlocked once)" if evaluate_test else "TUNE (validation only - test locked)"
    label = experiment_label or feature_set
    log.info("")
    log.info("--- %s | base=%s (%d cols) | mode=%s ---", label, feature_set, len(cols), mode)
    log.info(
        "Tuning on validation (%d rows) | grid_size=%d | liquidity_filter=%s thr=%s",
        len(y_va),
        len(PARAM_GRID),
        LIQUIDITY_FILTER,
        {k: round(v, 4) for k, v in thr.items()},
    )
    for params in PARAM_GRID:
        model, used, va_frame = _fit_one(params, X_tr, y_tr, X_va, y_va, va_meta)
        score = _rank_score(va_frame)
        if LOCKED_RECIPE:
            n_trees = LOCKED_N_TREES
        else:
            n_trees = int(model.n_estimators)
            if hasattr(model, "best_iteration") and model.best_iteration is not None:
                n_trees = max(MIN_TREES, int(model.best_iteration) + 1)
            else:
                n_trees = max(MIN_TREES, n_trees)
        rm = ranking_metrics(va_frame)
        if rm["ic_mean"] is not None and rm["long_short_mean"] is not None:
            log.info(
                "  depth=%s lr=%.3f mcw=%s trees=%s val_IC=%.4f val_LS=%.4f",
                used["max_depth"], used["learning_rate"], used["min_child_weight"],
                n_trees, rm["ic_mean"], rm["long_short_mean"],
            )
        else:
            log.info("  depth=%s trees=%s score=%.4f", used["max_depth"], n_trees, score)
        if score > best_score:
            best_score = score
            best_params = {**used, "best_n_estimators": n_trees}
            best_va_frame = va_frame
            best_model = model

    assert best_va_frame is not None and best_model is not None
    if LOCKED_RECIPE:
        n_trees_final = LOCKED_N_TREES
    else:
        n_trees_final = max(MIN_TREES, int(best_params.get("best_n_estimators", MIN_TREES)))
    best_params["final_n_estimators"] = n_trees_final
    best_params["locked_recipe"] = bool(LOCKED_RECIPE)

    # Honest validation metrics = train-only fits during tuning (not after train+val refit)
    vp = best_va_frame.copy()
    vp["month"] = pd.to_datetime(vp["month"]).dt.strftime("%Y-%m-%d")
    vp["Sector"] = _sector_series(va_meta).values
    vp = _attach_diag_cols(vp, va_meta)
    val_pm = predictive_metrics(vp["y_true"].values, vp["y_pred"].values)
    val_rank = ranking_metrics(vp)
    val_topk_picks, val_topk_summary = build_topk_book(vp, k=TOP_K)
    val_sector_picks, val_sector_summary = build_sector_diversified_book(vp)

    result = {
        "feature_set": feature_set,
        "n_features": len(cols),
        "feature_columns": cols,
        "best_params": best_params,
        "train_rows": len(y_tr),
        "val_rows": len(y_va),
        "test_rows": 0,
        "train_start": pd.to_datetime(train["month"]).min().date().isoformat(),
        "train_end": pd.to_datetime(train["month"]).max().date().isoformat(),
        "val_start": pd.to_datetime(val["month"]).min().date().isoformat(),
        "val_end": pd.to_datetime(val["month"]).max().date().isoformat(),
        "test_start": pd.to_datetime(test["month"]).min().date().isoformat(),
        "test_end": pd.to_datetime(test["month"]).max().date().isoformat(),
        "evaluated_on": "validation",
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
        "feature_importance": sorted(zip(cols, best_model.feature_importances_), key=lambda x: -x[1]),
        "preview": vp,
        "topk_picks": val_topk_picks,
        "topk_summary": val_topk_summary,
        "sector_picks": val_sector_picks,
        "sector_summary": val_sector_summary,
        "liquidity_thresholds": thr,
        "topk_path": None,
        "sector_path": None,
        "experiment_label": label,
        "feature_cols_override": list(cols) if feature_cols is not None else None,
    }

    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    suffix_tag = "test" if evaluate_test else "val"
    file_tag = (experiment_label or feature_set).replace(" ", "_")
    topk_path = SAVE_DIR / f"xgb_topk_picks_{file_tag}_{suffix_tag}.csv"
    sector_path = SAVE_DIR / f"xgb_sector_picks_{file_tag}_{suffix_tag}.csv"
    preview_path = SAVE_DIR / f"xgb_preview_{file_tag}_{suffix_tag}.csv"

    if not evaluate_test:
        log.info(
            "GOLDEN RULE: test set locked. Reporting VALIDATION only "
            "(%s -> %s). Re-run with --final-test when tuning is finished.",
            result["test_start"],
            result["test_end"],
        )
        if len(vp):
            vp.to_csv(preview_path, index=False)
            result["preview_path"] = str(preview_path)
            log.info("Saved full val preview (with diag cols) -> %s", preview_path)
        if len(val_topk_picks):
            val_topk_picks.to_csv(topk_path, index=False)
            result["topk_path"] = str(topk_path)
        if len(val_sector_picks):
            val_sector_picks.to_csv(sector_path, index=False)
            result["sector_path"] = str(sector_path)

        if do_stability:
            stab = compute_stability_diagnostics(
                vp, train, feature_set, best_params, n_trees_final,
                rolling=STABILITY_ROLLING_MONTHS,
                feature_cols=feature_cols,
            )
            result["stability"] = stab
            print_stability_report(stab, feature_set)
            if stab.get("ok") and stab.get("monthly") is not None:
                stab_path = SAVE_DIR / f"xgb_stability_monthly_{feature_set}_val.csv"
                stab["monthly"].to_csv(stab_path, index=False)
                result["stability_path"] = str(stab_path)
                log.info("Saved monthly stability -> %s", stab_path)
                summary_path = SAVE_DIR / f"xgb_stability_summary_{feature_set}_val.json"
                summary = {
                    k: v for k, v in stab.items()
                    if k not in ("monthly", "top10_by_slice")
                }
                summary["top10_by_slice"] = stab.get("top10_by_slice")
                summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
                log.info("Saved stability summary -> %s", summary_path)
        return result

    log.warning("GOLDEN RULE UNLOCK: scoring TEST %s -> %s", result["test_start"], result["test_end"])
    X_fit = pd.concat([X_tr, X_va], axis=0)
    y_fit = pd.concat([y_tr, y_va], axis=0)
    final = XGBRegressor(
        n_estimators=n_trees_final,
        max_depth=best_params["max_depth"],
        learning_rate=best_params["learning_rate"],
        min_child_weight=best_params["min_child_weight"],
        subsample=best_params["subsample"],
        colsample_bytree=best_params["colsample_bytree"],
        random_state=42,
        n_jobs=2,
        verbosity=0,
    )
    final.fit(X_fit, y_fit)

    pred = final.predict(X_te)
    y_raw = te_meta["target_excess_return"].astype(float).values
    preview = pd.DataFrame({
        "Ticker": te_meta["Ticker"].astype(str).values,
        "month": pd.to_datetime(te_meta["month"]).dt.strftime("%Y-%m-%d").values,
        "Sector": _sector_series(te_meta).values,
        "y_true": y_raw,
        "y_pred": pred,
        "error": y_raw - pred,
    }).sort_values(["Ticker", "month"])
    preview = _attach_diag_cols(preview, te_meta)

    scored = _metrics_from_preview(preview, cols, final)
    if len(preview):
        preview.to_csv(preview_path, index=False)
        result["preview_path"] = str(preview_path)
    if len(scored["topk_picks"]):
        scored["topk_picks"].to_csv(topk_path, index=False)
    if len(scored["sector_picks"]):
        scored["sector_picks"].to_csv(sector_path, index=False)

    result.update(scored)
    result["evaluated_on"] = "test"
    result["test_rows"] = len(pred)
    result["topk_path"] = str(topk_path) if len(scored["topk_picks"]) else None
    result["sector_path"] = str(sector_path) if len(scored["sector_picks"]) else None
    return result


def print_report(result: dict, title: str | None = None) -> None:
    fs = result["feature_set"]
    on = result.get("evaluated_on", "unknown")
    banner = "VALIDATION (test locked)" if on == "validation" else "TEST (final unlock)"
    log.info("")
    log.info("=" * 72)
    log.info(title or ("RESULTS | %s | %s" % (fs, banner)))
    log.info("=" * 72)
    log.info("Features: %s (%s cols) | sector_neutral + hybrid", fs, result["n_features"])
    log.info(
        "Split: train %s..%s (%s) | val %s..%s (%s) | test %s..%s (%s)",
        result["train_start"], result["train_end"], result["train_rows"],
        result["val_start"], result["val_end"], result["val_rows"],
        result["test_start"], result["test_end"], result["test_rows"],
    )
    bp = result["best_params"]
    log.info(
        "Tuned: depth=%s lr=%s mcw=%s trees~%s",
        bp["max_depth"], bp["learning_rate"], bp["min_child_weight"],
        bp.get("final_n_estimators", bp.get("best_n_estimators")),
    )
    if on == "validation":
        log.info("Window reported below: VALIDATION only. Test metrics intentionally hidden.")
    else:
        log.info("Window reported below: TEST. Do not retune after seeing these numbers.")

    log.info("")
    log.info("H1 LEVEL (exact number?)")
    log.info("  RMSE=%.6f  R2=%.4f  MAE=%.6f", result["rmse"], result["r2"], result["mae"])

    log.info("")
    log.info("SPEARMAN IC (right order?)")
    log.info("  Rank by prediction vs rank by actual return each month.")
    log.info("  IC>0 helpful | IC~0 random | IC<0 backwards")
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
    if ts["hit_rate"] is not None:
        log.info("  Months beat mkt  : %.1f%%", 100.0 * ts["hit_rate"])
    if result.get("topk_path"):
        log.info("  CSV: %s", result["topk_path"])

    ss = result["sector_summary"]
    log.info("")
    log.info("SECTOR BOOK (secondary — 1 pick per sector)")
    log.info(
        "  pick-sector edge=%s hit=%s",
        _pct(ss.get("mean_stock_minus_sector")),
        ss.get("hit_rate_beat_sector"),
    )
    if ss.get("hit_rate_beat_sector") is not None:
        log.info("  hit-rate beat sector: %.1f%%", 100.0 * ss["hit_rate_beat_sector"])

    log.info("")
    log.info("Top feature importance:")
    for name, score in result["feature_importance"][:12]:
        log.info("  %-32s %.4f", name, score)


def print_comparison(rows: list) -> dict:
    on = rows[0].get("evaluated_on", "test") if rows else "test"
    label = "VALIDATION" if on == "validation" else "TEST"
    log.info("")
    log.info("=" * 72)
    log.info("COMPARISON on %s (pick best by IC, then long-short — not R2)", label)
    log.info("=" * 72)
    log.info(
        "%-16s %6s %8s %8s %8s %8s %10s",
        "feature_set", "n_feat", "RMSE", "R2", "IC", "LS/mo", "TopK-edge",
    )
    ranked = sorted(
        rows,
        key=lambda r: (
            r["ic_mean"] if r["ic_mean"] is not None else -9,
            r["long_short_mean"] if r["long_short_mean"] is not None else -9,
        ),
        reverse=True,
    )
    for r in ranked:
        edge = r["topk_summary"].get("mean_edge")
        log.info(
            "%-16s %6d %8.4f %8.4f %8.4f %8s %10s",
            r["feature_set"], r["n_features"], r["rmse"], r["r2"],
            r["ic_mean"], _pct(r["long_short_mean"]), _pct(edge),
        )
    best = ranked[0]
    log.info(
        "BEST by IC/LS: %s  (IC=%.4f, LS=%s)",
        best["feature_set"], best["ic_mean"], _pct(best["long_short_mean"]),
    )
    return best


def _sector_importance_share(result: dict) -> tuple[float, float, str | None]:
    """Return (sector_share, top_sector_imp, top_sector_name) from feature_importance."""
    imp = result.get("feature_importance") or []
    if not imp:
        return 0.0, 0.0, None
    total = float(sum(max(0.0, float(s)) for _, s in imp))
    if total <= 0:
        return 0.0, 0.0, None
    sector_items = [(n, float(s)) for n, s in imp if str(n).startswith("sector__")]
    sector_sum = float(sum(s for _, s in sector_items))
    top_name, top_imp = (None, 0.0)
    if sector_items:
        top_name, top_imp = max(sector_items, key=lambda x: x[1])
    return sector_sum / total, top_imp / total, top_name


def _safe_delta(a, b) -> float | None:
    if a is None or b is None:
        return None
    try:
        return float(a) - float(b)
    except (TypeError, ValueError):
        return None


def print_sector_diagnosis(rows: list) -> dict:
    """Compare continuous-only vs hybrid to see how much edge is sector-driven (val only)."""
    by_name = {r["feature_set"]: r for r in rows}
    hybrid = by_name.get("h1_hybrid")
    clean = by_name.get("h1_clean")
    trad = by_name.get("traditional")
    baseline = clean or trad

    log.info("")
    log.info("=" * 72)
    log.info("SECTOR DIAGNOSIS (validation only — same pipeline, different feature sets)")
    log.info("=" * 72)
    log.info("Question: how much of hybrid performance is sector dummies / Financial Services?")
    log.info("")
    log.info(
        "%-14s %6s %8s %10s %10s %12s %10s",
        "feature_set", "n_feat", "IC", "LS/mo", "TopK-edge", "HitMkt", "SecImp%",
    )

    diag_rows = []
    for r in rows:
        edge = (r.get("topk_summary") or {}).get("mean_edge")
        hit = (r.get("topk_summary") or {}).get("hit_rate")
        sec_share, top_sec_share, top_sec = _sector_importance_share(r)
        log.info(
            "%-14s %6d %8.4f %10s %10s %11s %9.1f%%",
            r["feature_set"],
            r["n_features"],
            r["ic_mean"] if r["ic_mean"] is not None else float("nan"),
            _pct(r["long_short_mean"]),
            _pct(edge),
            ("%.0f%%" % (100.0 * hit)) if hit is not None else "n/a",
            100.0 * sec_share,
        )
        diag_rows.append({
            "feature_set": r["feature_set"],
            "n_features": r["n_features"],
            "evaluated_on": r.get("evaluated_on"),
            "val_start": r.get("val_start"),
            "val_end": r.get("val_end"),
            "ic": r.get("ic_mean"),
            "long_short": r.get("long_short_mean"),
            "topk_edge": edge,
            "topk_hit_rate": hit,
            "rmse": r.get("rmse"),
            "sector_importance_share": sec_share,
            "top_sector_feature": top_sec,
            "top_sector_importance_share": top_sec_share,
            "best_depth": (r.get("best_params") or {}).get("max_depth"),
            "best_lr": (r.get("best_params") or {}).get("learning_rate"),
            "best_trees": (r.get("best_params") or {}).get("final_n_estimators"),
        })

    if hybrid and baseline:
        d_ic = _safe_delta(hybrid.get("ic_mean"), baseline.get("ic_mean"))
        d_ls = _safe_delta(hybrid.get("long_short_mean"), baseline.get("long_short_mean"))
        d_edge = _safe_delta(
            (hybrid.get("topk_summary") or {}).get("mean_edge"),
            (baseline.get("topk_summary") or {}).get("mean_edge"),
        )
        sec_share, top_sec_share, top_sec = _sector_importance_share(hybrid)
        log.info("")
        log.info("Deltas: h1_hybrid minus %s", baseline["feature_set"])
        log.info("  Delta IC        = %+.4f", d_ic if d_ic is not None else float("nan"))
        log.info("  Delta LS/mo     = %s", _pct(d_ls))
        log.info("  Delta TopK-edge = %s", _pct(d_edge))
        log.info(
            "  Hybrid sector__* importance share = %.1f%%",
            100.0 * sec_share,
        )
        if top_sec:
            log.info(
                "  Top sector feature: %s (%.1f%% of total importance)",
                top_sec, 100.0 * top_sec_share,
            )

        # Plain-English call
        log.info("")
        log.info("Interpretation (validation only — do not unlock test yet):")
        if d_ic is not None and d_ic > 0.015 and sec_share >= 0.25:
            log.info(
                "  Sector features ADD clear ranking power, and sector dummies dominate importance."
            )
            log.info(
                "  Treat hybrid edge cautiously — part may be 'Financial Services did well in 2023'."
            )
        elif d_ic is not None and abs(d_ic) <= 0.01 and (d_edge is None or abs(d_edge) <= 0.002):
            log.info(
                "  Hybrid ~ continuous-only. Sector dummies are not the main driver of val edge."
            )
        elif d_ic is not None and d_ic < -0.01:
            log.info(
                "  Continuous-only BEATS hybrid on val IC. Prefer h1_clean/traditional for next tuning."
            )
        else:
            log.info(
                "  Mixed: hybrid helps a bit, but check whether top sector importance is concentrated."
            )
            if top_sec and top_sec_share >= 0.08:
                log.info(
                    "  Warning: %s alone is %.1f%% of importance - concentration risk.",
                    top_sec, 100.0 * top_sec_share,
                )

    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = SAVE_DIR / "xgb_sector_diagnosis_val.csv"
    pd.DataFrame(diag_rows).to_csv(out_csv, index=False)
    log.info("")
    log.info("Saved diagnosis table -> %s", out_csv)

    return print_comparison(rows)


def _build_robustness_sets(baseline_importance: list) -> dict[str, list[str]]:
    """Build A/B/C column lists from frozen-run feature_importance [(name, score), ...]."""
    from data.features import H1_CLEAN_COLS

    ranked = [n for n, _ in baseline_importance if n in H1_CLEAN_COLS]
    # Keep any h1_clean cols that somehow lacked importance at the end
    for c in H1_CLEAN_COLS:
        if c not in ranked:
            ranked.append(c)

    top_n = min(ROBUST_TOP_N, len(ranked))
    drop_n = min(ROBUST_DROP_BOTTOM_N, max(0, len(ranked) - 8))
    keep_n = len(ranked) - drop_n

    cols_a = ranked[:top_n]
    cols_b = ranked[:keep_n]
    cols_c = [c for c in H1_CLEAN_COLS if c not in ROBUST_DEDUPE_DROP]

    return {
        "A_top16": cols_a,
        "B_drop_bottom9": cols_b,
        "C_dedupe_tech": cols_c,
    }


def print_feature_robustness(rows: list[dict]) -> None:
    log.info("")
    log.info("=" * 72)
    log.info("FEATURE ROBUSTNESS (validation only | frozen h1_clean recipe)")
    log.info("=" * 72)
    log.info(
        "%-16s %6s %8s %10s %10s %8s",
        "experiment", "n_feat", "IC", "LS/mo", "TopK-edge", "HitMkt",
    )
    base = None
    for r in rows:
        label = r.get("experiment_label") or r["feature_set"]
        edge = (r.get("topk_summary") or {}).get("mean_edge")
        hit = (r.get("topk_summary") or {}).get("hit_rate")
        log.info(
            "%-16s %6d %8.4f %10s %10s %7s",
            label,
            r["n_features"],
            r["ic_mean"] if r["ic_mean"] is not None else float("nan"),
            _pct(r["long_short_mean"]),
            _pct(edge),
            ("%.0f%%" % (100.0 * hit)) if hit is not None else "n/a",
        )
        if label == "baseline":
            base = r

    if base is None or len(rows) < 2:
        return

    log.info("")
    log.info("Deltas vs baseline (IC / LS / TopK-edge):")
    for r in rows:
        label = r.get("experiment_label") or r["feature_set"]
        if label == "baseline":
            continue
        d_ic = _safe_delta(r.get("ic_mean"), base.get("ic_mean"))
        d_ls = _safe_delta(r.get("long_short_mean"), base.get("long_short_mean"))
        d_edge = _safe_delta(
            (r.get("topk_summary") or {}).get("mean_edge"),
            (base.get("topk_summary") or {}).get("mean_edge"),
        )
        log.info(
            "  %-14s  dIC=%+.4f  dLS=%s  dTopK=%s",
            label,
            d_ic if d_ic is not None else float("nan"),
            _pct(d_ls),
            _pct(d_edge),
        )
        # Adopt if not collapsed (IC within -0.01 of baseline or better)
        if d_ic is not None and d_ic < -0.015:
            log.info("    -> WEAKER: keep full h1_clean for this variant.")
        elif d_ic is not None and d_ic >= -0.005:
            log.info("    -> ROBUST: numbers hold (or improve) — candidate to adopt.")
        else:
            log.info("    -> MIXED: small drop — prefer full set unless you want a leaner model.")


def run_feature_robustness(notes: str = "") -> list[dict]:
    """Baseline + A/B/C on frozen recipe; validation only; no stability (speed)."""
    from data.features import H1_CLEAN_COLS

    log.info(
        "FEATURE ROBUSTNESS: frozen recipe on h1_clean variants "
        "(depth=%s lr=%s mcw=%s trees=%s). Test stays locked.",
        LOCKED_PARAMS["max_depth"],
        LOCKED_PARAMS["learning_rate"],
        LOCKED_PARAMS["min_child_weight"],
        LOCKED_N_TREES,
    )
    globals()["LOCKED_RECIPE"] = True
    globals()["PARAM_GRID"] = [dict(LOCKED_PARAMS)]

    baseline = run_xgboost_forecast(
        TICKERS,
        "h1_clean",
        evaluate_test=False,
        experiment_label="baseline",
        run_stability=False,
    )
    print_report(baseline, title="ROBUST | baseline h1_clean")
    _append_results_tracker(baseline, notes=notes or "robustness baseline")

    sets = _build_robustness_sets(baseline["feature_importance"])
    log.info("")
    log.info("Column sets from baseline importance:")
    for name, cols in sets.items():
        log.info("  %s (%d): %s", name, len(cols), ", ".join(cols))
    dropped_c = [c for c in H1_CLEAN_COLS if c in ROBUST_DEDUPE_DROP]
    log.info("  C drops: %s", ", ".join(dropped_c))

    rows = [baseline]
    for name, cols in sets.items():
        r = run_xgboost_forecast(
            TICKERS,
            "h1_clean",
            evaluate_test=False,
            feature_cols=cols,
            experiment_label=name,
            run_stability=False,
        )
        print_report(r, title="ROBUST | " + name)
        _append_results_tracker(r, notes=notes or ("robustness " + name))
        rows.append(r)

    print_feature_robustness(rows)

    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    out = SAVE_DIR / "xgb_feature_robustness_val.csv"
    pd.DataFrame([
        {
            "experiment": r.get("experiment_label"),
            "n_features": r["n_features"],
            "ic": r["ic_mean"],
            "long_short": r["long_short_mean"],
            "topk_edge": (r.get("topk_summary") or {}).get("mean_edge"),
            "topk_hit_rate": (r.get("topk_summary") or {}).get("hit_rate"),
            "rmse": r["rmse"],
            "features": "|".join(r.get("feature_columns") or []),
        }
        for r in rows
    ]).to_csv(out, index=False)
    log.info("Saved robustness table -> %s", out)
    return rows


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="Manual XGBoost with golden-rule test lock")
    p.add_argument(
        "--final-test",
        action="store_true",
        help="Unlock and score the held-out TEST set once (use only after tuning is finished).",
    )
    p.add_argument(
        "--force-retest",
        action="store_true",
        help="Allow a second TEST unlock (bugfix only; breaks clean golden-rule reporting).",
    )
    p.add_argument(
        "--diagnose-sector",
        action="store_true",
        help=(
            "Run traditional vs h1_clean vs h1_hybrid on VALIDATION only "
            "to measure how much edge comes from sector features."
        ),
    )
    p.add_argument(
        "--compare-sets",
        action="store_true",
        help="Re-run Step-2 style compare on COMPARE_SETS (turns off locked single recipe).",
    )
    p.add_argument(
        "--feature-robustness",
        action="store_true",
        help=(
            "Archive: full h1_clean baseline vs A top-16 / B drop-bottom-9 / C dedupe tech "
            "(validation only)."
        ),
    )
    p.add_argument(
        "--parsimony-a",
        action="store_true",
        help="Run secondary A_top16 (16 features) with frozen hyper-params (parsimony check only).",
    )
    p.add_argument(
        "--no-stability",
        action="store_true",
        help="Skip Step-3 stability diagnostics.",
    )
    p.add_argument(
        "--notes",
        default="",
        help="Short note stored in saved_data/xgb_results_tracker.csv",
    )
    return p.parse_args(argv)


def _run_multi(feature_sets: list[str], evaluate_test: bool, notes: str, diagnose: bool) -> dict:
    results = []
    for fs in feature_sets:
        results.append(run_xgboost_forecast(TICKERS, fs, evaluate_test=evaluate_test))
        print_report(results[-1])
        tag = "sector diagnose " + fs if diagnose else ("compare " + fs)
        _append_results_tracker(results[-1], notes=notes or tag)
    if diagnose:
        best = print_sector_diagnosis(results)
    else:
        best = print_comparison(results)
    log.info("")
    log.info("--- Detail for BEST set ---")
    print_report(best, title="BEST DETAIL | " + best["feature_set"])
    return best


if __name__ == "__main__":
    args = _parse_args()
    _configure_paths()

    evaluate_test = bool(args.final_test or EVALUATE_TEST)
    diagnose = bool(args.diagnose_sector or DIAGNOSE_SECTOR)
    compare = bool(args.compare_sets or COMPARE_FEATURE_SETS)
    robust = bool(args.feature_robustness)
    parsimony_a = bool(args.parsimony_a)
    # Module-level toggles for this process (Play = frozen B_drop_bottom9)
    if args.no_stability:
        globals()["RUN_STABILITY"] = False
    if compare or diagnose:
        globals()["LOCKED_RECIPE"] = False
        globals()["RUN_STABILITY"] = False
        globals()["PARAM_GRID"] = list(STEP2_PARAM_GRID)

    if (diagnose or robust or parsimony_a) and evaluate_test:
        # logging not set up yet — print to stderr
        print(
            "WARNING: This mode is VALIDATION-only. Ignoring --final-test so the golden rule stays intact.",
            file=sys.stderr,
        )
        evaluate_test = False

    if evaluate_test:
        _activate_final_test_output_dir()

    log_path = setup_logging()

    if evaluate_test:
        _assert_test_unlock_allowed(force_retest=args.force_retest)
        log.warning("Running with TEST unlocked (--final-test).")
        log.warning("All final-test artefacts will be saved under: %s", SAVE_DIR)
    else:
        log.info("Running in TUNE mode (validation only). Pass --final-test for the one-shot test.")
        log.info("Validation outputs: SAVE_DIR=%s | LOG_DIR=%s", SAVE_DIR, LOG_DIR)

    if robust:
        run_feature_robustness(notes=args.notes)
    elif diagnose:
        log.info(
            "SECTOR DIAGNOSE: same pipeline on %s (test stays locked).",
            ", ".join(DIAGNOSE_SECTOR_SETS),
        )
        best = _run_multi(DIAGNOSE_SECTOR_SETS, evaluate_test=False, notes=args.notes, diagnose=True)
    elif compare:
        log.info(
            "Step 2 compare replay: %s | PARAM_GRID size=%d.",
            ", ".join(COMPARE_SETS),
            len(PARAM_GRID),
        )
        best = _run_multi(COMPARE_SETS, evaluate_test=evaluate_test, notes=args.notes, diagnose=False)
        if evaluate_test:
            _write_test_lock(best, log_path)
    elif parsimony_a:
        log.info(
            "SECONDARY A_top16 parsimony check | %d features | depth=%s lr=%s mcw=%s trees=%s",
            len(SECONDARY_A_TOP16_COLS),
            LOCKED_PARAMS["max_depth"],
            LOCKED_PARAMS["learning_rate"],
            LOCKED_PARAMS["min_child_weight"],
            LOCKED_N_TREES,
        )
        result = run_xgboost_forecast(
            TICKERS,
            FEATURE_SET,
            evaluate_test=False,
            feature_cols=list(SECONDARY_A_TOP16_COLS),
            experiment_label="A_top16",
            run_stability=False,
        )
        print_report(result, title="SECONDARY | A_top16 (parsimony check)")
        _append_results_tracker(result, notes=args.notes or "secondary A_top16 parsimony")
    else:
        log.info(
            "PRIMARY frozen: %s (%d feats) | depth=%s lr=%s mcw=%s trees=%s | stability=%s",
            EXPERIMENT_LABEL,
            len(LOCKED_FEATURE_COLS),
            LOCKED_PARAMS["max_depth"],
            LOCKED_PARAMS["learning_rate"],
            LOCKED_PARAMS["min_child_weight"],
            LOCKED_N_TREES,
            RUN_STABILITY,
        )
        result = run_xgboost_forecast(
            TICKERS,
            FEATURE_SET,
            evaluate_test=evaluate_test,
            feature_cols=list(LOCKED_FEATURE_COLS),
            experiment_label=EXPERIMENT_LABEL,
        )
        print_report(result, title="PRIMARY | " + EXPERIMENT_LABEL)
        _append_results_tracker(
            result,
            notes=args.notes or (
                "final test unlock B_drop_bottom9"
                if evaluate_test
                else "primary frozen B_drop_bottom9"
            ),
        )
        if evaluate_test:
            _write_test_lock(result, log_path)

    log.info("Done. Full log saved to %s", log_path)

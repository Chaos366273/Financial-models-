"""Fama–French 5-factor: H1 traditional baseline (FTSE members, Use B).

Loads master_panel.parquet (FTSE 100/250). Does not download Bloomberg.

  y = target_excess_return (next-month excess)
  X = MKT_RF, SMB, HML, RMW, CMA at month t
  model = panel OLS + Newey–West HAC (4 lags)
  windows = 10y train → 1y eval (same as XGB/LSTM)

H1 metrics scored every step (same battery as later ML models):
  RMSE, MAE, standard R2, Gu R2_oos, Spearman IC,
  long-short decile spread, long-short Sharpe,
  Top-50 mean return, Top-50 edge, hit rate.

Run: python fama5.py   or   run_fama5.bat
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
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parent
DISS_MODELS = ROOT.parent / "dissertation-models"
FINAL_DATA = ROOT.parent.parent / "data" / "final data for AI"
MASTER_PANEL = FINAL_DATA / "07_panel" / "master_panel.parquet"
OUT_ROOT = ROOT / "fama french"
WF_DIR = OUT_ROOT / "walk_forward_baseline"
CLASSIC_DIR = OUT_ROOT / "classic_appendix"
RESULTS_TRACKER = OUT_ROOT / "ff5_results_tracker.csv"

FACTOR_COLS = ["MKT_RF", "SMB", "HML", "RMW", "CMA"]
TARGET_COL = "target_excess_return"
DESC_COLS = FACTOR_COLS + [TARGET_COL]
FILTER_MEMBERS = True
TOP_K = 50
AIS_2014_TICKER = "AIS.LN"
AIS_2014_MONTH = "2014-04-01"
H1_METRIC_KEYS = [
    "rmse", "mae", "mse", "r2", "r2_oos",
    "ic_mean", "long_short_mean", "long_short_sharpe",
    "topk_avg_return", "topk_edge", "hit_rate",
]

# Same windows as xgb_walk_forward.py
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

XGB_SUMMARY = (
    ROOT / "final test results" / "xgboost 2" / "walk_forward_baseline"
    / "walk_forward_baseline_summary.csv"
)
LSTM_SUMMARY = (
    ROOT / "saved_data" / "lstm_walk_forward" / "walk_forward_baseline" / "summary_metrics.csv"
)

FF5_ZIP_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    "F-F_Research_Data_5_Factors_2x3_CSV.zip"
)
CLASSIC_START = "2000-01-01"
CLASSIC_END = "2025-12-31"

log = logging.getLogger("fama5")


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
    log_path = out_dir / f"ff5_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    logger = logging.getLogger("fama5")
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
    from data.build_panel import build_panel, drop_known_data_errors, KNOWN_BAD_RETURN_ROWS

    pdata, _ = build_panel()
    panel = _filter_members(pdata.panel.copy())
    panel["month"] = pd.to_datetime(panel["month"])
    missing = [c for c in FACTOR_COLS + [TARGET_COL, "Ticker", "month"] if c not in panel.columns]
    if missing:
        raise ValueError(f"Panel missing required columns: {missing}")
    n_before = len(panel)
    panel = drop_known_data_errors(panel)
    n_dropped = n_before - len(panel)
    log.info(
        "Data hygiene: drop %s | extra rows dropped at WF filter=%d",
        list(KNOWN_BAD_RETURN_ROWS),
        n_dropped,
    )
    ais_left = int(
        ((panel["Ticker"].astype(str) == AIS_2014_TICKER)
         & (panel["month"] == pd.Timestamp(AIS_2014_MONTH))).sum()
    )
    if ais_left:
        raise RuntimeError("AIS.LN 2014-04 is still on the panel; RMSE 2014 would remain unusable.")
    log.info("AIS.LN 2014-04 remaining rows=%d (required 0)", ais_left)
    return panel


def _prepare_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Drop rows with missing target/factors; return X, y, meta row subset."""
    cols = FACTOR_COLS + [TARGET_COL, "Ticker", "month"]
    work = df[cols].replace([np.inf, -np.inf], np.nan).dropna(subset=FACTOR_COLS + [TARGET_COL]).copy()
    X = work[FACTOR_COLS].astype(float)
    y = work[TARGET_COL].astype(float)
    return X, y, work


def _fit_predict_step(train: pd.DataFrame, eval_df: pd.DataFrame) -> tuple[pd.DataFrame, dict, object]:
    X_tr, y_tr, _ = _prepare_xy(train)
    X_ev, y_ev, meta = _prepare_xy(eval_df)
    if X_tr.empty or X_ev.empty:
        raise ValueError("Empty train/eval matrix after dropping NaNs")

    X_tr_c = sm.add_constant(X_tr, has_constant="add")
    # Newey-West HAC (4 lags) — standard for monthly finance; point estimates unchanged
    model = sm.OLS(y_tr, X_tr_c).fit(cov_type="HAC", cov_kwds={"maxlags": 4})
    X_ev_c = sm.add_constant(X_ev, has_constant="add")
    X_ev_c = X_ev_c.reindex(columns=X_tr_c.columns, fill_value=0.0)
    pred = model.predict(X_ev_c)

    frame = pd.DataFrame({
        "Ticker": meta["Ticker"].astype(str).values,
        "month": pd.to_datetime(meta["month"]).dt.strftime("%Y-%m-%d").values,
        "y_true": y_ev.values,
        "y_pred": np.asarray(pred, dtype=float),
    })
    for c in FACTOR_COLS:
        frame[c] = meta[c].astype(float).values
    frame["error"] = frame["y_true"] - frame["y_pred"]
    frame["abs_error"] = frame["error"].abs()

    coeffs = {str(k): float(v) for k, v in model.params.items()}
    coeffs_se = {f"se_{k}": float(v) for k, v in model.bse.items()}
    coeffs = {**coeffs, **coeffs_se}
    return frame, coeffs, model


def _blank_h1_metrics() -> dict:
    return {k: np.nan for k in H1_METRIC_KEYS}


def _step_metrics(frame: pd.DataFrame) -> dict:
    from evaluation.predictive import economic_metrics, predictive_metrics, ranking_metrics

    pm = predictive_metrics(frame["y_true"].values, frame["y_pred"].values)
    rm = ranking_metrics(frame)
    em = economic_metrics(frame, k=TOP_K)
    return {
        "mse": pm.get("mse"),
        "mae": pm.get("mae"),
        "rmse": pm.get("rmse"),
        "r2": pm.get("r2"),
        "r2_oos": pm.get("r2_oos"),
        "n_obs": pm.get("n_obs"),
        "ic_mean": rm.get("ic_mean"),
        "ic_ir": rm.get("ic_ir"),
        "long_short_mean": rm.get("long_short_mean"),
        "long_short_sharpe": rm.get("long_short_sharpe"),
        "topk_avg_return": em.get("topk_avg_return"),
        "topk_edge": em.get("topk_edge"),
        "hit_rate": em.get("hit_rate"),
        "n_months": rm.get("n_months") or em.get("n_months"),
        "ranking_identified": bool(rm.get("ranking_identified") or em.get("ranking_identified")),
    }


def pooled_metrics(df: pd.DataFrame) -> dict:
    from evaluation.predictive import economic_metrics, predictive_metrics, ranking_metrics

    if df is None or df.empty:
        out = _blank_h1_metrics()
        out["n_obs"] = 0
        out["ic_ir"] = None
        out["n_months"] = 0
        out["ranking_identified"] = False
        return out
    pm = predictive_metrics(df["y_true"].values, df["y_pred"].values)
    rm = ranking_metrics(df)
    em = economic_metrics(df, k=TOP_K)
    return {
        **pm,
        "ic_mean": rm.get("ic_mean"),
        "ic_ir": rm.get("ic_ir"),
        "long_short_mean": rm.get("long_short_mean"),
        "long_short_sharpe": rm.get("long_short_sharpe"),
        "topk_avg_return": em.get("topk_avg_return"),
        "topk_edge": em.get("topk_edge"),
        "hit_rate": em.get("hit_rate"),
        "n_months": rm.get("n_months") or em.get("n_months"),
        "ranking_identified": bool(rm.get("ranking_identified") or em.get("ranking_identified")),
    }


def write_descriptive_and_pearson(panel: pd.DataFrame) -> tuple[Path, Path]:
    """Analysis-section tables: mean/median/SD/min/max, then Pearson of factors + target."""
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    work = panel[DESC_COLS].replace([np.inf, -np.inf], np.nan).dropna()
    rows = []
    for col in DESC_COLS:
        s = work[col].astype(float)
        rows.append({
            "variable": col,
            "n": int(s.shape[0]),
            "mean": float(s.mean()),
            "median": float(s.median()),
            "std": float(s.std(ddof=1)),
            "min": float(s.min()),
            "max": float(s.max()),
        })
    desc = pd.DataFrame(rows)
    desc_csv = OUT_ROOT / "descriptive_stats.csv"
    desc.to_csv(desc_csv, index=False)

    pearson = work[DESC_COLS].astype(float).corr(method="pearson")
    pear_csv = OUT_ROOT / "pearson_correlation.csv"
    pearson.to_csv(pear_csv)

    desc_md = [
        "# Table 4.1 - Descriptive statistics (FTSE members, complete cases)",
        "",
        "Sample: `is_member` rows with non-missing target and five factors.",
        f"n = {len(work):,} stock-months.",
        "Hygiene: AIS.LN × April 2014 dropped (one stock-month; unadjusted corporate action).",
        "",
        "| Variable | n | Mean | Median | SD | Min | Max |",
        "|----------|--:|-----:|-------:|---:|----:|----:|",
    ]
    for r in rows:
        desc_md.append(
            f"| {r['variable']} | {r['n']} | {_fmt(r['mean'])} | {_fmt(r['median'])} | "
            f"{_fmt(r['std'])} | {_fmt(r['min'])} | {_fmt(r['max'])} |"
        )
    desc_md.extend(["", f"Source: `{desc_csv}`.", ""])
    (OUT_ROOT / "descriptive_stats.md").write_text("\n".join(desc_md), encoding="utf-8")

    pear_md = [
        "# Table 4.2 - Pearson correlation (factors and next-month excess return)",
        "",
        "Pearson r between MKT_RF, SMB, HML, RMW, CMA and `target_excess_return`.",
        "This is the pre-regression association table, not the H1 forecast test.",
        "",
        "|  | " + " | ".join(DESC_COLS) + " |",
        "|--|" + "|".join(["-----:"] * len(DESC_COLS)) + "|",
    ]
    for idx in DESC_COLS:
        cells = " | ".join(_fmt(pearson.loc[idx, c]) for c in DESC_COLS)
        pear_md.append(f"| {idx} | {cells} |")
    pear_md.extend(["", f"Source: `{pear_csv}`.", ""])
    (OUT_ROOT / "pearson_correlation.md").write_text("\n".join(pear_md), encoding="utf-8")
    log.info("Wrote descriptive + Pearson -> %s | %s", desc_csv, pear_csv)
    return desc_csv, pear_csv


def _append_tracker(row: dict) -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    write_header = not RESULTS_TRACKER.is_file()
    with RESULTS_TRACKER.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    log.info("Appended tracker row -> %s", RESULTS_TRACKER)


def run_walk_forward_ff5(panel: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Use B: panel OLS walk-forward. Returns (summary, pooled_preds, pooled_metrics).

    Note: Kenneth French Data Library does not publish a UK-only 5-factor series.
    The Europe factors are the closest available proxy and are standard in
    cross-border research.
    """
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    WF_DIR.mkdir(parents=True, exist_ok=True)
    if panel is None:
        panel = load_panel_members()
    write_descriptive_and_pearson(panel)

    # Note: Kenneth French Data Library does not publish a UK-only 5-factor series.
    # The Europe factors are the closest available proxy and are standard in cross-border research.
    log.info(
        "Panel rows=%d | months %s -> %s | factors=%s | "
        "factor source: Europe FF5 proxy (no UK-only series from Kenneth French) | "
        "hygiene: AIS.LN 2014-04 dropped",
        len(panel),
        panel["month"].min().date(),
        panel["month"].max().date(),
        FACTOR_COLS,
    )

    rows: list[dict] = []
    pooled_parts: list[pd.DataFrame] = []
    coeff_rows: list[dict] = []

    for i, (tr0, tr1, ev_year) in enumerate(WALK_WINDOWS, start=1):
        step_tag = f"step_{i:02d}_{ev_year}"
        step_dir = WF_DIR / step_tag
        step_dir.mkdir(parents=True, exist_ok=True)

        train = _slice_years(panel, tr0, tr1)
        eval_df = _slice_eval_year(panel, ev_year)
        log.info(
            "[%s] train %d-%d (%d rows) -> eval %d (%d rows)",
            step_tag, tr0, tr1, len(train), ev_year, len(eval_df),
        )

        if train.empty or eval_df.empty:
            rows.append({
                "step": i, "eval_year": ev_year, "train_start": tr0, "train_end": tr1,
                "n_train": len(train), "n_eval": 0,
                **_blank_h1_metrics(),
                "ic_ir": np.nan,
                "status": "skipped_empty",
            })
            continue

        try:
            frame, coeffs, _model = _fit_predict_step(train, eval_df)
            metrics = _step_metrics(frame)
        except Exception as exc:
            log.exception("  FAILED: %s", exc)
            rows.append({
                "step": i, "eval_year": ev_year, "train_start": tr0, "train_end": tr1,
                "n_train": len(train), "n_eval": len(eval_df),
                **_blank_h1_metrics(),
                "ic_ir": np.nan,
                "status": f"error: {exc}",
            })
            continue

        frame = frame.copy()
        frame["eval_year"] = ev_year
        pooled_parts.append(frame)

        preview_cols = [
            "Ticker", "month", "y_true", "y_pred", "error", "abs_error",
            *FACTOR_COLS,
        ]
        frame[preview_cols].to_csv(step_dir / "preview.csv", index=False)
        (step_dir / "coefficients.json").write_text(
            json.dumps({
                "formula": "target_excess_return ~ const + MKT_RF + SMB + HML + RMW + CMA",
                "cov_type": "HAC",
                "cov_kwds": {"maxlags": 4},
                "n_train": int(len(train)),
                "n_eval": int(len(frame)),
                "coefficients": coeffs,
            }, indent=2),
            encoding="utf-8",
        )
        coeff_rows.append({
            "step": i,
            "eval_year": ev_year,
            "train_start": tr0,
            "train_end": tr1,
            "cov_type": "HAC",
            "maxlags": 4,
            **coeffs,
        })
        (step_dir / "metrics.json").write_text(
            json.dumps({
                "step": i,
                "eval_year": ev_year,
                "train_start": tr0,
                "train_end": tr1,
                "n_train": int(len(train)),
                "n_eval": int(len(frame)),
                **{k: (None if v is None or (isinstance(v, float) and not np.isfinite(v)) else v)
                   for k, v in metrics.items()},
                "status": "ok",
            }, indent=2),
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
            "r2_oos": metrics.get("r2_oos"),
            "ic_mean": metrics.get("ic_mean"),
            "ic_ir": metrics.get("ic_ir"),
            "long_short_mean": metrics.get("long_short_mean"),
            "long_short_sharpe": metrics.get("long_short_sharpe"),
            "topk_avg_return": metrics.get("topk_avg_return"),
            "topk_edge": metrics.get("topk_edge"),
            "hit_rate": metrics.get("hit_rate"),
            "ranking_identified": metrics.get("ranking_identified"),
            "status": "ok",
        })
        log.info(
            "  RMSE=%.4f MAE=%.4f R2=%.4f R2_oos=%.4f n=%d",
            metrics.get("rmse") or float("nan"),
            metrics.get("mae") or float("nan"),
            metrics.get("r2") or float("nan"),
            metrics.get("r2_oos") or float("nan"),
            metrics.get("n_obs") or 0,
        )

    summary = pd.DataFrame(rows)
    ok = summary[summary["status"] == "ok"]
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
        "r2_oos": float(ok["r2_oos"].mean()) if len(ok) and "r2_oos" in ok.columns else np.nan,
        "ic_mean": float(ok["ic_mean"].mean()) if len(ok) else np.nan,
        "ic_ir": float(ok["ic_ir"].mean()) if len(ok) else np.nan,
        "long_short_mean": float(ok["long_short_mean"].mean()) if len(ok) else np.nan,
        "long_short_sharpe": float(ok["long_short_sharpe"].mean()) if len(ok) and "long_short_sharpe" in ok.columns else np.nan,
        "topk_avg_return": float(ok["topk_avg_return"].mean()) if len(ok) and "topk_avg_return" in ok.columns else np.nan,
        "topk_edge": float(ok["topk_edge"].mean()) if len(ok) and "topk_edge" in ok.columns else np.nan,
        "hit_rate": float(ok["hit_rate"].mean()) if len(ok) and "hit_rate" in ok.columns else np.nan,
        "status": "average",
    }
    summary = pd.concat([summary, pd.DataFrame([avg])], ignore_index=True)
    summary_path = WF_DIR / "walk_forward_baseline_summary.csv"
    summary.to_csv(summary_path, index=False)
    log.info("Saved summary -> %s", summary_path)

    if coeff_rows:
        coeff_df = pd.DataFrame(coeff_rows)
        coeff_path = WF_DIR / "rolling_coefficients.csv"
        coeff_df.to_csv(coeff_path, index=False)
        log.info("Saved rolling coefficients -> %s (%d windows)", coeff_path, len(coeff_df))

    if pooled_parts:
        pooled = pd.concat(pooled_parts, ignore_index=True)
    else:
        pooled = pd.DataFrame(columns=[
            "Ticker", "month", "y_true", "y_pred", "error", "abs_error",
            *FACTOR_COLS, "eval_year",
        ])
    pooled_path = WF_DIR / "pooled_oos_predictions.csv"
    pooled.to_csv(pooled_path, index=False)
    log.info("Saved pooled OOS predictions -> %s (%d rows)", pooled_path, len(pooled))

    pm = pooled_metrics(pooled)
    log.info(
        "POOLED OOS | RMSE=%.4f MAE=%.4f R2=%.4f R2_oos=%.4f n=%s",
        pm.get("rmse") or float("nan"),
        pm.get("mae") or float("nan"),
        pm.get("r2") or float("nan"),
        pm.get("r2_oos") or float("nan"),
        pm.get("n_obs"),
    )

    _append_tracker({
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "mode": "walk_forward",
        "rmse": pm.get("rmse"),
        "mae": pm.get("mae"),
        "mse": pm.get("mse"),
        "r2": pm.get("r2"),
        "r2_oos": pm.get("r2_oos"),
        "ic_mean": pm.get("ic_mean"),
        "long_short_mean": pm.get("long_short_mean"),
        "long_short_sharpe": pm.get("long_short_sharpe"),
        "topk_avg_return": pm.get("topk_avg_return"),
        "topk_edge": pm.get("topk_edge"),
        "hit_rate": pm.get("hit_rate"),
        "n_obs": pm.get("n_obs"),
        "n_steps_ok": int((summary["status"] == "ok").sum()),
        "output_dir": str(WF_DIR),
    })

    return summary, pooled, pm


# ---------------------------------------------------------------------------
# Classic Use A (Kenneth French + Yahoo) — appendix only
# ---------------------------------------------------------------------------

def load_french_ff5(start: str, end: str) -> pd.DataFrame:
    """Load monthly Mkt-RF, SMB, HML, RMW, CMA, RF from Kenneth French Data Library."""
    import urllib.request

    with urllib.request.urlopen(FF5_ZIP_URL) as r:
        zbytes = r.read()
    with ZipFile(BytesIO(zbytes)) as zf:
        csv_name = next(n for n in zf.namelist() if n.lower().endswith(".csv"))
        raw = zf.read(csv_name).decode("utf-8", errors="replace")

    lines = raw.splitlines()
    rows = []
    for line in lines:
        parts = [p.strip() for p in line.split(",") if p.strip()]
        if len(parts) < 7 or not parts[0].isdigit() or len(parts[0]) != 6:
            continue
        ym = int(parts[0])
        y, m = ym // 100, ym % 100
        rows.append({
            "Date": pd.Timestamp(year=y, month=m, day=1),
            "Mkt-RF": float(parts[1]) / 100.0,
            "SMB": float(parts[2]) / 100.0,
            "HML": float(parts[3]) / 100.0,
            "RMW": float(parts[4]) / 100.0,
            "CMA": float(parts[5]) / 100.0,
            "RF": float(parts[6]) / 100.0,
        })

    ff5 = pd.DataFrame(rows).set_index("Date").sort_index()
    mask = (ff5.index >= pd.Timestamp(start)) & (ff5.index <= pd.Timestamp(end))
    return ff5.loc[mask]


def load_monthly_returns(ticker: str, start: str, end: str) -> pd.Series:
    """Monthly total returns from Yahoo Finance (adj. close)."""
    import yfinance as yf

    px = yf.download(
        ticker, start=start, end=end, interval="1mo",
        auto_adjust=True, progress=False, group_by="column",
    )
    if isinstance(px.columns, pd.MultiIndex):
        px.columns = px.columns.get_level_values(0)
    close = px["Close"].astype(float)
    rets = close.pct_change().dropna()
    rets.index = rets.index.to_period("M").to_timestamp()
    return rets.rename("Return")


def run_classic_ff5(
    ticker: str = "AAPL",
    start: str = CLASSIC_START,
    end: str = CLASSIC_END,
) -> object:
    """Use A: contemporaneous R_t - RF_t on French factors (appendix)."""
    CLASSIC_DIR.mkdir(parents=True, exist_ok=True)
    ff5 = load_french_ff5(start, end)
    stock_ret = load_monthly_returns(ticker, start, end)
    data = pd.merge(stock_ret.to_frame(), ff5, left_index=True, right_index=True, how="inner")
    if data.empty:
        raise ValueError(f"No overlapping months between {ticker} and FF5 factors.")

    data["Excess_Return"] = data["Return"] - data["RF"]
    y = data["Excess_Return"]
    X = sm.add_constant(data[["Mkt-RF", "SMB", "HML", "RMW", "CMA"]])
    model = sm.OLS(y, X).fit()

    summary_txt = [
        f"Classic FF5 (Use A) — {ticker}",
        f"Source: Kenneth French Data Library 5 Factors (2x3)",
        f"Period: {start} -> {end}",
        f"Observations: {int(model.nobs)}",
        "",
        str(model.summary()),
        "",
        f"Alpha (monthly)     : {model.params['const']:.4%}",
        f"Alpha (annualised)  : {model.params['const'] * 12:.2%}",
        f"Market beta         : {model.params['Mkt-RF']:.3f}",
        f"SMB loading         : {model.params['SMB']:.3f}",
        f"HML loading         : {model.params['HML']:.3f}",
        f"RMW loading         : {model.params['RMW']:.3f}",
        f"CMA loading         : {model.params['CMA']:.3f}",
        f"R-squared           : {model.rsquared:.3f}",
        "",
        "NOTE: Contemporaneous regression (explains same-month returns).",
        "Not used for H1 predictive comparison against XGB/LSTM.",
    ]
    out_txt = CLASSIC_DIR / f"classic_{ticker}_summary.txt"
    out_txt.write_text("\n".join(summary_txt), encoding="utf-8")
    coeffs = {str(k): float(v) for k, v in model.params.items()}
    (CLASSIC_DIR / f"classic_{ticker}_coefficients.json").write_text(
        json.dumps({"ticker": ticker, "coefficients": coeffs, "rsquared": float(model.rsquared),
                    "nobs": int(model.nobs)}, indent=2),
        encoding="utf-8",
    )
    log.info("Classic appendix saved -> %s", CLASSIC_DIR)
    print("\n".join(summary_txt[-12:]))
    return model


# ---------------------------------------------------------------------------
# Dissertation H1 tables
# ---------------------------------------------------------------------------

LS_TOP_FRAC = 0.10


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
    if ok.empty or "n_eval_rows" not in ok.columns:
        return None
    return int(pd.to_numeric(ok["n_eval_rows"], errors="coerce").fillna(0).sum())


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
    """Normalise eval_year labels so 2012 / 2012.0 / '2012' match."""
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return ""
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    s = str(x).strip().upper()
    if s in {"AVERAGE", "AVG", "NAN", ""}:
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


def _rmse_arr(y_true, y_pred) -> float | None:
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    m = np.isfinite(yt) & np.isfinite(yp)
    if not m.any():
        return None
    return float(np.sqrt(np.mean((yt[m] - yp[m]) ** 2)))


def _load_mcap_lookup(keys: pd.DataFrame) -> pd.DataFrame:
    """Attach market_cap for Ticker/month keys (parquet columns only)."""
    need = keys[["Ticker", "month"]].drop_duplicates().copy()
    need["Ticker"] = need["Ticker"].astype(str)
    need["month"] = pd.to_datetime(need["month"])
    cols = ["Ticker", "month", "market_cap", "cur_mkt_cap", "log_market_cap"]
    panel = pd.read_parquet(MASTER_PANEL, columns=cols)
    panel["Ticker"] = panel["Ticker"].astype(str)
    panel["month"] = pd.to_datetime(panel["month"])
    mcap = panel["market_cap"].astype(float)
    mcap = mcap.fillna(panel["cur_mkt_cap"].astype(float))
    logc = panel["log_market_cap"].astype(float)
    mcap = mcap.fillna(np.exp(logc.clip(lower=-50, upper=50)))
    panel = panel.assign(mcap=mcap)[["Ticker", "month", "mcap"]]
    out = need.merge(panel, on=["Ticker", "month"], how="left")
    return out


def _decile_long_short(
    frame: pd.DataFrame,
    *,
    weight_col: str | None = None,
    top_frac: float = LS_TOP_FRAC,
    min_names: int = 10,
) -> dict:
    """Monthly top/bottom frac long-short; EW or value-weighted within legs."""
    work = frame[["month", "y_true", "y_pred"] + ([weight_col] if weight_col else [])].copy()
    work["month"] = pd.to_datetime(work["month"])
    work = work.replace([np.inf, -np.inf], np.nan).dropna(subset=["y_true", "y_pred"])
    spreads: list[float] = []
    for _, g in work.groupby("month", sort=True):
        if len(g) < min_names:
            continue
        # Pure FF5 factors are identical across names in a month → no cross-section rank
        if float(np.nanstd(g["y_pred"].astype(float))) < 1e-12:
            continue
        k = max(1, int(np.floor(len(g) * top_frac)))
        g = g.sort_values("y_pred")
        bottom = g.head(k)
        top = g.tail(k)
        if weight_col and weight_col in g.columns:
            def _wmean(leg: pd.DataFrame) -> float | None:
                w = leg[weight_col].astype(float).clip(lower=0)
                y = leg["y_true"].astype(float)
                m = np.isfinite(w) & np.isfinite(y) & (w > 0)
                if not m.any():
                    return None
                return float(np.average(y[m], weights=w[m]))

            t = _wmean(top)
            b = _wmean(bottom)
            if t is None or b is None:
                continue
            spreads.append(t - b)
        else:
            spreads.append(float(top["y_true"].mean() - bottom["y_true"].mean()))
    if not spreads:
        return {"long_short_mean": None, "long_short_std": None, "n_months": 0}
    return {
        "long_short_mean": float(np.mean(spreads)),
        "long_short_std": float(np.std(spreads, ddof=1)) if len(spreads) > 1 else 0.0,
        "n_months": int(len(spreads)),
    }


def _load_lstm_pooled_preview() -> pd.DataFrame:
    parts = []
    base = LSTM_SUMMARY.parent
    for p in sorted(base.glob("step_*/preview.csv")):
        d = pd.read_csv(p)
        d["month"] = pd.to_datetime(d["month"])
        parts.append(d)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def _ais_sensitivity(pooled: pd.DataFrame) -> dict:
    """Confirm AIS.LN April 2014 is off the H1 panel (hygiene drop)."""
    if pooled is None or pooled.empty:
        return {}
    work = pooled.copy()
    work["month"] = pd.to_datetime(work["month"])
    work["Ticker"] = work["Ticker"].astype(str)
    ais_mask = (work["Ticker"] == AIS_2014_TICKER) & (
        work["month"] == pd.Timestamp(AIS_2014_MONTH)
    )
    y14 = work[work["month"].dt.year == 2014]
    return {
        "ais_ticker": AIS_2014_TICKER,
        "ais_month": AIS_2014_MONTH,
        "dropped_for_hygiene": True,
        "ais_n": int(ais_mask.sum()),
        "n_2014": int(len(y14)),
        "ais_y_true": None,
        "rmse_full": _rmse_arr(work["y_true"], work["y_pred"]),
        "rmse_2014": _rmse_arr(y14["y_true"], y14["y_pred"]) if len(y14) else None,
        "note": (
            "One observation (AIS.LN, April 2014) showed a +21,800% excess return "
            "from an unadjusted corporate action and was excluded from the panel."
        ),
    }


def write_long_short_tables(
    ff5_pooled: pd.DataFrame | None,
    lstm_preview: pd.DataFrame | None,
    xgb_ew_from_summary: float | None,
) -> tuple[Path, Path]:
    """EW + VW top/bottom decile long-short for models with full OOS preds."""
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    rows = []

    def _attach_mcap(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame()
        out = df.copy()
        out["Ticker"] = out["Ticker"].astype(str)
        out["month"] = pd.to_datetime(out["month"])
        mcap = _load_mcap_lookup(out)
        return out.merge(mcap, on=["Ticker", "month"], how="left")

    if ff5_pooled is not None and not ff5_pooled.empty:
        ff = _attach_mcap(ff5_pooled)
        ew = _decile_long_short(ff, weight_col=None)
        vw = _decile_long_short(ff, weight_col="mcap")
        rows.append({
            "Model": "Fama-French 5 (panel OLS)",
            "construction": f"top/bottom {int(LS_TOP_FRAC*100)}% by y_pred each month",
            "EW_long_short": ew["long_short_mean"],
            "EW_std": ew["long_short_std"],
            "VW_long_short": vw["long_short_mean"],
            "VW_std": vw["long_short_std"],
            "n_months_EW": ew["n_months"],
            "n_months_VW": vw["n_months"],
            "n_obs": int(len(ff)),
            "note": (
                "n/a for ranking: market-wide factors => constant cross-section y_pred; "
                "months without pred dispersion skipped"
            ),
        })

    if lstm_preview is not None and not lstm_preview.empty:
        lp = _attach_mcap(lstm_preview)
        ew = _decile_long_short(lp, weight_col=None)
        vw = _decile_long_short(lp, weight_col="mcap")
        rows.append({
            "Model": "LSTM",
            "construction": f"top/bottom {int(LS_TOP_FRAC*100)}% by y_pred each month",
            "EW_long_short": ew["long_short_mean"],
            "EW_std": ew["long_short_std"],
            "VW_long_short": vw["long_short_mean"],
            "VW_std": vw["long_short_std"],
            "n_months_EW": ew["n_months"],
            "n_months_VW": vw["n_months"],
            "n_obs": int(len(lp)),
            "note": "scored on LSTM preview stock-months",
        })

    rows.append({
        "Model": "XGBoost (B_drop_bottom9)",
        "construction": f"top/bottom {int(LS_TOP_FRAC*100)}% by y_pred each month",
        "EW_long_short": xgb_ew_from_summary,
        "EW_std": None,
        "VW_long_short": None,
        "VW_std": None,
        "n_months_EW": None,
        "n_months_VW": None,
        "n_obs": None,
        "note": "EW from walk-forward summary AVERAGE; VW N/A (full preview.csv not saved)",
    })

    ls_csv = OUT_ROOT / "h1_long_short_decile.csv"
    pd.DataFrame(rows).to_csv(ls_csv, index=False)

    md = [
        "# Table 4.z - Decile long-short (economic magnitude)",
        "",
        f"Each month: long top {int(LS_TOP_FRAC*100)}% / short bottom {int(LS_TOP_FRAC*100)}% by predicted next-month excess return.",
        "EW = equal-weight within each leg. VW = market-cap weights within each leg.",
        "Reported value = mean monthly long-short spread (excess return units).",
        "",
        "| Model | EW long-short | EW std | VW long-short | VW std | n_months (EW) | note |",
        "|-------|---------------|--------|---------------|--------|---------------|------|",
    ]
    for r in rows:
        md.append(
            f"| {r['Model']} | {_fmt(r['EW_long_short'])} | {_fmt(r['EW_std'])} | "
            f"{_fmt(r['VW_long_short'])} | {_fmt(r['VW_std'])} | "
            f"{r['n_months_EW'] if r['n_months_EW'] is not None else '-'} | {r['note']} |"
        )
    md.append("")
    ls_md = OUT_ROOT / "h1_long_short_decile.md"
    ls_md.write_text("\n".join(md), encoding="utf-8")
    log.info("Wrote long-short tables -> %s | %s", ls_md, ls_csv)
    return ls_md, ls_csv


def write_h1_comparison(
    ff5_pooled_metrics: dict | None = None,
    ff5_summary: pd.DataFrame | None = None,
) -> tuple[Path, Path]:
    """Merge FF5 + XGB + LSTM summaries into dissertation markdown + CSV."""
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    pooled_path = WF_DIR / "pooled_oos_predictions.csv"
    pooled = pd.read_csv(pooled_path) if pooled_path.is_file() else pd.DataFrame()
    if ff5_pooled_metrics is None and not pooled.empty:
        ff5_pooled_metrics = pooled_metrics(pooled)
    if ff5_summary is None:
        sp = WF_DIR / "walk_forward_baseline_summary.csv"
        ff5_summary = pd.read_csv(sp) if sp.is_file() else pd.DataFrame()

    xgb = pd.read_csv(XGB_SUMMARY) if XGB_SUMMARY.is_file() else pd.DataFrame()
    lstm = pd.read_csv(LSTM_SUMMARY) if LSTM_SUMMARY.is_file() else pd.DataFrame()
    xgb_avg = _avg_row(xgb)
    lstm_avg = _avg_row(lstm)
    xgb_ok = _ok_steps(xgb)
    lstm_ok = _ok_steps(lstm)
    ff5_ok = _ok_steps(ff5_summary)

    ff5_step_rmse = _safe_float(ff5_ok["rmse"].mean()) if len(ff5_ok) and "rmse" in ff5_ok.columns else None
    ff5_step_rmse_med = _safe_float(ff5_ok["rmse"].median()) if len(ff5_ok) and "rmse" in ff5_ok.columns else None
    lstm_step_rmse = _safe_float(lstm_ok["rmse"].mean()) if len(lstm_ok) and "rmse" in lstm_ok.columns else None
    lstm_step_rmse_med = _safe_float(lstm_ok["rmse"].median()) if len(lstm_ok) and "rmse" in lstm_ok.columns else None
    lstm_step_r2 = _safe_float(lstm_ok["r2"].mean()) if len(lstm_ok) and "r2" in lstm_ok.columns else None

    n_ff5 = int(len(pooled)) if not pooled.empty else (
        int(ff5_pooled_metrics.get("n_obs") or 0) if ff5_pooled_metrics else None
    )
    n_xgb = _sum_n_eval(xgb)
    n_lstm = _sum_n_eval(lstm)

    sens = _ais_sensitivity(pooled) if not pooled.empty else {}

    rows = [
        {
            "Model": "Fama-French 5 (panel OLS)",
            "Features": "5 factors only",
            "RMSE_mean_year": ff5_step_rmse,
            "RMSE_median_year": ff5_step_rmse_med,
            "RMSE_pooled": ff5_pooled_metrics.get("rmse") if ff5_pooled_metrics else None,
            "RMSE_pooled_ex_AIS2014": sens.get("rmse_ex_ais"),
            "MAE": ff5_pooled_metrics.get("mae") if ff5_pooled_metrics else None,
            "MSE": ff5_pooled_metrics.get("mse") if ff5_pooled_metrics else None,
            "R2": ff5_pooled_metrics.get("r2") if ff5_pooled_metrics else None,
            "R2_oos": ff5_pooled_metrics.get("r2_oos") if ff5_pooled_metrics else None,
            "IC_Spearman": ff5_pooled_metrics.get("ic_mean") if ff5_pooled_metrics else None,
            "Long_short_EW": ff5_pooled_metrics.get("long_short_mean") if ff5_pooled_metrics else None,
            "Long_short_Sharpe": ff5_pooled_metrics.get("long_short_sharpe") if ff5_pooled_metrics else None,
            "Top50_mean": ff5_pooled_metrics.get("topk_avg_return") if ff5_pooled_metrics else None,
            "Top50_edge": ff5_pooled_metrics.get("topk_edge") if ff5_pooled_metrics else None,
            "Hit_rate": ff5_pooled_metrics.get("hit_rate") if ff5_pooled_metrics else None,
            "n_obs": n_ff5,
        },
        {
            "Model": "XGBoost (B_drop_bottom9)",
            "Features": "19 features",
            "RMSE_mean_year": None,
            "RMSE_median_year": None,
            "RMSE_pooled": None,
            "RMSE_pooled_ex_AIS2014": None,
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
            "RMSE_pooled_ex_AIS2014": None,
            "MAE": None,
            "MSE": None,
            "R2": lstm_step_r2 if lstm_step_r2 is not None else (
                _safe_float(lstm_avg.get("r2")) if lstm_avg is not None else None
            ),
            "IC_Spearman": _safe_float(lstm_avg.get("ic")) if lstm_avg is not None else None,
            "Long_short_EW": _safe_float(lstm_avg.get("long_short")) if lstm_avg is not None else None,
            "n_obs": n_lstm,
        },
    ]

    cmp_csv = OUT_ROOT / "h1_comparison_template.csv"
    pd.DataFrame(rows).to_csv(cmp_csv, index=False)

    # Year-by-year RMSE table
    year_rows = []
    years = [ev for (_, _, ev) in WALK_WINDOWS]
    if not ff5_ok.empty and "eval_year" in ff5_ok.columns:
        ff5_ok = ff5_ok.copy()
        ff5_ok["_yk"] = ff5_ok["eval_year"].map(_year_key)
    if not xgb_ok.empty and "eval_year" in xgb_ok.columns:
        xgb_ok = xgb_ok.copy()
        xgb_ok["_yk"] = xgb_ok["eval_year"].map(_year_key)
    if not lstm_ok.empty and "eval_year" in lstm_ok.columns:
        lstm_ok = lstm_ok.copy()
        lstm_ok["_yk"] = lstm_ok["eval_year"].map(_year_key)

    for yr in years:
        yk = _year_key(yr)
        ff_r = ff5_ok.loc[ff5_ok["_yk"] == yk] if not ff5_ok.empty and "_yk" in ff5_ok.columns else pd.DataFrame()
        ff_rmse = _safe_float(ff_r.iloc[0]["rmse"]) if len(ff_r) and "rmse" in ff_r.columns else None
        # Within-month FF5 predictions are constant → IC not cross-sectionally identified
        ff_ic = None
        if len(ff_r):
            if "n_eval" in ff_r.columns:
                n_ff = int(ff_r.iloc[0]["n_eval"])
            elif "n_obs" in ff_r.columns:
                n_ff = int(ff_r.iloc[0]["n_obs"])
            else:
                n_ff = None
        else:
            n_ff = None
        xgb_r = xgb_ok.loc[xgb_ok["_yk"] == yk] if not xgb_ok.empty and "_yk" in xgb_ok.columns else pd.DataFrame()
        xgb_ic = _safe_float(xgb_r.iloc[0]["ic"]) if len(xgb_r) and "ic" in xgb_r.columns else None
        n_xg = int(xgb_r.iloc[0]["n_eval_rows"]) if len(xgb_r) and "n_eval_rows" in xgb_r.columns else None
        lstm_r = lstm_ok.loc[lstm_ok["_yk"] == yk] if not lstm_ok.empty and "_yk" in lstm_ok.columns else pd.DataFrame()
        lstm_rmse = _safe_float(lstm_r.iloc[0]["rmse"]) if len(lstm_r) and "rmse" in lstm_r.columns else None
        lstm_ic = _safe_float(lstm_r.iloc[0]["ic"]) if len(lstm_r) and "ic" in lstm_r.columns else None
        n_ls = int(lstm_r.iloc[0]["n_eval_rows"]) if len(lstm_r) and "n_eval_rows" in lstm_r.columns else None
        year_rows.append({
            "eval_year": yr,
            "FF5_RMSE": ff_rmse,
            "LSTM_RMSE": lstm_rmse,
            "FF5_IC": ff_ic,
            "XGB_IC": xgb_ic,
            "LSTM_IC": lstm_ic,
            "FF5_n": n_ff,
            "XGB_n": n_xg,
            "LSTM_n": n_ls,
        })

    year_df = pd.DataFrame(year_rows)
    if len(year_df):
        mean_row = {
            "eval_year": "Mean",
            "FF5_RMSE": year_df["FF5_RMSE"].mean(skipna=True),
            "LSTM_RMSE": year_df["LSTM_RMSE"].mean(skipna=True),
            "FF5_IC": year_df["FF5_IC"].mean(skipna=True),
            "XGB_IC": year_df["XGB_IC"].mean(skipna=True),
            "LSTM_IC": year_df["LSTM_IC"].mean(skipna=True),
            "FF5_n": float(year_df["FF5_n"].sum(skipna=True)),
            "XGB_n": float(year_df["XGB_n"].sum(skipna=True)),
            "LSTM_n": float(year_df["LSTM_n"].sum(skipna=True)),
        }
        med_row = {
            "eval_year": "Median",
            "FF5_RMSE": year_df["FF5_RMSE"].median(skipna=True),
            "LSTM_RMSE": year_df["LSTM_RMSE"].median(skipna=True),
            "FF5_IC": year_df["FF5_IC"].median(skipna=True),
            "XGB_IC": year_df["XGB_IC"].median(skipna=True),
            "LSTM_IC": year_df["LSTM_IC"].median(skipna=True),
            "FF5_n": np.nan,
            "XGB_n": np.nan,
            "LSTM_n": np.nan,
        }
        year_df = pd.concat([year_df, pd.DataFrame([mean_row, med_row])], ignore_index=True)
    year_csv = OUT_ROOT / "h1_rmse_by_year.csv"
    year_df.to_csv(year_csv, index=False)

    lstm_preview = _load_lstm_pooled_preview()
    xgb_ew = _safe_float(xgb_avg.get("long_short")) if xgb_avg is not None else None
    ls_md, ls_csv = write_long_short_tables(
        pooled if not pooled.empty else None,
        lstm_preview if not lstm_preview.empty else None,
        xgb_ew,
    )

    ff = rows[0]
    xg = rows[1]
    ls = rows[2]
    mae_mean = _safe_float(ff5_ok["mae"].mean()) if len(ff5_ok) and "mae" in ff5_ok.columns else None
    mae_med = _safe_float(ff5_ok["mae"].median()) if len(ff5_ok) and "mae" in ff5_ok.columns else None
    r2_mean = _safe_float(ff5_ok["r2"].mean()) if len(ff5_ok) and "r2" in ff5_ok.columns else None
    r2_med = _safe_float(ff5_ok["r2"].median()) if len(ff5_ok) and "r2" in ff5_ok.columns else None
    r2oos_mean = _safe_float(ff5_ok["r2_oos"].mean()) if len(ff5_ok) and "r2_oos" in ff5_ok.columns else None
    r2oos_med = _safe_float(ff5_ok["r2_oos"].median()) if len(ff5_ok) and "r2_oos" in ff5_ok.columns else None
    md_lines = [
        "# Table 4.x - FF5 H1 metric scorecard (walk-forward 2012-2025)",
        "",
        "Protocol: 10-year train -> 1-year eval; FTSE members (`is_member`); master panel.",
        "Model: panel OLS of `target_excess_return` on MKT_RF, SMB, HML, RMW, CMA.",
        "This is the traditional **baseline**. XGB/LSTM use the same windows, target, and metric list.",
        "",
        "| Metric | Pooled OOS | Mean across years | Median across years |",
        "|--------|-----------:|------------------:|--------------------:|",
        f"| RMSE | {_fmt(ff['RMSE_pooled'])} | {_fmt(ff['RMSE_mean_year'])} | {_fmt(ff['RMSE_median_year'])} |",
        f"| MAE | {_fmt(ff['MAE'])} | {_fmt(mae_mean)} | {_fmt(mae_med)} |",
        f"| R² (standard) | {_fmt(ff['R2'])} | {_fmt(r2_mean)} | {_fmt(r2_med)} |",
        f"| Gu R²_oos | {_fmt(ff['R2_oos'])} | {_fmt(r2oos_mean)} | {_fmt(r2oos_med)} |",
        f"| Spearman IC | {_fmt(ff['IC_Spearman'])} | - | - |",
        f"| Long-short decile | {_fmt(ff['Long_short_EW'])} | - | - |",
        f"| Long-short Sharpe | {_fmt(ff.get('Long_short_Sharpe'))} | - | - |",
        f"| Top-50 mean return | {_fmt(ff.get('Top50_mean'))} | - | - |",
        f"| Top-50 edge | {_fmt(ff.get('Top50_edge'))} | - | - |",
        f"| Hit rate | {_fmt(ff.get('Hit_rate'))} | - | - |",
        f"| n_obs | {ff['n_obs'] if ff['n_obs'] is not None else '-'} | - | - |",
        "",
        "Ranking metrics (IC, long-short, Top-50, hit rate) are blank when every stock in a month "
        "gets the same predicted return (market-wide factors). That is expected for this FF5 spec, not a crash.",
        "Descriptive stats and Pearson: `descriptive_stats.md`, `pearson_correlation.md`.",
        "OLS betas (each train window): `walk_forward_baseline/rolling_coefficients.csv`.",
        "",
        "# Table 4.x-b - Out-of-sample comparison stub (FF5 vs later models)",
        "",
        "| Model | Features | RMSE (mean yr) | RMSE (median yr) | RMSE pooled | IC (Spearman) | Long-short EW | n_obs |",
        "|-------|----------|----------------|------------------|-------------|---------------|---------------|-------|",
        (
            f"| {ff['Model']} | {ff['Features']} | {_fmt(ff['RMSE_mean_year'])} | {_fmt(ff['RMSE_median_year'])} | "
            f"{_fmt(ff['RMSE_pooled'])} | {_fmt(ff['IC_Spearman'])} | {_fmt(ff['Long_short_EW'])} | "
            f"{ff['n_obs'] if ff['n_obs'] is not None else '-'} |"
        ),
        (
            f"| {xg['Model']} | {xg['Features']} | {_fmt(xg['RMSE_mean_year'])} | {_fmt(xg['RMSE_median_year'])} | "
            f"{_fmt(xg['RMSE_pooled'])} | {_fmt(xg['IC_Spearman'])} | {_fmt(xg['Long_short_EW'])} | "
            f"{xg['n_obs'] if xg['n_obs'] is not None else '-'} |"
        ),
        (
            f"| {ls['Model']} | {ls['Features']} | {_fmt(ls['RMSE_mean_year'])} | {_fmt(ls['RMSE_median_year'])} | "
            f"{_fmt(ls['RMSE_pooled'])} | {_fmt(ls['IC_Spearman'])} | {_fmt(ls['Long_short_EW'])} | "
            f"{ls['n_obs'] if ls['n_obs'] is not None else '-'} |"
        ),
        "",
        "Notes:",
        "- XGBoost walk-forward summary stores IC / long-short (not RMSE).",
        "- FF5 and XGB share the same hygiene drop: AIS.LN × April 2014 (one stock-month).",
        "- **FF5 IC / cross-sectional long-short:** not identified under Use B with market-wide factors only "
        "(all stocks in a month receive the same `y_pred`). Compare FF5 on RMSE; use XGB/LSTM for ranking metrics.",
        (
            "- **Data hygiene:** one observation (AIS.LN, April 2014) showed a +21,800% excess return "
            "from an unadjusted corporate action and was excluded from the panel. "
            f"AIS remaining on OOS pooled = {sens.get('ais_n')}. "
            f"2014 eval n = {sens.get('n_2014')}. "
            f"FF5 RMSE 2014 = {_fmt(sens.get('rmse_2014'))}; "
            f"pooled RMSE = {_fmt(sens.get('rmse_full'))}."
        ),
        "- Prefer **median year RMSE** as well as pooled RMSE (no AIS spike after the drop).",
        "",
        "# Table 4.y - Walk-forward metrics by evaluation year",
        "",
        "| Eval year | FF5 RMSE | LSTM RMSE | FF5 IC | XGB IC | LSTM IC | FF5 n | XGB n | LSTM n |",
        "|-----------|----------|-----------|--------|--------|---------|-------|-------|--------|",
    ]
    for _, r in year_df.iterrows():
        md_lines.append(
            f"| {r['eval_year']} | {_fmt(r['FF5_RMSE'])} | {_fmt(r['LSTM_RMSE'])} | "
            f"{_fmt(r['FF5_IC'])} | {_fmt(r['XGB_IC'])} | {_fmt(r['LSTM_IC'])} | "
            f"{'-' if pd.isna(r.get('FF5_n')) else int(r['FF5_n']) if r.get('FF5_n') is not None else '-'} | "
            f"{'-' if pd.isna(r.get('XGB_n')) else int(r['XGB_n']) if r.get('XGB_n') is not None else '-'} | "
            f"{'-' if pd.isna(r.get('LSTM_n')) else int(r['LSTM_n']) if r.get('LSTM_n') is not None else '-'} |"
        )
    md_lines.extend([
        "",
        f"See also decile long-short (EW/VW): `{ls_md.name}` / `{ls_csv.name}`.",
        "",
        "## Sources",
        f"- FF5 pooled: `{pooled_path}`",
        f"- FF5 summary: `{WF_DIR / 'walk_forward_baseline_summary.csv'}`",
        f"- XGB summary: `{XGB_SUMMARY}`",
        f"- LSTM summary: `{LSTM_SUMMARY}`",
        f"- Long-short: `{ls_csv}`",
        f"- Descriptive stats: `{OUT_ROOT / 'descriptive_stats.csv'}`",
        f"- Pearson: `{OUT_ROOT / 'pearson_correlation.csv'}`",
        f"- Rolling OLS betas: `{WF_DIR / 'rolling_coefficients.csv'}`",
        f"- 2014 diagnostics: `{OUT_ROOT / 'diagnostics'}`",
        "",
    ])
    md_path = OUT_ROOT / "dissertation_table_h1.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    # Keep sensitivity sidecar for thesis appendix
    if sens:
        sens_path = OUT_ROOT / "h1_2014_ais_sensitivity.json"
        sens_path.write_text(json.dumps(sens, indent=2), encoding="utf-8")
    log.info("Wrote H1 tables -> %s | %s | %s | %s", md_path, cmp_csv, year_csv, ls_csv)
    return md_path, cmp_csv



def write_readme(log_path: Path | None = None) -> Path:
    path = OUT_ROOT / "README_ff5_walk_forward.txt"
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join([
            "Fama–French 5 — predictive walk-forward (Use B)",
            "==============================================",
            f"Created: {datetime.now().isoformat(timespec='seconds')}",
            f"Log: {log_path or '(see ff5_run_*.txt)'}",
            "",
            "Primary protocol (H1):",
            "  target = target_excess_return (next-month excess)",
            "  features = MKT_RF, SMB, HML, RMW, CMA (month t)",
            "  model = panel OLS (statsmodels)",
            "  windows = same as xgb_walk_forward (10y train / 1y eval)",
            "  universe = FTSE members (is_member) from master_panel.parquet",
            "  hygiene = drop AIS.LN × 2014-04 (one stock-month; unadjusted corporate action)",
            "",
            "Commands:",
            "  python fama5.py                  # walk-forward + analysis tables",
            "  python fama5.py --walk-forward   # same",
            "  python fama5.py --build-h1-table # rebuild tables from saved CSVs",
            "",
            "Do not use --classic (Yahoo single-ticker). H1 uses FTSE master_panel only.",
            "",
            "H1 metrics: RMSE, MAE, R2, Gu R2_oos, Spearman IC, LS decile, LS Sharpe,",
            "            Top-50 mean, Top-50 edge, hit rate.",
            "",
            "Outputs:",
            "  descriptive_stats.csv / .md      # mean, median, SD, min, max",
            "  pearson_correlation.csv / .md    # Pearson of factors + target",
            "  walk_forward_baseline/",
            "    walk_forward_baseline_summary.csv",
            "    pooled_oos_predictions.csv",
            "    rolling_coefficients.csv      # OLS betas + HAC SEs per train window",
            "    step_XX_YYYY/preview.csv | metrics.json | coefficients.json",
            "  dissertation_table_h1.md",
            "  h1_comparison_template.csv",
            "  h1_rmse_by_year.csv",
            "  h1_long_short_decile.md/.csv",
            "  ff5_results_tracker.csv",
            "",
            f"All FF5 artefacts root: {OUT_ROOT}",
            "",
        ]),
        encoding="utf-8",
    )
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="FF5 Use B walk-forward + classic appendix")
    parser.add_argument("--walk-forward", action="store_true", help="Run predictive panel OLS (default)")
    parser.add_argument("--classic", action="store_true", help="Use A: Kenneth French + Yahoo single ticker")
    parser.add_argument("--ticker", default="AAPL", help="Ticker for --classic")
    parser.add_argument("--build-h1-table", action="store_true", help="Rebuild dissertation H1 tables only")
    parser.add_argument("--start", default=CLASSIC_START)
    parser.add_argument("--end", default=CLASSIC_END)
    args = parser.parse_args(argv)

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    log_path = setup_logging(OUT_ROOT)
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    if args.classic:
        run_classic_ff5(ticker=args.ticker, start=args.start, end=args.end)
        write_readme(log_path)
        return 0

    if args.build_h1_table and not args.walk_forward:
        _configure_paths()
        write_h1_comparison()
        write_readme(log_path)
        return 0

    # Default: walk-forward Use B
    _configure_paths()
    summary, pooled, pm = run_walk_forward_ff5()
    write_h1_comparison(ff5_pooled_metrics=pm, ff5_summary=summary)
    write_readme(log_path)
    log.info("Done. Outputs under %s", OUT_ROOT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

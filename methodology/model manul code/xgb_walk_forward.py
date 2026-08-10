"""
Rolling walk-forward for frozen XGB B_drop_bottom9.

Two parallel experiments (identical windows / HPs / target):
  A) Baseline — 19 features (no sentiment)
  B) Sentiment — same 19 + Bloomberg sentiment columns

Also runs FINAL-TEST-STYLE (optional, on by default):
  train months <= 2023-12, score 2024+  (same protocol as xgb_return --final-test)
  Does NOT write / clear / write the golden-rule lock.

Outputs under:
  final test results/xgboost 2/...
  + mirrored copy -> esemble/xgboost data/

Does NOT delete or overwrite final test results/xgboost/ (original one-shot).
"""

from __future__ import annotations

import argparse
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
OUT_ROOT = ROOT / "final test results" / "xgboost 2"
# Mirror copy for ensemble stacking (does not unlock golden rule)
ENSEMBLE_COPY_DIR = ROOT / "esemble" / "xgboost data"

# Frozen recipe (unchanged)
LOCKED_N_TREES = 178
LOCKED_PARAMS = {
    "max_depth": 4,
    "learning_rate": 0.03,
    "min_child_weight": 8,
    "subsample": 0.80,
    "colsample_bytree": 0.80,
}
BASELINE_COLS = [
    "log_market_cap", "sma_20", "SMB", "MOM", "bb_lower", "debt_equity", "CMA",
    "sma_15", "HML", "macd_signal", "volatility_30d", "price_to_sma_20", "atr",
    "volume_trend", "RMW", "bb_middle", "eqy_turnover", "revenue_growth", "macd",
]
SENTIMENT_CANDIDATES = [
    "news_sentiment",
    "news_publication_count",
    "twitter_sentiment",
    "twitter_publication_count",
    "sentiment_proxy",
]

SECTOR_NEUTRAL_TARGET = True
DEMEAN_WITHIN_MONTH = True
WINSORISE_Y = True
WINSOR_LIMITS = (0.01, 0.99)
FILTER_MEMBERS = True
TOP_K = 50
TOP_FRAC = 0.10
SHAP_MAX_SAMPLES = 4000  # cap TreeSHAP cost per step
SHAP_TOP_N = 15
COMPUTE_SHAP = True

# Rolling walk-forward: train_start_year, train_end_year, eval_year
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

log = logging.getLogger("xgb_walk_forward")


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
    log_path = out_dir / f"walk_forward_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    logger = logging.getLogger("xgb_walk_forward")
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


def _sector_series(df: pd.DataFrame) -> pd.Series:
    if "Sector" not in df.columns:
        return pd.Series("Unknown", index=df.index, dtype=object)
    return df["Sector"].fillna("Unknown").astype(str).replace({"": "Unknown"})


def _filter_members(df: pd.DataFrame) -> pd.DataFrame:
    if not FILTER_MEMBERS or "is_member" not in df.columns:
        return df
    mask = df["is_member"]
    if mask.dtype != bool:
        mask = mask.astype(bool)
    return df.loc[mask].copy()


def _add_model_target(train: pd.DataFrame, eval_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = [train, eval_df]
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
    return out[0], out[1]


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


def _monthly_long_short(frame: pd.DataFrame) -> float:
    """Mean monthly top-frac minus bottom-frac by y_pred."""
    work = frame.copy()
    work["month"] = pd.to_datetime(work["month"])
    spreads = []
    for _, g in work.groupby("month"):
        if len(g) < 10:
            continue
        n = max(1, int(len(g) * TOP_FRAC))
        g = g.sort_values("y_pred")
        bottom = g.head(n)["y_true"].mean()
        top = g.tail(n)["y_true"].mean()
        spreads.append(float(top - bottom))
    return float(np.mean(spreads)) if spreads else float("nan")


def _topk_metrics(frame: pd.DataFrame, k: int = TOP_K) -> tuple[pd.DataFrame, dict]:
    work = frame.copy()
    work["month"] = pd.to_datetime(work["month"])
    market = (
        work.groupby("month", as_index=False)["y_true"]
        .mean()
        .rename(columns={"y_true": "market_avg_return"})
    )
    work = work.merge(market, on="month", how="left")
    picks = (
        work.sort_values(["month", "y_pred"], ascending=[True, False])
        .groupby("month", as_index=False)
        .head(k)
        .copy()
    )
    picks["pick_minus_market"] = picks["y_true"] - picks["market_avg_return"]
    monthly = picks.groupby("month", as_index=False).agg(
        port_return=("y_true", "mean"),
        market_return=("market_avg_return", "first"),
    )
    monthly["edge"] = monthly["port_return"] - monthly["market_return"]
    summary = {
        "topk_avg_return": float(monthly["port_return"].mean()) if len(monthly) else None,
        "topk_edge": float(monthly["edge"].mean()) if len(monthly) else None,
        "hit_rate": float((monthly["edge"] > 0).mean()) if len(monthly) else None,
        "n_months": int(monthly["month"].nunique()) if len(monthly) else 0,
        "n_picks": int(len(picks)),
    }
    return picks, summary


def _mean_abs_shap(model: XGBRegressor, X: pd.DataFrame, max_samples: int = SHAP_MAX_SAMPLES) -> pd.Series:
    """Global importance = mean |SHAP| on (possibly subsampled) eval features."""
    import shap

    if X.empty:
        return pd.Series(dtype=float)
    Xs = X
    if len(X) > max_samples:
        Xs = X.sample(n=max_samples, random_state=42)
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(Xs)
    if isinstance(sv, list):
        sv = sv[0]
    mean_abs = np.abs(np.asarray(sv)).mean(axis=0)
    return pd.Series(mean_abs, index=list(Xs.columns), name="mean_abs_shap")


def _fit_predict(
    train: pd.DataFrame,
    eval_df: pd.DataFrame,
    feature_cols: list[str],
    compute_shap: bool = True,
) -> tuple[pd.DataFrame, list[tuple[str, float]], pd.Series | None]:
    from data.features import prepare_xy

    train_t, eval_t = _add_model_target(train, eval_df)
    X_tr, y_tr, cols = prepare_xy(
        train_t, "h1_clean", target_col="target_model", feature_cols=feature_cols
    )
    X_ev, _, _ = prepare_xy(
        eval_t, "h1_clean", target_col="target_model", feature_cols=feature_cols
    )
    if X_tr.empty or X_ev.empty:
        raise ValueError("Empty train/eval feature matrix for this window.")

    model = XGBRegressor(
        n_estimators=LOCKED_N_TREES,
        max_depth=LOCKED_PARAMS["max_depth"],
        learning_rate=LOCKED_PARAMS["learning_rate"],
        min_child_weight=LOCKED_PARAMS["min_child_weight"],
        subsample=LOCKED_PARAMS["subsample"],
        colsample_bytree=LOCKED_PARAMS["colsample_bytree"],
        random_state=42,
        n_jobs=2,
        verbosity=0,
    )
    model.fit(X_tr, y_tr)
    pred = model.predict(X_ev)
    meta = eval_t.loc[X_ev.index]
    frame = pd.DataFrame({
        "Ticker": meta["Ticker"].astype(str).values,
        "month": pd.to_datetime(meta["month"]).values,
        "y_true": meta["target_excess_return"].astype(float).values,
        "y_pred": pred,
        "Sector": _sector_series(meta).values,
    })
    for c in feature_cols:
        if c in meta.columns:
            frame[c] = meta[c].to_numpy()

    imp = sorted(zip(cols, model.feature_importances_.tolist()), key=lambda x: -x[1])
    shap_s = _mean_abs_shap(model, X_ev[cols]) if compute_shap else None
    return frame, imp, shap_s


def _step_metrics(frame: pd.DataFrame, imp: list[tuple[str, float]]) -> dict:
    from evaluation.predictive import ranking_metrics

    rm = ranking_metrics(frame)
    ls = _monthly_long_short(frame)
    picks, topk = _topk_metrics(frame, k=TOP_K)
    return {
        "ic": rm.get("ic_mean"),
        "ic_std": rm.get("ic_std"),
        "long_short": ls if ls == ls else rm.get("long_short_mean"),
        "topk_avg_return": topk["topk_avg_return"],
        "topk_edge": topk["topk_edge"],
        "hit_rate": topk["hit_rate"],
        "n_eval_rows": int(len(frame)),
        "n_months": topk["n_months"],
        "n_picks": topk["n_picks"],
        "top10_importance": [{"feature": f, "importance": float(v)} for f, v in imp[:10]],
        "picks": picks,
    }


def _write_shap_global(shap_by_step: list[pd.Series], out_dir: Path, experiment_name: str) -> pd.DataFrame:
    """Average mean|SHAP| across steps → CSV, top15 table, bar plot."""
    shap_dir = out_dir / "shap"
    shap_dir.mkdir(parents=True, exist_ok=True)
    if not shap_by_step:
        return pd.DataFrame()

    mat = pd.concat(shap_by_step, axis=1)
    global_s = mat.mean(axis=1).sort_values(ascending=False)
    global_df = global_s.rename("mean_abs_shap").reset_index().rename(columns={"index": "feature"})
    global_df["rank"] = np.arange(1, len(global_df) + 1)
    global_df["share"] = global_df["mean_abs_shap"] / global_df["mean_abs_shap"].sum()
    global_df.to_csv(shap_dir / "global_mean_abs_shap.csv", index=False)

    top = global_df.head(SHAP_TOP_N).copy()
    top.to_csv(shap_dir / "top15_features.csv", index=False)

    # bar plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plot_df = top.iloc[::-1]
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.barh(plot_df["feature"], plot_df["mean_abs_shap"], color="#2c5f7c")
        ax.set_xlabel("Mean |SHAP| (avg across walk-forward steps)")
        ax.set_title(f"Global SHAP importance — {experiment_name}")
        fig.tight_layout()
        fig.savefig(shap_dir / "global_mean_abs_shap.png", dpi=150)
        plt.close(fig)
        log.info("Saved SHAP bar plot -> %s", shap_dir / "global_mean_abs_shap.png")
    except Exception as exc:
        log.warning("Could not save SHAP plot: %s", exc)

    log.info("Saved global SHAP tables -> %s", shap_dir)
    return global_df


def run_experiment(
    panel: pd.DataFrame,
    feature_cols: list[str],
    experiment_name: str,
    out_dir: Path,
    compute_shap: bool = COMPUTE_SHAP,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    shap_by_step: list[pd.Series] = []
    log.info("=" * 70)
    log.info("EXPERIMENT: %s | n_features=%d | shap=%s", experiment_name, len(feature_cols), compute_shap)
    log.info("Features: %s", ", ".join(feature_cols))
    log.info("Output: %s", out_dir)
    log.info("=" * 70)

    for i, (tr0, tr1, ev_year) in enumerate(WALK_WINDOWS, start=1):
        step_tag = f"step_{i:02d}_{ev_year}"
        step_dir = out_dir / step_tag
        step_dir.mkdir(parents=True, exist_ok=True)

        train = _slice_years(panel, tr0, tr1)
        eval_df = _slice_eval_year(panel, ev_year)
        log.info(
            "[%s] train %d-%d (%d rows) -> eval %d (%d rows)",
            step_tag, tr0, tr1, len(train), ev_year, len(eval_df),
        )
        if train.empty or eval_df.empty:
            log.warning("  SKIP empty train/eval")
            rows.append({
                "step": i,
                "eval_year": ev_year,
                "train_start": tr0,
                "train_end": tr1,
                "ic": np.nan,
                "long_short": np.nan,
                "topk_avg_return": np.nan,
                "topk_edge": np.nan,
                "hit_rate": np.nan,
                "n_eval_rows": 0,
                "n_months": 0,
                "status": "skipped_empty",
            })
            continue

        try:
            frame, imp, shap_s = _fit_predict(
                train, eval_df, feature_cols, compute_shap=compute_shap
            )
            metrics = _step_metrics(frame, imp)
        except Exception as exc:
            log.exception("  FAILED: %s", exc)
            rows.append({
                "step": i,
                "eval_year": ev_year,
                "train_start": tr0,
                "train_end": tr1,
                "ic": np.nan,
                "long_short": np.nan,
                "topk_avg_return": np.nan,
                "topk_edge": np.nan,
                "hit_rate": np.nan,
                "n_eval_rows": len(eval_df),
                "n_months": 0,
                "status": f"error: {exc}",
            })
            continue

        metrics_path = step_dir / "metrics.json"
        payload = {
            "step": i,
            "eval_year": ev_year,
            "train_start": tr0,
            "train_end": tr1,
            "experiment": experiment_name,
            "n_features": len(feature_cols),
            "feature_cols": feature_cols,
            "hyperparameters": {**LOCKED_PARAMS, "n_estimators": LOCKED_N_TREES},
            "ic": metrics["ic"],
            "ic_std": metrics["ic_std"],
            "long_short": metrics["long_short"],
            "topk_avg_return": metrics["topk_avg_return"],
            "topk_edge": metrics["topk_edge"],
            "hit_rate": metrics["hit_rate"],
            "n_eval_rows": metrics["n_eval_rows"],
            "n_months": metrics["n_months"],
            "top10_importance": metrics["top10_importance"],
        }
        if shap_s is not None and len(shap_s):
            shap_step = shap_s.sort_values(ascending=False).rename("mean_abs_shap").reset_index()
            shap_step.columns = ["feature", "mean_abs_shap"]
            shap_step["rank"] = np.arange(1, len(shap_step) + 1)
            shap_step.to_csv(step_dir / "shap_mean_abs.csv", index=False)
            shap_by_step.append(shap_s.rename(f"step_{i}"))
            top_shap = list(shap_s.sort_values(ascending=False).head(5).items())
            payload["shap_top5"] = [{"feature": f, "mean_abs_shap": float(v)} for f, v in top_shap]
            sent_shap = [(f, float(v)) for f, v in shap_s.items() if f in SENTIMENT_CANDIDATES]
            sent_shap.sort(key=lambda x: -x[1])
            if sent_shap:
                log.info(
                    "  SHAP sentiment: %s",
                    ", ".join(f"{f}={v:.5f}(rank {int((shap_s.rank(ascending=False)[f]))})" for f, v in sent_shap[:3]),
                )

        metrics_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        # Full OOS panel for ensemble blend (parity with LSTM walk_forward preview.csv)
        try:
            frame.to_csv(step_dir / "preview.csv", index=False)
        except Exception as preview_err:
            log.warning("Could not save preview.csv for %s: %s", step_tag, preview_err)
        picks = metrics["picks"]
        if picks is not None and len(picks):
            picks.to_csv(step_dir / "top50_picks.csv", index=False)

        log.info(
            "  IC=%.4f  LS=%.4f  Top50_edge=%.4f  hit=%.1f%%  rows=%d",
            metrics["ic"] or float("nan"),
            metrics["long_short"] or float("nan"),
            metrics["topk_edge"] or float("nan"),
            100.0 * (metrics["hit_rate"] or 0.0),
            metrics["n_eval_rows"],
        )

        rows.append({
            "step": i,
            "eval_year": ev_year,
            "train_start": tr0,
            "train_end": tr1,
            "ic": metrics["ic"],
            "long_short": metrics["long_short"],
            "topk_avg_return": metrics["topk_avg_return"],
            "topk_edge": metrics["topk_edge"],
            "hit_rate": metrics["hit_rate"],
            "n_eval_rows": metrics["n_eval_rows"],
            "n_months": metrics["n_months"],
            "top1_feature": metrics["top10_importance"][0]["feature"] if metrics["top10_importance"] else "",
            "top1_importance": metrics["top10_importance"][0]["importance"] if metrics["top10_importance"] else np.nan,
            "status": "ok",
        })

    summary = pd.DataFrame(rows)
    num_cols = ["ic", "long_short", "topk_avg_return", "topk_edge", "hit_rate", "n_eval_rows", "n_months"]
    avg = {"step": 0, "eval_year": "AVERAGE", "train_start": "", "train_end": "", "status": "average"}
    ok = summary[summary["status"] == "ok"]
    for c in num_cols:
        avg[c] = float(ok[c].mean()) if len(ok) and c in ok.columns else np.nan
    summary = pd.concat([summary, pd.DataFrame([avg])], ignore_index=True)

    summary_name = f"walk_forward_{experiment_name}_summary.csv"
    summary_path = out_dir / summary_name
    summary.to_csv(summary_path, index=False)
    summary.to_csv(OUT_ROOT / summary_name, index=False)
    log.info("Saved summary -> %s", summary_path)

    shap_global = _write_shap_global(shap_by_step, out_dir, experiment_name) if compute_shap else pd.DataFrame()
    return summary, shap_global


def _write_shap_comparison(base_shap: pd.DataFrame, sent_shap: pd.DataFrame, out_root: Path) -> None:
    """Side-by-side SHAP ranks + short markdown note for the dissertation."""
    if base_shap.empty or sent_shap.empty:
        log.warning("Skipping SHAP comparison — missing global tables.")
        return

    b = base_shap.rename(columns={
        "mean_abs_shap": "baseline_mean_abs_shap",
        "rank": "baseline_rank",
        "share": "baseline_share",
    })
    s = sent_shap.rename(columns={
        "mean_abs_shap": "sentiment_mean_abs_shap",
        "rank": "sentiment_rank",
        "share": "sentiment_share",
    })
    m = b.merge(s, on="feature", how="outer")
    m["delta_mean_abs_shap"] = m["sentiment_mean_abs_shap"] - m["baseline_mean_abs_shap"]
    m["is_sentiment_feature"] = m["feature"].isin(SENTIMENT_CANDIDATES)
    m = m.sort_values("sentiment_mean_abs_shap", ascending=False, na_position="last")
    cmp_path = out_root / "shap_baseline_vs_sentiment_comparison.csv"
    m.to_csv(cmp_path, index=False)
    log.info("Saved SHAP comparison -> %s", cmp_path)

    # Brief narrative
    sent_rows = m[m["is_sentiment_feature"]].copy()
    trad = ["MOM", "SMB", "HML", "RMW", "CMA", "log_market_cap"]
    lines = [
        "# SHAP comparison — baseline vs sentiment (walk-forward average)",
        "",
        "Importance metric: **mean |SHAP|** averaged across all 14 rolling steps.",
        f"Eval sample capped at {SHAP_MAX_SAMPLES} rows/step for TreeSHAP.",
        "",
        "## Does sentiment rank high?",
    ]
    if sent_rows.empty:
        lines.append("No sentiment features present in the SHAP tables.")
    else:
        for _, r in sent_rows.sort_values("sentiment_rank").iterrows():
            rank = r["sentiment_rank"]
            share = r["sentiment_share"]
            lines.append(
                f"- `{r['feature']}`: rank **{int(rank) if pd.notna(rank) else 'n/a'}** "
                f"(share {100 * share:.1f}% of total mean|SHAP|)" if pd.notna(share)
                else f"- `{r['feature']}`: rank n/a"
            )
        best = sent_rows.sort_values("sentiment_rank").iloc[0]
        if pd.notna(best["sentiment_rank"]) and best["sentiment_rank"] <= 15:
            lines.append(
                f"\n**Answer:** Yes — best sentiment feature `{best['feature']}` is inside the top 15 "
                f"(rank {int(best['sentiment_rank'])})."
            )
        else:
            lines.append(
                "\n**Answer:** No — sentiment features do **not** enter the global top 15; "
                "traditional / technical signals dominate."
            )

    lines.extend(["", "## Do traditional factors lose importance?"])
    for f in trad:
        br = m.loc[m["feature"] == f]
        if br.empty:
            continue
        row = br.iloc[0]
        b_rank, s_rank = row.get("baseline_rank"), row.get("sentiment_rank")
        b_sh, s_sh = row.get("baseline_share"), row.get("sentiment_share")
        if pd.isna(b_rank) or pd.isna(s_rank):
            continue
        delta_share = (s_sh - b_sh) if pd.notna(b_sh) and pd.notna(s_sh) else np.nan
        lines.append(
            f"- `{f}`: baseline rank {int(b_rank)} → sentiment model rank {int(s_rank)}"
            + (f" (share Δ {100 * delta_share:+.1f} pp)" if pd.notna(delta_share) else "")
        )

    # Top 15 each
    lines.extend(["", "## Top 15 — baseline", ""])
    for _, r in base_shap.head(SHAP_TOP_N).iterrows():
        lines.append(f"{int(r['rank'])}. `{r['feature']}` — {r['mean_abs_shap']:.6f}")
    lines.extend(["", "## Top 15 — sentiment model", ""])
    for _, r in sent_shap.head(SHAP_TOP_N).iterrows():
        lines.append(f"{int(r['rank'])}. `{r['feature']}` — {r['mean_abs_shap']:.6f}")

    note_path = out_root / "shap_comparison_note.md"
    note_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info("Saved SHAP note -> %s", note_path)


def run_final_test_style(
    panel: pd.DataFrame,
    feature_cols: list[str],
    experiment_name: str,
    out_dir: Path,
    compute_shap: bool = False,
) -> dict:
    """
    Mimic xgb_return --final-test protocol WITHOUT unlocking the golden rule:
      fit on all months through 2023-12, score 2024-01 .. end of panel.
    Saves under out_dir; never writes GOLDEN_RULE_*.txt.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    m = pd.to_datetime(panel["month"])
    train = panel.loc[m <= pd.Timestamp("2023-12-01")].copy()
    test = panel.loc[m >= pd.Timestamp("2024-01-01")].copy()
    log.info("=" * 70)
    log.info(
        "FINAL-TEST-STYLE (NO GOLDEN LOCK) | %s | train<=2023 (%d) -> test 2024+ (%d)",
        experiment_name, len(train), len(test),
    )
    log.info("=" * 70)
    if train.empty or test.empty:
        raise ValueError("Empty train/test for final-test-style split.")

    frame, imp, shap_s = _fit_predict(train, test, feature_cols, compute_shap=compute_shap)
    metrics = _step_metrics(frame, imp)

    picks = metrics["picks"]
    if picks is not None and len(picks):
        picks.to_csv(out_dir / f"xgb_topk_picks_{experiment_name}_test.csv", index=False)
    frame.to_csv(out_dir / f"xgb_preview_{experiment_name}_test.csv", index=False)

    if shap_s is not None and len(shap_s):
        shap_df = shap_s.sort_values(ascending=False).rename("mean_abs_shap").reset_index()
        shap_df.columns = ["feature", "mean_abs_shap"]
        shap_df["rank"] = np.arange(1, len(shap_df) + 1)
        shap_df.to_csv(out_dir / "shap_mean_abs_test.csv", index=False)

    payload = {
        "mode": "final_test_style_no_golden_lock",
        "experiment": experiment_name,
        "train_end": "2023-12",
        "test_start": "2024-01",
        "test_end": str(pd.to_datetime(test["month"]).max().date()),
        "n_features": len(feature_cols),
        "feature_cols": feature_cols,
        "hyperparameters": {**LOCKED_PARAMS, "n_estimators": LOCKED_N_TREES},
        "ic": metrics["ic"],
        "ic_std": metrics["ic_std"],
        "long_short": metrics["long_short"],
        "topk_avg_return": metrics["topk_avg_return"],
        "topk_edge": metrics["topk_edge"],
        "hit_rate": metrics["hit_rate"],
        "n_eval_rows": metrics["n_eval_rows"],
        "n_months": metrics["n_months"],
        "top10_importance": metrics["top10_importance"],
        "note": (
            "Same fit protocol as xgb_return --final-test (train through 2023, score 2024-2025). "
            "This does NOT write or clear the golden-rule lock."
        ),
    }
    (out_dir / "metrics_final_test_style.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8"
    )

    # Tracker-like one-row CSV (no lock interaction)
    tracker = pd.DataFrame([{
        "run_id": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "date": datetime.now().strftime("%d/%m/%Y"),
        "model": "XGB",
        "experiment": experiment_name,
        "evaluated_on": "test_final_style_no_lock",
        "depth": LOCKED_PARAMS["max_depth"],
        "lr": LOCKED_PARAMS["learning_rate"],
        "trees": LOCKED_N_TREES,
        "test_ic": metrics["ic"],
        "test_ls": metrics["long_short"],
        "test_topk_edge": metrics["topk_edge"],
        "test_hit_rate": metrics["hit_rate"],
        "n_test_rows": metrics["n_eval_rows"],
        "golden_lock_written": False,
    }])
    tracker.to_csv(out_dir / "results_tracker_final_style.csv", index=False)

    log.info(
        "  FINAL-STYLE %s | IC=%.4f LS=%.4f Top50_edge=%.4f hit=%.1f%% | NO LOCK WRITTEN",
        experiment_name,
        metrics["ic"] or float("nan"),
        metrics["long_short"] or float("nan"),
        metrics["topk_edge"] or float("nan"),
        100.0 * (metrics["hit_rate"] or 0.0),
    )
    return payload


def _mirror_copy_to_ensemble(src_root: Path, dst_root: Path, log_path: Path | None = None) -> None:
    """Copy walk-forward + final-style artefacts into esemble/xgboost data."""
    import shutil

    dst_root.mkdir(parents=True, exist_ok=True)
    # Fresh mirror of key trees (replace previous mirror content for those names)
    for name in (
        "walk_forward_baseline",
        "walk_forward_sentiment",
        "final_test_style_baseline",
        "final_test_style_sentiment",
    ):
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
        "shap_baseline_vs_sentiment_comparison.csv",
        "shap_comparison_note.md",
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
            "Copy of XGB walk-forward + final-test-style outputs for ensemble use.",
            f"Source: {src_root}",
            f"Copied: {datetime.now().isoformat(timespec='seconds')}",
            "",
            "NOTE: final_test_style_* does NOT unlock/write the golden-rule lock.",
            "Original locked one-shot remains in: final test results/xgboost/",
            "",
        ]),
        encoding="utf-8",
    )
    log.info("Ensemble mirror ready -> %s", dst_root)


def _write_comparison(base: pd.DataFrame, sent: pd.DataFrame, path: Path) -> None:
    b = base[base["status"].isin(["ok", "average"])].copy()
    s = sent[sent["status"].isin(["ok", "average"])].copy()
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
    m = m.sort_values(["step"])
    path.parent.mkdir(parents=True, exist_ok=True)
    m.to_csv(path, index=False)
    log.info("Saved comparison -> %s", path)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="XGB rolling walk-forward (baseline vs sentiment)")
    p.add_argument("--baseline-only", action="store_true")
    p.add_argument("--sentiment-only", action="store_true")
    p.add_argument("--skip-shap", action="store_true", help="Skip TreeSHAP (faster).")
    p.add_argument(
        "--skip-walk-forward",
        action="store_true",
        help="Skip 14-step rolling WF; only run final-test-style block.",
    )
    p.add_argument(
        "--skip-final-style",
        action="store_true",
        help="Skip final-test-style (train<=2023 -> score 2024+) block.",
    )
    p.add_argument(
        "--skip-ensemble-copy",
        action="store_true",
        help="Do not mirror outputs into esemble/xgboost data.",
    )
    args = p.parse_args(argv)

    _configure_paths()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    log_path = setup_logging(OUT_ROOT)
    warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)

    from data.build_panel import build_panel

    compute_shap = not args.skip_shap
    log.info("Loading panel... | compute_shap=%s", compute_shap)
    log.info("Golden rule: this script NEVER writes GOLDEN_RULE lock files.")
    pdata, _splits = build_panel()
    panel = _filter_members(pdata.panel.copy())
    panel["month"] = pd.to_datetime(panel["month"])
    log.info(
        "Panel rows=%d | months %s -> %s | members-filtered",
        len(panel),
        panel["month"].min().date(),
        panel["month"].max().date(),
    )

    sent_cols = [c for c in SENTIMENT_CANDIDATES if c in panel.columns]
    for c in SENTIMENT_CANDIDATES:
        if c in panel.columns:
            log.info("Sentiment col %-28s non-null=%.1f%%", c, 100.0 * panel[c].notna().mean())
        else:
            log.info("Sentiment col %-28s MISSING", c)

    sentiment_features = list(BASELINE_COLS) + sent_cols
    run_base = not args.sentiment_only
    run_sent = not args.baseline_only

    base_summary = None
    sent_summary = None
    base_shap = pd.DataFrame()
    sent_shap = pd.DataFrame()

    if not args.skip_walk_forward:
        if run_base:
            base_summary, base_shap = run_experiment(
                panel,
                list(BASELINE_COLS),
                "baseline",
                OUT_ROOT / "walk_forward_baseline",
                compute_shap=compute_shap,
            )
        if run_sent:
            sent_summary, sent_shap = run_experiment(
                panel,
                sentiment_features,
                "sentiment",
                OUT_ROOT / "walk_forward_sentiment",
                compute_shap=compute_shap,
            )

        if base_summary is not None and sent_summary is not None:
            _write_comparison(base_summary, sent_summary, OUT_ROOT / "walk_forward_comparison.csv")
        if compute_shap and not base_shap.empty and not sent_shap.empty:
            _write_shap_comparison(base_shap, sent_shap, OUT_ROOT)
    else:
        log.info("Skipping walk-forward (--skip-walk-forward).")

    if not args.skip_final_style:
        log.info(
            "Running FINAL-TEST-STYLE blocks (train through 2023, score 2024-2025) — NO GOLDEN LOCK."
        )
        if run_base:
            run_final_test_style(
                panel,
                list(BASELINE_COLS),
                "baseline",
                OUT_ROOT / "final_test_style_baseline",
                compute_shap=False,
            )
        if run_sent:
            run_final_test_style(
                panel,
                sentiment_features,
                "sentiment",
                OUT_ROOT / "final_test_style_sentiment",
                compute_shap=False,
            )
    else:
        log.info("Skipping final-test-style (--skip-final-style).")

    readme = OUT_ROOT / "README_walk_forward.txt"
    readme.write_text(
        "\n".join([
            "XGBoost rolling walk-forward (xgboost 2)",
            "=======================================",
            f"Created: {datetime.now().isoformat(timespec='seconds')}",
            f"Log: {log_path}",
            "",
            "Settings:",
            "  Rolling 10y train / 1y eval / 1y step",
            "  HPs: depth=4, lr=0.03, mcw=8, trees=178",
            "  Target: sector-neutral, winsor 1%/99%",
            "  Liquidity filter: OFF",
            f"  SHAP: {'ON' if compute_shap else 'OFF'} (mean |SHAP|, max {SHAP_MAX_SAMPLES} rows/step)",
            "",
            "Also runs FINAL-TEST-STYLE (no golden lock):",
            "  train months <= 2023-12, score 2024+  (same protocol as xgb_return --final-test)",
            "  Outputs: final_test_style_baseline/ and final_test_style_sentiment/",
            "  NEVER writes GOLDEN_RULE_test_already_scored.txt",
            "",
            "Folders:",
            "  walk_forward_baseline/   — B_drop_bottom9 (19 features)",
            "  walk_forward_sentiment/  — 19 + sentiment columns",
            "  final_test_style_* /     — holdout refit report (no lock)",
            "  walk_forward_comparison.csv",
            "  shap_baseline_vs_sentiment_comparison.csv",
            "  shap_comparison_note.md",
            "",
            "Ensemble mirror copy:",
            f"  {ENSEMBLE_COPY_DIR}",
            "",
            "Original one-shot final test lock folder (untouched by this script):",
            "  ../xgboost/",
            "",
        ]),
        encoding="utf-8",
    )

    if not args.skip_ensemble_copy:
        _mirror_copy_to_ensemble(OUT_ROOT, ENSEMBLE_COPY_DIR, log_path=log_path)
    else:
        log.info("Skipping ensemble copy (--skip-ensemble-copy).")

    log.info("Done. Outputs under %s", OUT_ROOT)
    log.info("Golden rule lock was NOT written by this run.")


if __name__ == "__main__":
    main()

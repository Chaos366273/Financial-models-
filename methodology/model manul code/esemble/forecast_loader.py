"""Load saved XGB/LSTM walk-forward forecasts for Stage-1 ensemble (no retrain)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

CORE_COLS = ("Ticker", "month", "y_pred", "y_true")


def _read_step_frames(root: Path) -> tuple[pd.DataFrame, str]:
    """Concat per-step forecast CSVs. Prefer preview.csv; fall back to top50_picks.csv."""
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Forecast root not found: {root}")

    preview_paths = sorted(root.glob("step_*/preview.csv"))
    top50_paths = sorted(root.glob("step_*/top50_picks.csv"))
    source = "preview"
    paths = preview_paths
    if not paths:
        paths = top50_paths
        source = "top50_picks"
    if not paths:
        raise FileNotFoundError(
            f"No step_*/preview.csv or top50_picks.csv under {root}"
        )

    frames: list[pd.DataFrame] = []
    for p in paths:
        df = pd.read_csv(p)
        missing = [c for c in ("Ticker", "month", "y_pred") if c not in df.columns]
        if missing:
            raise ValueError(f"{p} missing columns: {missing}")
        keep = [c for c in CORE_COLS if c in df.columns]
        if "Sector" in df.columns:
            keep.append("Sector")
        chunk = df[keep].copy()
        chunk["Ticker"] = chunk["Ticker"].astype(str)
        chunk["month"] = pd.to_datetime(chunk["month"])
        chunk["y_pred"] = pd.to_numeric(chunk["y_pred"], errors="coerce")
        if "y_true" in chunk.columns:
            chunk["y_true"] = pd.to_numeric(chunk["y_true"], errors="coerce")
        frames.append(chunk)

    out = pd.concat(frames, ignore_index=True)
    out = out.drop_duplicates(subset=["Ticker", "month"], keep="last")
    return out, source


def _mean_ic_from_summary(summary_path: Path | None) -> float | None:
    if summary_path is None or not Path(summary_path).is_file():
        return None
    s = pd.read_csv(summary_path)
    if "ic" not in s.columns:
        return None
    work = s.copy()
    if "step" in work.columns:
        work = work.loc[work["step"].astype(str) != "0"]
        work = work.loc[work["step"].astype(str).str.upper() != "AVERAGE"]
    ic = pd.to_numeric(work["ic"], errors="coerce").dropna()
    if ic.empty:
        return None
    return float(ic.mean())


def resolve_summary_path(wf_root: Path) -> Path | None:
    """Find a walk-forward summary CSV near the step folders."""
    root = Path(wf_root)
    candidates = [
        root / "walk_forward_baseline_summary.csv",
        root / "summary_metrics.csv",
        root.parent / f"{root.name}_summary.csv",
        root.parent / "walk_forward_baseline_summary.csv",
    ]
    # e.g. .../walk_forward_baseline -> sibling summary at parent
    name = root.name
    candidates.append(root.parent / f"{name}_summary.csv")
    for c in candidates:
        if c.is_file():
            return c
    # any *summary*.csv in root or parent
    for folder in (root, root.parent):
        hits = sorted(folder.glob("*summary*.csv"))
        if hits:
            return hits[0]
    return None


def ic_blend_weight(
    xgb_summary: Path | None,
    lstm_summary: Path | None,
    *,
    fallback: float = 0.7,
) -> float:
    """Weight on XGB from relative mean IC; clip to [0.05, 0.95]."""
    ix = _mean_ic_from_summary(xgb_summary)
    il = _mean_ic_from_summary(lstm_summary)
    if ix is None and il is None:
        return float(fallback)
    if ix is None:
        return 0.3
    if il is None:
        return 0.85
    # Shift ICs so both positive-ish for weighting; allow negative IC legs low weight
    sx = max(ix, 0.0)
    sl = max(il, 0.0)
    if sx + sl <= 1e-12:
        return float(fallback)
    w = sx / (sx + sl)
    return float(np.clip(w, 0.05, 0.95))


def load_and_blend_forecasts(
    xgb_root: Path,
    lstm_root: Path,
    *,
    blend_mode: str = "ic_weight",
    fixed_w_xgb: float = 0.7,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Merge XGB+LSTM OOS forecasts and produce y_pred blend frame."""
    xgb_df, xgb_src = _read_step_frames(xgb_root)
    lstm_df, lstm_src = _read_step_frames(lstm_root)

    xgb_df = xgb_df.rename(columns={"y_pred": "xgb_pred"})
    lstm_df = lstm_df.rename(columns={"y_pred": "lstm_pred"})

    xgb_keep = ["Ticker", "month", "xgb_pred"]
    if "y_true" in xgb_df.columns:
        xgb_keep.append("y_true")
    lstm_keep = ["Ticker", "month", "lstm_pred"]
    if "y_true" in lstm_df.columns and "y_true" not in xgb_keep:
        lstm_keep.append("y_true")

    merged = xgb_df[xgb_keep].merge(
        lstm_df[lstm_keep],
        on=["Ticker", "month"],
        how="inner",
        suffixes=("_xgb", "_lstm"),
    )
    if merged.empty:
        raise ValueError(
            "No overlapping (Ticker, month) rows between XGB and LSTM forecasts. "
            f"xgb_src={xgb_src} n={len(xgb_df)} | lstm_src={lstm_src} n={len(lstm_df)}"
        )

    if "y_true" in merged.columns:
        y_true = merged["y_true"]
    elif "y_true_xgb" in merged.columns:
        y_true = merged["y_true_xgb"].fillna(merged.get("y_true_lstm"))
    elif "y_true_lstm" in merged.columns:
        y_true = merged["y_true_lstm"]
    else:
        y_true = np.nan

    xgb_summary = resolve_summary_path(xgb_root)
    lstm_summary = resolve_summary_path(lstm_root)
    if blend_mode == "fixed":
        w = float(np.clip(fixed_w_xgb, 0.0, 1.0))
    else:
        w = ic_blend_weight(xgb_summary, lstm_summary, fallback=fixed_w_xgb)

    merged["y_pred"] = w * merged["xgb_pred"].astype(float) + (1.0 - w) * merged["lstm_pred"].astype(float)
    merged["y_true"] = y_true
    merged["error"] = merged["y_true"] - merged["y_pred"]
    merged["abs_error"] = merged["error"].abs()

    meta = {
        "xgb_root": str(xgb_root),
        "lstm_root": str(lstm_root),
        "xgb_source": xgb_src,
        "lstm_source": lstm_src,
        "xgb_summary": str(xgb_summary) if xgb_summary else None,
        "lstm_summary": str(lstm_summary) if lstm_summary else None,
        "blend_mode": blend_mode,
        "w_xgb": w,
        "n_rows": int(len(merged)),
        "n_tickers": int(merged["Ticker"].nunique()),
        "month_min": str(merged["month"].min().date()),
        "month_max": str(merged["month"].max().date()),
    }
    cols = ["Ticker", "month", "xgb_pred", "lstm_pred", "y_pred", "y_true", "error", "abs_error"]
    return merged[cols].sort_values(["month", "Ticker"]).reset_index(drop=True), meta

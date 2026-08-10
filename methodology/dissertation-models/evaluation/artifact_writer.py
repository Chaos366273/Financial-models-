"""Write prediction/backtest CSV artifacts and attach calculation blocks."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from data.config_loader import ROOT, load_config
from evaluation.calculation_manifest import metric_definitions, shared_manifest
from evaluation.example_selection import records_from_frame, select_examples


def artifact_root(run_id: str) -> Path:
    cfg = load_config()
    safe_run = run_id.replace(":", "").replace("+", "")
    return Path(cfg["outputs"]["dir"]) / "artifacts" / safe_run


def write_predictions_csv(
    run_id: str,
    model_id: str,
    frame: pd.DataFrame,
    filename: str = "predictions.csv",
) -> dict[str, str] | None:
    if frame is None or frame.empty:
        return None
    out_dir = artifact_root(run_id) / model_id
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / filename
    frame.to_csv(path, index=False)
    rel = path.relative_to(ROOT).as_posix()
    return {
        "predictions_csv": rel,
        "predictions_file": filename,
        "model_id": model_id,
        "run_id": run_id,
    }


def build_calculation(
    *,
    run_id: str,
    model_id: str,
    bucket: str,
    data_context: dict[str, Any] | None,
    frame: pd.DataFrame | None,
    model_meta: dict[str, Any],
    feature_set: str = "traditional",
    error_col: str = "abs_error",
    preview_columns: list[str] | None = None,
    n_examples: int = 10,
    artifact_key: str | None = None,
) -> dict[str, Any]:
    base = shared_manifest(data_context, feature_set=feature_set)
    calc: dict[str, Any] = {
        **base,
        "model": model_meta,
        "metric_definitions": metric_definitions(bucket),
        "oos": {
            "unit": model_meta.get("oos_unit", "ticker_month"),
            "n_total": int(len(frame)) if frame is not None else 0,
            "selection_method": "stratified_abs_error_v1",
            "examples_preview": [],
            "artifacts": {},
        },
    }

    # Allow model-specific target overrides (Markov, H3a, H3b)
    if "target_override" in model_meta:
        calc["target"] = model_meta["target_override"]

    if frame is None or frame.empty:
        return calc

    preview = select_examples(
        frame,
        run_id=run_id,
        model_id=model_id,
        error_col=error_col,
        n=n_examples,
    )
    cols = preview_columns or [
        c for c in (
            "Ticker", "month", "date",
            "y_true", "y_pred", "error", "abs_error",
            "actual", "forecast", "port_return", "violation",
            "mkt_actual", "mkt_pred", "regime_prob_0", "regime_prob_1",
            "rf_pred", "xgb_pred", "lstm_pred",
        )
        if c in preview.columns
    ]
    calc["oos"]["examples_preview"] = records_from_frame(preview, cols)
    calc["oos"]["n_total"] = int(len(frame))

    key = artifact_key or model_id
    artifacts = write_predictions_csv(run_id, key, frame)
    if artifacts:
        artifacts["model_id"] = model_id
        calc["oos"]["artifacts"] = artifacts
    return calc

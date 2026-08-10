#!/usr/bin/env python3
"""CLI: run dissertation models and export Lab-compatible JSON bundle."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from data.build_panel import build_panel
from data.config_loader import load_config
from data.features import get_feature_columns
from evaluation.export_results import ResultsStore
from models.ff5 import run_ff5
from models.garch_lstm import run_garch_lstm
from models.garch_var import run_garch_var
from models.h2_runner import run_h2_and_shap
from models.lstm_model import run_lstm
from models.markov_switching import run_markov_switching
from models.portfolios import run_portfolios
from models.random_forest import run_random_forest
from models.stacked_ensemble import run_stacked_ensemble
from models.xgboost_model import run_xgboost

H1_MODELS = ["ff5", "markov_switching", "random_forest", "xgboost", "lstm", "stacked_ensemble"]
H3A_MODELS = ["equal_weight", "mean_variance", "min_variance", "black_litterman", "ml_black_litterman"]
H3B_MODELS = ["garch_var", "garch_lstm"]
ALL_MODELS = H1_MODELS + ["h2"] + H3A_MODELS + H3B_MODELS


def _frame_period(frame: pd.DataFrame) -> dict:
    dates = pd.to_datetime(frame["month"])
    return {
        "start": dates.min().date().isoformat(),
        "end": dates.max().date().isoformat(),
        "rows": int(len(frame)),
    }


def set_run_context(store: ResultsStore, pdata, splits: dict) -> None:
    features_path = Path(splits["features_csv"]) if splits.get("features_csv") else Path(".")
    fundamentals_raw = splits.get("fundamentals_csv")
    fundamentals_path = Path(fundamentals_raw) if fundamentals_raw else Path(".")
    master = splits.get("master_panel")
    dataset_label = os.environ.get("DISSLAB_DATASET_LABEL")
    if not dataset_label:
        if master:
            dataset_label = "final_data_master_panel"
        else:
            dataset_label = "pilot" if "data\\pilot" in str(features_path).lower() else features_path.parent.name
    context = {
        "dataset": dataset_label,
        "features_csv": str(master or features_path),
        "fundamentals_csv": str(fundamentals_path) if fundamentals_raw else "",
        "master_panel": master,
        "train": _frame_period(splits["train"]),
        "validation": _frame_period(splits["val"]),
        "test": _frame_period(splits["test"]),
        "tickers": len(pdata.tickers),
    }
    store.set_data_context(context)
    print(
        f"Data: {dataset_label} | {Path(context['features_csv']).name} | "
        f"{context['tickers']} tickers | "
        f"train {context['train']['start']}..{context['train']['end']} | "
        f"validation {context['validation']['start']}..{context['validation']['end']} | "
        f"test {context['test']['start']}..{context['test']['end']}"
    )


def build_ml_forecasts(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """XGBoost per-ticker return forecasts for ML→Black-Litterman views."""
    cfg = load_config()
    cols = [c for c in get_feature_columns("traditional") if c in train.columns]
    tr = train.dropna(subset=cols + ["target_excess_return"])
    te = test.dropna(subset=cols).copy()
    if tr.empty or te.empty:
        return pd.DataFrame(columns=["Ticker", "pred"])
    sc = StandardScaler()
    X_tr = sc.fit_transform(tr[cols].replace([np.inf, -np.inf], np.nan).fillna(0.0))
    X_te = sc.transform(te[cols].replace([np.inf, -np.inf], np.nan).fillna(0.0))
    model = XGBRegressor(
        n_estimators=cfg["xgboost"]["n_estimators"],
        max_depth=cfg["xgboost"]["max_depth"],
        learning_rate=cfg["xgboost"]["learning_rate"],
        random_state=42,
        n_jobs=1,
    )
    model.fit(X_tr, tr["target_excess_return"])
    te["pred"] = model.predict(X_te)
    return te[["Ticker", "pred"]]


def run_single(model_id: str, store: ResultsStore) -> None:
    pdata, splits = build_panel()
    set_run_context(store, pdata, splits)
    train, val, test = splits["train"], splits["val"], splits["test"]
    fold = splits["fold"]
    val_end = splits["val_end"]
    daily = pdata.daily
    test_start = val_end + pd.Timedelta(days=1)

    if model_id == "ff5":
        run_ff5(train, test, fold, store)
    elif model_id == "markov_switching":
        run_markov_switching(train, val, test, fold, store)
    elif model_id == "random_forest":
        run_random_forest(train, test, fold, store)
    elif model_id == "xgboost":
        run_xgboost(train, test, fold, store)
    elif model_id == "lstm":
        run_lstm(train, test, fold, store)
    elif model_id == "stacked_ensemble":
        run_stacked_ensemble(train, val, test, fold, store)
    elif model_id == "h2":
        run_h2_and_shap(train, val, test, fold, store)
    elif model_id in H3A_MODELS:
        ml_fc = build_ml_forecasts(train, test) if model_id == "ml_black_litterman" else None
        hist_panel = pd.concat([train, val], ignore_index=True)
        run_portfolios(
            hist_panel,
            test,
            fold,
            store,
            ml_forecasts=ml_fc,
            strategy_ids={model_id},
        )
    elif model_id == "garch_var":
        run_garch_var(daily, fold, store, test_start=test_start)
    elif model_id == "garch_lstm":
        run_garch_lstm(daily, fold, store, test_start=test_start)
    else:
        raise ValueError(f"Unknown model: {model_id}")


def run_all(store: ResultsStore) -> None:
    pdata, splits = build_panel()
    set_run_context(store, pdata, splits)
    train, val, test = splits["train"], splits["val"], splits["test"]
    fold = splits["fold"]
    val_end = splits["val_end"]
    daily = pdata.daily
    test_start = val_end + pd.Timedelta(days=1)

    run_ff5(train, test, fold, store)
    run_markov_switching(train, val, test, fold, store)
    run_random_forest(train, test, fold, store)
    run_xgboost(train, test, fold, store)
    run_lstm(train, test, fold, store)
    run_stacked_ensemble(train, val, test, fold, store)
    run_h2_and_shap(train, val, test, fold, store)

    ml_fc = build_ml_forecasts(train, test)
    hist_panel = pd.concat([train, val], ignore_index=True)
    run_portfolios(hist_panel, test, fold, store, ml_forecasts=ml_fc)

    run_garch_var(daily, fold, store, test_start=test_start)
    run_garch_lstm(daily, fold, store, test_start=test_start)


def validate_full_run(store: ResultsStore) -> None:
    expected = {
        "h1": set(H1_MODELS),
        "h3a": set(H3A_MODELS),
        "h3b": set(H3B_MODELS),
    }
    actual = {
        "h1": {item["model_id"] for item in store.h1},
        "h3a": {item["strategy_id"] for item in store.h3a},
        "h3b": {item["model_id"] for item in store.h3b},
    }
    missing = [
        f"{bucket}: {', '.join(sorted(ids - actual[bucket]))}"
        for bucket, ids in expected.items()
        if ids - actual[bucket]
    ]
    if missing:
        raise RuntimeError("Full run incomplete; bundle not written. Missing " + "; ".join(missing))


def main() -> int:
    parser = argparse.ArgumentParser(description="Run dissertation models (Stage A)")
    parser.add_argument("--model", default="all", help=f"Model id or 'all'. Choices: {', '.join(ALL_MODELS)}")
    parser.add_argument("--export", default=None, help="Output path for lab_results_bundle.json")
    args = parser.parse_args()

    cfg = load_config()
    export_path = args.export or cfg["outputs"]["bundle"]
    if not Path(export_path).is_absolute():
        export_path = str(ROOT / export_path)

    store = ResultsStore()
    model = args.model.lower().strip()

    if model == "all":
        print("Running all 13 models + H2/SHAP on pilot data...")
        run_all(store)
        validate_full_run(store)
    elif model in ALL_MODELS:
        print(f"Running model: {model}")
        run_single(model, store)
    else:
        print(f"Unknown model '{model}'. Valid: {', '.join(ALL_MODELS + ['all'])}")
        return 1

    store.save(export_path, merge_existing=model != "all")
    print(f"Saved bundle: {export_path}")
    print(f"  h1={len(store.h1)} h2={len(store.h2)} h3a={len(store.h3a)} h3b={len(store.h3b)} shap={len(store.shap)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

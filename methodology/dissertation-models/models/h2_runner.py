"""H2 dual feature-set runs + SHAP export."""
from __future__ import annotations

import numpy as np
import pandas as pd

from data.config_loader import load_config
from data.features import prepare_xy
from evaluation.calculation_manifest import shared_manifest
from evaluation.export_results import ResultsStore
from models.lstm_model import run_lstm
from models.stacked_ensemble import run_stacked_ensemble
from models.xgboost_model import run_xgboost


def _metrics(rec: dict) -> dict:
    return {k: rec.get(k) for k in ("mse", "mae", "rmse", "r2", "n_obs")}


def _run_pair_with_context(runner, train, val, test, fold, needs_val, parent_store):
    s_wo, s_wi = ResultsStore(), ResultsStore()
    s_wo.set_data_context(parent_store._data_context)
    s_wi.set_data_context(parent_store._data_context)
    s_wo._run_id = parent_store.run_id
    s_wi._run_id = parent_store.run_id
    if needs_val:
        runner(train, val, test, fold, s_wo, "traditional")
        runner(train, val, test, fold, s_wi, "with_alt_data")
    else:
        runner(train, test, fold, s_wo, "traditional")
        runner(train, test, fold, s_wi, "with_alt_data")
    without = s_wo.h1[-1] if s_wo.h1 else {}
    with_alt = s_wi.h1[-1] if s_wi.h1 else {}
    return _metrics(without), _metrics(with_alt), without.get("calculation"), with_alt.get("calculation")


def run_h2_and_shap(train, val, test, fold: str, store: ResultsStore) -> None:
    pairs = [
        ("xgboost", run_xgboost, False),
        ("lstm", run_lstm, False),
        ("stacked_ensemble", run_stacked_ensemble, True),
    ]
    for model_id, runner, needs_val in pairs:
        without, with_alt, calc_wo, calc_wi = _run_pair_with_context(
            runner, train, val, test, fold, needs_val, store
        )
        calc = {
            **shared_manifest(store._data_context, feature_set="with_alt_data"),
            "model": {
                "family": f"H2 dual feature-set comparison ({model_id})",
                "formula": "same model trained on traditional vs with_alt_data features",
                "plain_english": (
                    "Retrain the model once without alternative-data features and once with "
                    "Bloomberg News/Twitter sentiment (plus sentiment_proxy fallback), "
                    "then compare out-of-sample RMSE."
                ),
                "parameters": {"compared_model": model_id},
                "oos_unit": "ticker_month",
            },
            "traditional": calc_wo,
            "with_alt_data": calc_wi,
            "metric_definitions": {
                "rmse": "Out-of-sample RMSE of next-month excess returns",
                "delta_rmse": "with_alt_data RMSE − traditional RMSE (negative = alt data helped)",
            },
        }
        store.add_h2(model_id, without, with_alt, fold, calculation=calc)

    _shap_tree_models(train, test, store)


def _shap_tree_models(train: pd.DataFrame, test: pd.DataFrame, store: ResultsStore) -> None:
    for model_id, model_cls, cfg_key in [
        ("xgboost", __import__("xgboost", fromlist=["XGBRegressor"]).XGBRegressor, "xgboost"),
        ("random_forest", __import__("sklearn.ensemble", fromlist=["RandomForestRegressor"]).RandomForestRegressor, "random_forest"),
    ]:
        try:
            import shap
            from sklearn.preprocessing import StandardScaler

            cfg = load_config()
            p = cfg[cfg_key]
            for feature_set in ("traditional", "with_alt_data"):
                X_tr, y_tr, cols = prepare_xy(train, feature_set)
                X_te, _, _ = prepare_xy(test, feature_set)
                if X_tr.empty or not cols:
                    continue
                sc = StandardScaler()
                X_tr_s = sc.fit_transform(X_tr)
                if model_id == "xgboost":
                    model = model_cls(
                        n_estimators=p["n_estimators"],
                        max_depth=p["max_depth"],
                        learning_rate=p.get("learning_rate", 0.05),
                        random_state=42,
                        n_jobs=1,
                    )
                else:
                    model = model_cls(
                        n_estimators=p["n_estimators"],
                        max_depth=p["max_depth"],
                        random_state=42,
                        n_jobs=-1,
                    )
                model.fit(X_tr_s, y_tr)
                explainer = shap.TreeExplainer(model)
                sample = sc.transform(X_te) if not X_te.empty else X_tr_s[: min(100, len(X_tr_s))]
                sv = explainer.shap_values(sample)
                mean_abs = np.abs(sv).mean(axis=0)
                feats = [
                    {"name": cols[i], "mean_abs_shap": float(mean_abs[i])}
                    for i in range(len(cols))
                ]
                feats.sort(key=lambda x: -x["mean_abs_shap"])
                store.add_shap(model_id, feats[:15], feature_set)
        except Exception:
            continue

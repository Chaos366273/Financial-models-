"""Random Forest return predictor."""
from __future__ import annotations

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

from data.config_loader import load_config
from data.features import prepare_xy
from evaluation.artifact_writer import build_calculation
from evaluation.export_results import ResultsStore
from evaluation.predictive import predictive_metrics


def _fit_predict(model, train: pd.DataFrame, test: pd.DataFrame, feature_set: str):
    X_tr, y_tr, cols = prepare_xy(train, feature_set)
    X_te, y_te, _ = prepare_xy(test, feature_set)
    if X_tr.empty or X_te.empty:
        return None, None, cols, None, None
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)
    model.fit(X_tr_s, y_tr)
    pred = model.predict(X_te_s)
    frame = pd.DataFrame({
        "Ticker": test.loc[X_te.index, "Ticker"].astype(str).values,
        "month": pd.to_datetime(test.loc[X_te.index, "month"]).dt.strftime("%Y-%m-%d").values,
        "y_true": y_te.values,
        "y_pred": pred,
    })
    frame["error"] = frame["y_true"] - frame["y_pred"]
    frame["abs_error"] = frame["error"].abs()
    return y_te.values, pred, cols, scaler, frame


def run_random_forest(train, test, fold, store: ResultsStore, feature_set: str = "traditional"):
    cfg = load_config()
    rf_cfg = cfg["random_forest"]
    model = RandomForestRegressor(
        n_estimators=rf_cfg["n_estimators"],
        max_depth=rf_cfg["max_depth"],
        random_state=42,
        n_jobs=-1,
    )
    y, pred, cols, scaler, frame = _fit_predict(model, train, test, feature_set)
    if y is None:
        store.add_h1("random_forest", {"mse": None, "mae": None, "rmse": None, "r2": None, "n_obs": 0}, fold, feature_set)
        return None, None, cols, None
    metrics = predictive_metrics(y, pred)
    calc = build_calculation(
        run_id=store.run_id,
        model_id="random_forest",
        bucket="h1",
        data_context=store._data_context,
        frame=frame,
        feature_set=feature_set,
        artifact_key=f"random_forest__{feature_set}",
        model_meta={
            "family": "RandomForestRegressor (sklearn)",
            "formula": "StandardScaler(features) → RandomForestRegressor → target_excess_return",
            "plain_english": (
                "Train a random forest on technical, fundamental, and factor features to "
                "predict next-month excess returns; score on unseen test ticker-months."
            ),
            "parameters": {
                "n_estimators": rf_cfg["n_estimators"],
                "max_depth": rf_cfg["max_depth"],
                "feature_columns": cols,
                "feature_set": feature_set,
            },
            "oos_unit": "ticker_month",
        },
    )
    store.add_h1("random_forest", metrics, fold, feature_set, calculation=calc)
    return model, (y, pred), cols, scaler

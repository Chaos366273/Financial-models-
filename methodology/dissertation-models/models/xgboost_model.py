"""XGBoost return predictor (Lab engine).

Uses h1_clean-compatible training defaults from config when feature_set allows;
reports ranking metrics alongside RMSE/R².
"""
from __future__ import annotations

import pandas as pd
from xgboost import XGBRegressor

from data.config_loader import load_config
from data.features import prepare_xy
from evaluation.artifact_writer import build_calculation
from evaluation.export_results import ResultsStore
from evaluation.predictive import predictive_metrics, ranking_metrics


def _filter_members(df: pd.DataFrame) -> pd.DataFrame:
    if "is_member" not in df.columns:
        return df
    mask = df["is_member"]
    if mask.dtype != bool:
        mask = mask.astype(bool)
    return df.loc[mask].copy()


def _prepare_target(train: pd.DataFrame, test: pd.DataFrame, demean: bool, winsor: bool):
    frames = []
    for df in (train, test):
        d = df.copy()
        y = d["target_excess_return"].astype(float)
        if demean and "month" in d.columns:
            y = y - d.groupby("month")["target_excess_return"].transform("mean")
        d["target_model"] = y
        frames.append(d)
    train_o, test_o = frames
    if winsor:
        lo, hi = train_o["target_model"].quantile([0.01, 0.99])
        train_o["target_model"] = train_o["target_model"].clip(float(lo), float(hi))
        test_o["target_model"] = test_o["target_model"].clip(float(lo), float(hi))
    return train_o, test_o


def _fit_predict(model, train, test, feature_set: str, demean: bool, winsor: bool, members_only: bool):
    if members_only:
        train = _filter_members(train)
        test = _filter_members(test)
    train, test = _prepare_target(train, test, demean=demean, winsor=winsor)
    X_tr, y_tr, cols = prepare_xy(train, feature_set, target_col="target_model")
    X_te, _, _ = prepare_xy(test, feature_set, target_col="target_model")
    if X_tr.empty or X_te.empty:
        return None, None, cols, None, None
    # Trees: no StandardScaler
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)
    te_meta = test.loc[X_te.index]
    frame = pd.DataFrame({
        "Ticker": te_meta["Ticker"].astype(str).values,
        "month": pd.to_datetime(te_meta["month"]).dt.strftime("%Y-%m-%d").values,
        "y_true": te_meta["target_excess_return"].astype(float).values,
        "y_pred": pred,
    })
    frame["error"] = frame["y_true"] - frame["y_pred"]
    frame["abs_error"] = frame["error"].abs()
    return frame["y_true"].values, pred, cols, None, frame


def run_xgboost(train, test, fold, store: ResultsStore, feature_set: str = "traditional"):
    cfg = load_config()
    x_cfg = cfg["xgboost"]
    # H1 Lab default stays traditional for ERP comparability unless caller asks h1_clean
    members_only = bool(x_cfg.get("members_only", True))
    demean = bool(x_cfg.get("demean_within_month", True))
    winsor = bool(x_cfg.get("winsorise_y", True))

    model = XGBRegressor(
        n_estimators=int(x_cfg.get("n_estimators", 400)),
        max_depth=int(x_cfg.get("max_depth", 4)),
        learning_rate=float(x_cfg.get("learning_rate", 0.03)),
        min_child_weight=float(x_cfg.get("min_child_weight", 10)),
        subsample=float(x_cfg.get("subsample", 0.8)),
        colsample_bytree=float(x_cfg.get("colsample_bytree", 0.8)),
        random_state=42,
        n_jobs=2,
        verbosity=0,
    )
    y, pred, cols, scaler, frame = _fit_predict(
        model, train, test, feature_set, demean=demean, winsor=winsor, members_only=members_only
    )
    if y is None:
        store.add_h1("xgboost", {"mse": None, "mae": None, "rmse": None, "r2": None, "n_obs": 0}, fold, feature_set)
        return None, None, cols, None

    metrics = predictive_metrics(y, pred)
    rank = ranking_metrics(frame)
    metrics.update({
        "ic_mean": rank["ic_mean"],
        "ic_std": rank["ic_std"],
        "ic_ir": rank["ic_ir"],
        "long_short_mean": rank["long_short_mean"],
        "long_short_std": rank["long_short_std"],
        "rank_n_months": rank["n_months"],
    })
    calc = build_calculation(
        run_id=store.run_id,
        model_id="xgboost",
        bucket="h1",
        data_context=store._data_context,
        frame=frame,
        feature_set=feature_set,
        artifact_key=f"xgboost__{feature_set}",
        model_meta={
            "family": "XGBRegressor (xgboost)",
            "formula": (
                "is_member filter → month-demean+winsorise(y) → XGBRegressor(features) → "
                "raw excess-return metrics + Spearman IC / long-short"
            ),
            "plain_english": (
                "Gradient-boosted trees predict next-month excess returns on FTSE members; "
                "RMSE/R² are H1 level metrics; IC and long-short measure ranking usefulness."
            ),
            "parameters": {
                "n_estimators": x_cfg.get("n_estimators", 400),
                "max_depth": x_cfg.get("max_depth", 4),
                "learning_rate": x_cfg.get("learning_rate", 0.03),
                "min_child_weight": x_cfg.get("min_child_weight", 10),
                "subsample": x_cfg.get("subsample", 0.8),
                "colsample_bytree": x_cfg.get("colsample_bytree", 0.8),
                "members_only": members_only,
                "demean_within_month": demean,
                "winsorise_y": winsor,
                "feature_columns": cols,
                "feature_set": feature_set,
            },
            "oos_unit": "ticker_month",
        },
    )
    store.add_h1("xgboost", metrics, fold, feature_set, calculation=calc)
    return model, (y, pred), cols, scaler

"""Stacked ensemble: XGBoost + LSTM → PPO meta-learner (RF removed)."""
from __future__ import annotations

import pandas as pd
from sklearn.preprocessing import StandardScaler

from data.config_loader import load_config
from data.features import prepare_xy
from evaluation.artifact_writer import build_calculation
from evaluation.export_results import ResultsStore
from evaluation.predictive import predictive_metrics
from models.lstm_model import _build_sequences, fit_predict_lstm
from models.ppo_meta import fallback_blend, predict_ppo_meta, sb3_available, train_ppo_meta


def _pred_xgb_keyed(train: pd.DataFrame, df: pd.DataFrame, feature_set: str) -> pd.DataFrame | None:
    """Fit XGB on train, predict on df; keyed by (Ticker, month)."""
    from xgboost import XGBRegressor

    cfg = load_config()
    xgb = XGBRegressor(
        n_estimators=cfg["xgboost"]["n_estimators"],
        max_depth=cfg["xgboost"]["max_depth"],
        learning_rate=cfg["xgboost"]["learning_rate"],
        random_state=42,
        n_jobs=1,
    )
    X_tr, y_tr, _ = prepare_xy(train, feature_set)
    if X_tr.empty:
        return None
    sc = StandardScaler()
    X_tr_s = sc.fit_transform(X_tr)
    xgb.fit(X_tr_s, y_tr)

    X_df, y_df, _ = prepare_xy(df, feature_set)
    if X_df.empty:
        return None
    X_df_s = sc.transform(X_df)
    idx = X_df.index
    return pd.DataFrame({
        "Ticker": df.loc[idx, "Ticker"].astype(str).values,
        "month": pd.to_datetime(df.loc[idx, "month"]).dt.strftime("%Y-%m-%d").values,
        "y": y_df.values,
        "xgb_pred": xgb.predict(X_df_s),
    })


def _lstm_frame(train, df, feature_set: str, seq_len: int, hidden: int, epochs: int, batch: int) -> pd.DataFrame | None:
    X_tr, y_tr, _, _, _ = _build_sequences(train, feature_set, seq_len)
    X_df, _, _, keys_df, _ = _build_sequences(df, feature_set, seq_len)
    if X_tr is None or X_df is None:
        return None
    pred = fit_predict_lstm(X_tr, y_tr, X_df, hidden, epochs, batch)
    tickers, months = zip(*keys_df)
    return pd.DataFrame({"Ticker": list(tickers), "month": list(months), "lstm_pred": pred})


def run_stacked_ensemble(train, val, test, fold, store: ResultsStore, feature_set: str = "traditional"):
    cfg = load_config()
    seq_len = int(cfg["lstm"]["seq_len"])
    hidden = int(cfg["lstm"]["hidden_size"])
    epochs = int(cfg["lstm"]["epochs"])
    batch = int(cfg["lstm"]["batch_size"])

    xgb_val = _pred_xgb_keyed(train, val, feature_set)
    xgb_test = _pred_xgb_keyed(train, test, feature_set)
    if xgb_val is None or xgb_test is None:
        store.add_h1("stacked_ensemble", {"mse": None, "mae": None, "rmse": None, "r2": None, "n_obs": 0}, fold, feature_set)
        return

    lstm_val = _lstm_frame(train, val, feature_set, seq_len, hidden, epochs, batch)
    lstm_test = _lstm_frame(train, test, feature_set, seq_len, hidden, epochs, batch)

    meta_val = xgb_val.merge(lstm_val, on=["Ticker", "month"], how="left") if lstm_val is not None else xgb_val.copy()
    meta_test = xgb_test.merge(lstm_test, on=["Ticker", "month"], how="left") if lstm_test is not None else xgb_test.copy()

    use_lstm = (
        lstm_val is not None
        and lstm_test is not None
        and meta_val["lstm_pred"].notna().sum() >= 5
        and meta_test["lstm_pred"].notna().sum() >= 1
    )

    ppo_model = train_ppo_meta(meta_val, use_lstm, cfg) if len(meta_val) >= 5 else None
    if ppo_model is not None:
        pred_test, blend_stats = predict_ppo_meta(ppo_model, meta_test, use_lstm)
        meta_method = "ppo"
    else:
        pred_test = fallback_blend(meta_test, use_lstm)
        blend_stats = None
        meta_method = "fallback_equal_weight" if use_lstm else "xgb_only"
        if not sb3_available():
            print(f"[info] stacked_ensemble ({feature_set}): stable-baselines3 not installed — using {meta_method}")

    y_test = meta_test["y"].values
    metrics = predictive_metrics(y_test, pred_test)
    frame = meta_test.copy()
    frame = frame.rename(columns={"y": "y_true"})
    frame["y_pred"] = pred_test
    frame["error"] = frame["y_true"] - frame["y_pred"]
    frame["abs_error"] = frame["error"].abs()

    preview_cols = ["Ticker", "month", "y_true", "xgb_pred"]
    if use_lstm:
        preview_cols.append("lstm_pred")
    preview_cols += ["y_pred", "error", "abs_error"]

    ppo_params = dict(cfg.get("ppo_meta", {}))
    if blend_stats:
        ppo_params["blend_stats"] = blend_stats

    calc = build_calculation(
        run_id=store.run_id,
        model_id="stacked_ensemble",
        bucket="h1",
        data_context=store._data_context,
        frame=frame,
        feature_set=feature_set,
        artifact_key=f"stacked_ensemble__{feature_set}",
        model_meta={
            "family": "Stacked ensemble (XGBoost + LSTM → PPO meta)",
            "formula": "PPO blend(XGB_pred, LSTM_pred) → target_excess_return",
            "plain_english": (
                "Supervised XGBoost and LSTM produce return forecasts; a PPO agent "
                "trained on the validation window learns how to blend them for the test window."
            ),
            "parameters": {
                "supervised_legs": ["xgboost", "lstm"] if use_lstm else ["xgboost"],
                "meta_learner": meta_method,
                "used_lstm": bool(use_lstm),
                "ppo_config": ppo_params,
                "feature_set": feature_set,
                "github_private_bridge": "data/export_ppo_forecasts.py",
            },
            "oos_unit": "ticker_month",
            "caveats": [
                "PPO meta is fit on validation forecasts (not train) to reduce leakage.",
                "LSTM leg omitted when per-ticker history is too short.",
                "Full daily trading PPO (TG5) consumes exported forecasts as obs channels.",
            ],
        },
        preview_columns=preview_cols,
    )
    store.add_h1("stacked_ensemble", metrics, fold, feature_set, calculation=calc)

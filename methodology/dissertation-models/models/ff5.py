"""FF5 OLS benchmark — fit on train, evaluate out-of-sample on test."""
from __future__ import annotations

import pandas as pd
import statsmodels.api as sm

from data.config_loader import load_config
from data.features import FACTOR_COLS
from evaluation.artifact_writer import build_calculation
from evaluation.export_results import ResultsStore
from evaluation.predictive import predictive_metrics


def run_ff5(train: pd.DataFrame, test: pd.DataFrame, fold: str, store: ResultsStore) -> None:
    cols = [c for c in FACTOR_COLS if c in train.columns and c in test.columns]
    tr = train.dropna(subset=cols + ["target_excess_return"]).copy()
    te = test.dropna(subset=cols + ["target_excess_return"]).copy()
    if tr.empty or te.empty or len(cols) < 2:
        store.add_h1("ff5", {"mse": None, "mae": None, "rmse": None, "r2": None, "n_obs": 0}, fold)
        return
    X_tr = sm.add_constant(tr[cols], has_constant="add")
    y_tr = tr["target_excess_return"]
    model = sm.OLS(y_tr, X_tr).fit()
    X_te = sm.add_constant(te[cols], has_constant="add")
    X_te = X_te.reindex(columns=X_tr.columns, fill_value=0.0)
    pred = model.predict(X_te)
    metrics = predictive_metrics(te["target_excess_return"], pred)

    frame = te[["Ticker", "month"] + cols].copy()
    frame["month"] = pd.to_datetime(frame["month"]).dt.strftime("%Y-%m-%d")
    frame["y_true"] = te["target_excess_return"].values
    frame["y_pred"] = pred.values
    frame["error"] = frame["y_true"] - frame["y_pred"]
    frame["abs_error"] = frame["error"].abs()

    coeffs = {str(k): float(v) for k, v in model.params.items()}
    calc = build_calculation(
        run_id=store.run_id,
        model_id="ff5",
        bucket="h1",
        data_context=store._data_context,
        frame=frame,
        model_meta={
            "family": "OLS (statsmodels)",
            "formula": "target_excess_return ~ const + MKT_RF + SMB + HML + RMW + CMA",
            "plain_english": (
                "Estimate Fama-French five-factor loadings on the train window, "
                "then predict next-month excess returns on unseen test ticker-months."
            ),
            "parameters": {
                "factor_columns": cols,
                "coefficients": coeffs,
                "n_train": int(len(tr)),
                "risk_free_rate": float(load_config()["data"].get("risk_free_rate", 0.0)),
            },
            "oos_unit": "ticker_month",
            "caveats": [
                "FF5 factors here are pilot cross-sectional proxies, not Kenneth French official factors.",
            ],
        },
        preview_columns=["Ticker", "month", "y_true", "y_pred", "error", "abs_error"] + cols,
    )
    store.add_h1("ff5", metrics, fold, calculation=calc)

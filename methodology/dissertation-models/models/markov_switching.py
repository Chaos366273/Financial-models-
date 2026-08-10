"""Markov-switching regression — fit on pre-test market series, OOS regime forecast."""
from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

from evaluation.artifact_writer import build_calculation
from evaluation.export_results import ResultsStore
from evaluation.predictive import predictive_metrics


def run_markov_switching(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    fold: str,
    store: ResultsStore,
) -> None:
    """Fit 2-regime Markov regression on train+val MKT_RF; forecast test months OOS.

    Forecast uses filtered regime probabilities rolled forward with the estimated
    transition matrix and regime-conditional means (not a flat in-sample constant).
    """
    hist = pd.concat([train, val], ignore_index=True)
    mkt_hist = hist.groupby("month")["MKT_RF"].mean().dropna().sort_index()
    mkt_test = test.groupby("month")["MKT_RF"].mean().dropna().sort_index()

    if len(mkt_hist) < 24 or mkt_test.empty:
        store.add_h1("markov_switching", {"mse": None, "mae": None, "rmse": None, "r2": None, "n_obs": 0}, fold)
        return

    try:
        mod = MarkovRegression(mkt_hist.values, k_regimes=2, trend="c", switching_variance=True)
        res = mod.fit(disp=False, maxiter=200)

        params = res.params
        try:
            mu = np.array([float(params[f"const[{i}]"]) for i in range(2)])
        except Exception:
            sm_prob = np.asarray(res.smoothed_marginal_probabilities)
            mu = np.array([
                float(np.average(mkt_hist.values, weights=sm_prob[:, i] + 1e-12))
                for i in range(2)
            ])

        P = np.asarray(res.regime_transition).reshape(2, 2)
        if P.shape != (2, 2):
            P = np.asarray(res.regime_transition)

        try:
            prob = np.asarray(res.filtered_marginal_probabilities)[-1].astype(float)
        except Exception:
            prob = np.asarray(res.smoothed_marginal_probabilities)[-1].astype(float)
        if prob.ndim > 1:
            prob = prob.flatten()[:2]
        prob = prob / (prob.sum() + 1e-12)

        preds = []
        regime_rows = []
        for month, actual in mkt_test.items():
            pred = float(prob @ mu)
            preds.append(pred)
            regime_rows.append({
                "month": pd.Timestamp(month).strftime("%Y-%m-%d"),
                "mkt_actual": float(actual),
                "mkt_pred": pred,
                "y_true": float(actual),
                "y_pred": pred,
                "error": float(actual) - pred,
                "abs_error": abs(float(actual) - pred),
                "regime_prob_0": float(prob[0]),
                "regime_prob_1": float(prob[1]),
            })
            prob = prob @ P
            prob = prob / (prob.sum() + 1e-12)

        metrics = predictive_metrics(mkt_test.values, np.array(preds))
        frame = pd.DataFrame(regime_rows)
        calc = build_calculation(
            run_id=store.run_id,
            model_id="markov_switching",
            bucket="h1",
            data_context=store._data_context,
            frame=frame,
            model_meta={
                "family": "MarkovRegression (statsmodels)",
                "formula": "MKT_RF_t ~ regime-dependent constant; 2 regimes, switching variance",
                "plain_english": (
                    "Estimate bull/bear-like market regimes on train+validation market returns, "
                    "then forecast the market factor on unseen test months."
                ),
                "parameters": {
                    "k_regimes": 2,
                    "switching_variance": True,
                    "regime_means": [float(x) for x in mu],
                    "transition_matrix": [[float(x) for x in row] for row in np.asarray(P)],
                    "n_hist_months": int(len(mkt_hist)),
                },
                "oos_unit": "market_month",
                "target_override": {
                    "name": "MKT_RF",
                    "unit": "monthly_market_return",
                    "formula": "equal-weight mean monthly equity return (pilot proxy market factor)",
                    "plain_english": (
                        "This model scores MARKET-LEVEL returns (MKT_RF), not per-stock "
                        "target_excess_return. n_obs equals test months, not ticker-months."
                    ),
                },
                "caveats": [
                    "Market-level OOS unit — not comparable 1:1 with FF5/XGBoost ticker-month RMSE.",
                ],
            },
            preview_columns=[
                "month", "mkt_actual", "mkt_pred", "error", "abs_error",
                "regime_prob_0", "regime_prob_1",
            ],
        )
        store.add_h1("markov_switching", metrics, fold, calculation=calc)
    except Exception as e:
        print(f"[warn] markov_switching failed: {e}")
        store.add_h1("markov_switching", {"mse": None, "mae": None, "rmse": None, "r2": None, "n_obs": 0}, fold)

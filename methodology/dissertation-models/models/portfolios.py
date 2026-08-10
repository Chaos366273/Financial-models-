"""Portfolio strategies H3a."""
from __future__ import annotations

import numpy as np
import pandas as pd
from pypfopt import expected_returns, risk_models
from pypfopt.black_litterman import BlackLittermanModel, market_implied_prior_returns
from pypfopt.efficient_frontier import EfficientFrontier

from data.config_loader import load_config
from evaluation.artifact_writer import build_calculation
from evaluation.export_results import ResultsStore
from evaluation.portfolio_metrics import portfolio_metrics

H3A_IDS = ["equal_weight", "mean_variance", "min_variance", "black_litterman", "ml_black_litterman"]


def _returns_matrix(panel: pd.DataFrame, months: pd.DatetimeIndex) -> pd.DataFrame:
    sub = panel[panel["month"].isin(months)]
    wide = sub.pivot_table(index="month", columns="Ticker", values="monthly_return")
    wide = wide.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-0.99, 2.0)
    return wide.dropna(axis=1, how="all")


def _simulate_weights(rets: pd.DataFrame, weights: np.ndarray) -> pd.Series:
    w = np.asarray(weights, dtype=float)
    if w.sum() <= 0:
        w = np.ones(len(w)) / len(w)
    else:
        w = w / w.sum()
    port = rets.values @ w
    return pd.Series(port, index=rets.index)


def _market_caps(panel: pd.DataFrame, tickers: list[str]) -> dict:
    if "market_cap" not in panel.columns:
        return {t: 1.0 for t in tickers}
    latest = (
        panel[panel["Ticker"].isin(tickers)]
        .sort_values("month")
        .groupby("Ticker")["market_cap"]
        .last()
    )
    return {t: (float(latest[t]) if t in latest.index and pd.notna(latest[t]) else 1.0) for t in tickers}


def _emit_strategy(
    store: ResultsStore,
    strategy_id: str,
    port_rets: pd.Series,
    weights: np.ndarray,
    tickers: list[str],
    period: str,
    rf: float,
    formula: str,
    plain: str,
    params: dict | None = None,
) -> None:
    metrics = portfolio_metrics(port_rets, rf)
    frame = pd.DataFrame({
        "month": pd.to_datetime(port_rets.index).strftime("%Y-%m-%d"),
        "port_return": port_rets.values,
        "actual": port_rets.values,
        "forecast": port_rets.values,
        "abs_error": 0.0,
    })
    weight_map = {tickers[i]: float(weights[i]) for i in range(len(tickers))}
    calc = build_calculation(
        run_id=store.run_id,
        model_id=strategy_id,
        bucket="h3a",
        data_context=store._data_context,
        frame=frame,
        error_col="abs_error",
        model_meta={
            "family": "Portfolio optimiser / benchmark",
            "formula": formula,
            "plain_english": plain,
            "parameters": {
                "weights": weight_map,
                "n_tickers": len(tickers),
                "estimation_window": "train+validation (pre-test only)",
                **(params or {}),
            },
            "oos_unit": "portfolio_month",
            "target_override": {
                "name": "portfolio_monthly_return",
                "unit": "monthly_portfolio_return",
                "formula": "sum_i w_i * monthly_return_{i,t}",
                "plain_english": (
                    "Weights are estimated from pre-test history, then applied to "
                    "unseen test-month stock returns to form a portfolio return series."
                ),
            },
            "caveats": [
                "Static weights estimated once from train+validation — no monthly re-optimisation in Stage A.",
            ],
        },
        preview_columns=["month", "port_return"],
    )
    store.add_h3a(strategy_id, metrics, period, calculation=calc)


def run_portfolios(
    hist_panel: pd.DataFrame,
    test: pd.DataFrame,
    fold: str,
    store: ResultsStore,
    ml_forecasts: pd.DataFrame | None = None,
    strategy_ids: set[str] | None = None,
) -> None:
    """hist_panel must contain only data available before the test period starts
    (e.g. train+val) — mu/covariance for portfolio construction are estimated from
    it, so including test months here would leak future returns into the weights
    used to "backtest" that same period.
    """
    selected = set(strategy_ids or H3A_IDS)
    unknown = selected.difference(H3A_IDS)
    if unknown:
        raise ValueError(f"Unknown H3a strategy ids: {', '.join(sorted(unknown))}")
    months = sorted(test["month"].unique())
    if len(months) < 3:
        for sid in H3A_IDS:
            if sid not in selected:
                continue
            store.add_h3a(sid, {"sharpe": 0.0, "sortino": 0.0, "max_drawdown": 0.0, "net_return_annual": 0.0}, fold)
        return

    test_rets = _returns_matrix(test, months)
    tickers = list(test_rets.columns)
    period = f"{months[0].date()}\u2013{months[-1].date()}"
    rf = float(load_config()["data"]["risk_free_rate"])

    hist = _returns_matrix(hist_panel, hist_panel["month"].unique())
    common = [t for t in tickers if t in hist.columns]
    if len(common) < 2:
        print("[warn] portfolios: fewer than 2 tickers with pre-test history; equal-weight on full test universe")
        common = tickers
        test_common = test_rets
        hist = hist.reindex(columns=common).fillna(0.0) if len(hist.columns) else test_rets
    else:
        hist = hist[common]
        test_common = test_rets[common]

    n = len(common)
    ew = np.ones(n) / n
    if "equal_weight" in selected:
        _emit_strategy(
            store, "equal_weight", _simulate_weights(test_common, ew), ew, common, period, rf,
            formula="w_i = 1/N",
            plain="Equal-weight benchmark: each ticker gets the same weight; backtested on unseen test months.",
        )
    if selected == {"equal_weight"}:
        return

    def _fallback(sid: str) -> None:
        if sid in selected:
            _emit_strategy(
                store, sid, _simulate_weights(test_common, ew), ew, common, period, rf,
                formula="fallback to equal-weight after optimiser failure",
                plain=f"{sid} optimiser failed; equal-weight fallback used for this run.",
                params={"fallback": True},
            )

    if len(common) < 2:
        for sid in ["mean_variance", "min_variance", "black_litterman", "ml_black_litterman"]:
            _fallback(sid)
        return

    try:
        mu = expected_returns.mean_historical_return(hist, returns_data=True, frequency=12)
        S = risk_models.sample_cov(hist, returns_data=True, frequency=12)
    except Exception as e:
        print(f"[warn] portfolios: mu/covariance estimation failed: {e}")
        for sid in ["mean_variance", "min_variance", "black_litterman", "ml_black_litterman"]:
            _fallback(sid)
        return

    if "mean_variance" in selected:
        try:
            ef = EfficientFrontier(mu, S)
            ef.max_sharpe()
            w_mv = np.array([ef.clean_weights().get(t, 0.0) for t in common])
            _emit_strategy(
                store, "mean_variance", _simulate_weights(test_common, w_mv), w_mv, common, period, rf,
                formula="max Sharpe on historical μ and Σ (train+val)",
                plain="Markowitz mean-variance portfolio maximising Sharpe using pre-test μ/Σ, then held through the test window.",
            )
        except Exception as e:
            print(f"[warn] mean_variance failed: {e}")
            _fallback("mean_variance")

    if "min_variance" in selected:
        try:
            ef2 = EfficientFrontier(mu, S)
            ef2.min_volatility()
            w_min = np.array([ef2.clean_weights().get(t, 0.0) for t in common])
            _emit_strategy(
                store, "min_variance", _simulate_weights(test_common, w_min), w_min, common, period, rf,
                formula="min volatility on historical Σ (train+val)",
                plain="Minimum-variance portfolio from pre-test covariance, evaluated on unseen test months.",
            )
        except Exception as e:
            print(f"[warn] min_variance failed: {e}")
            _fallback("min_variance")

    if "black_litterman" in selected:
        try:
            mcaps = pd.Series(_market_caps(hist_panel, common))
            bl_ret = market_implied_prior_returns(mcaps, risk_aversion=1, cov_matrix=S, risk_free_rate=rf)
            ef3 = EfficientFrontier(bl_ret, S)
            ef3.max_sharpe()
            w_bl = np.array([ef3.clean_weights().get(t, 0.0) for t in common])
            _emit_strategy(
                store, "black_litterman", _simulate_weights(test_common, w_bl), w_bl, common, period, rf,
                formula="max Sharpe on market-implied prior returns (views-free baseline)",
                plain="Views-free Black-Litterman baseline using market-implied equilibrium returns.",
            )
        except Exception as e:
            print(f"[warn] black_litterman failed: {e}")
            _fallback("black_litterman")

    if "ml_black_litterman" in selected:
        try:
            views = {}
            if ml_forecasts is not None and not ml_forecasts.empty:
                for t in common:
                    sub = ml_forecasts[ml_forecasts["Ticker"] == t]["pred"]
                    if not sub.empty:
                        views[t] = float(sub.mean())
            if not views:
                raise ValueError("no ML views available for ml_black_litterman")
            mcaps = _market_caps(hist_panel, common)
            bl_ml = BlackLittermanModel(S, pi="market", market_caps=mcaps, absolute_views=views, risk_free_rate=rf)
            bl_ml_ret = bl_ml.bl_returns()
            ef4 = EfficientFrontier(bl_ml_ret, S)
            ef4.max_sharpe()
            w_ml = np.array([ef4.clean_weights().get(t, 0.0) for t in common])
            _emit_strategy(
                store, "ml_black_litterman", _simulate_weights(test_common, w_ml), w_ml, common, period, rf,
                formula="Black-Litterman with XGBoost absolute views → max Sharpe",
                plain="Black-Litterman posterior using ML return forecasts as absolute views, then max-Sharpe weights.",
                params={"n_views": len(views)},
            )
        except Exception as e:
            print(f"[warn] ml_black_litterman failed: {e}")
            _fallback("ml_black_litterman")

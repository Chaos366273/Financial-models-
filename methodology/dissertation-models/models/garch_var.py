"""GARCH(1,1)-VaR — rolling one-step OOS backtest on test window."""
from __future__ import annotations

import numpy as np
import pandas as pd
from arch import arch_model

from evaluation.artifact_writer import build_calculation
from evaluation.export_results import ResultsStore


def _market_daily_returns(daily: pd.DataFrame) -> pd.Series:
    d = daily.sort_values(["Ticker", "date"]).copy()
    d["ret"] = d.groupby("Ticker")["close"].pct_change()
    return d.groupby("date")["ret"].mean().dropna()


def _rolling_garch_var(
    mkt: pd.Series,
    test_start: pd.Timestamp,
    z: float = 1.645,
    refit_every: int = 20,
) -> tuple[float, float, float, pd.DataFrame]:
    """Fit GARCH on pre-test data, roll one-step VaR over all test days.

    Returns (var_95_mean, violation_rate, cvar_95_mean, detail_frame).
    """
    hist = mkt[mkt.index < test_start]
    test = mkt[mkt.index >= test_start]
    empty = pd.DataFrame()
    if len(hist) < 100 or test.empty:
        return float("nan"), float("nan"), float("nan"), empty

    vars_list: list[float] = []
    actuals: list[float] = []
    dates: list = []
    window = hist.copy()
    res = None
    steps_since_fit = refit_every

    for dt, actual in test.items():
        if res is None or steps_since_fit >= refit_every:
            try:
                am = arch_model(window * 100, vol="Garch", p=1, q=1, rescale=False)
                res = am.fit(disp="off", show_warning=False)
                steps_since_fit = 0
            except Exception as e:
                print(f"[warn] garch_var refit failed at {dt}: {e}")
                if res is None:
                    continue
        try:
            fcast = res.forecast(horizon=1, reindex=False)
            sigma = float(fcast.variance.values[-1, 0] ** 0.5 / 100.0)
            var_t = -abs(sigma * z)
        except Exception:
            sigma = float(window.iloc[-60:].std() or 0.01)
            var_t = -abs(sigma * z)

        vars_list.append(var_t)
        actuals.append(float(actual))
        dates.append(pd.Timestamp(dt))
        window = pd.concat([window, pd.Series([actual], index=[dt])])
        steps_since_fit += 1

    if not vars_list:
        return float("nan"), float("nan"), float("nan"), empty

    var_arr = np.array(vars_list)
    act_arr = np.array(actuals)
    violations = float((act_arr < var_arr).mean())
    var_95 = float(np.mean(var_arr))
    cvar_95 = float(np.mean(var_arr) * 1.2)
    frame = pd.DataFrame({
        "date": [d.strftime("%Y-%m-%d") for d in dates],
        "actual": act_arr,
        "forecast": var_arr,
        "var_95": var_arr,
        "violation": (act_arr < var_arr).astype(int),
        "abs_error": np.abs(act_arr - var_arr),
    })
    return var_95, violations, cvar_95, frame


def run_garch_var(
    daily: pd.DataFrame,
    fold: str,
    store: ResultsStore,
    test_start: pd.Timestamp | None = None,
) -> None:
    mkt = _market_daily_returns(daily)
    if test_start is None:
        test_start = mkt.index[int(len(mkt) * 0.8)]
    try:
        var_95, violations, cvar_95, frame = _rolling_garch_var(mkt, test_start)
        if np.isnan(var_95):
            store.add_h3b("garch_var", {"var_95": None, "violation_rate": None, "cvar_95": None}, fold)
            return
        metrics = {"var_95": var_95, "violation_rate": violations, "cvar_95": cvar_95}
        calc = build_calculation(
            run_id=store.run_id,
            model_id="garch_var",
            bucket="h3b",
            data_context=store._data_context,
            frame=frame,
            model_meta={
                "family": "GARCH(1,1) VaR (arch)",
                "formula": "VaR_95,t = -1.645 * σ̂_t  where σ̂_t from GARCH(1,1)",
                "plain_english": (
                    "Fit GARCH on pre-test daily market returns, then roll one-step 95% VaR "
                    "forecasts across the unseen test days and count breaches."
                ),
                "parameters": {"p": 1, "q": 1, "z": 1.645, "refit_every": 20},
                "oos_unit": "market_day",
                "target_override": {
                    "name": "daily_market_return",
                    "unit": "daily_return",
                    "formula": "equal-weight mean daily close-to-close return across tickers",
                    "plain_english": (
                        "Scores risk forecasts, not return prediction RMSE. A violation occurs "
                        "when the realised daily market return is worse than the VaR forecast."
                    ),
                },
                "caveats": [
                    "CVaR is approximated as mean(VaR) × 1.2 on this Stage A pilot setup.",
                ],
            },
            preview_columns=["date", "actual", "var_95", "violation", "abs_error"],
        )
        store.add_h3b("garch_var", metrics, fold, calculation=calc)
    except Exception as e:
        print(f"[warn] garch_var failed: {e}")
        store.add_h3b("garch_var", {"var_95": None, "violation_rate": None, "cvar_95": None}, fold)

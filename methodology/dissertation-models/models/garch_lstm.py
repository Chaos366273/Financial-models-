"""GARCH-LSTM hybrid VaR — train on pre-test, roll OOS over test window."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from arch import arch_model

from evaluation.artifact_writer import build_calculation
from evaluation.export_results import ResultsStore
from models.garch_var import _market_daily_returns


class VolLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(2, 16, batch_first=True)
        self.fc = nn.Linear(16, 1)

    def forward(self, x):
        o, _ = self.lstm(x)
        return self.fc(o[:, -1, :]).squeeze(-1)


def _fit_garch_paths(rets: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Return aligned (resid, cond_vol) arrays for the series."""
    am = arch_model(rets.values * 100, vol="Garch", p=1, q=1, rescale=False)
    res = am.fit(disp="off", show_warning=False)
    cond_vol = np.asarray(res.conditional_volatility) / 100.0
    mu = float(res.params.get("mu", 0.0)) / 100.0
    resid = rets.values - mu
    n = min(len(resid), len(cond_vol))
    return resid[:n], cond_vol[:n]


def _emit(store: ResultsStore, metrics: dict, frame: pd.DataFrame, fold: str, note: str = "") -> None:
    calc = build_calculation(
        run_id=store.run_id,
        model_id="garch_lstm",
        bucket="h3b",
        data_context=store._data_context,
        frame=frame,
        model_meta={
            "family": "GARCH-LSTM hybrid VaR",
            "formula": "LSTM(resid, cond_vol) → σ̂_t → VaR_95 = -1.645 * σ̂_t",
            "plain_english": (
                "Train an LSTM on GARCH residuals/volatility from the pre-test window, "
                "then roll one-step VaR forecasts over unseen test days."
            ),
            "parameters": {"seq_len": 10, "z": 1.645, "note": note or None},
            "oos_unit": "market_day",
            "target_override": {
                "name": "daily_market_return",
                "unit": "daily_return",
                "formula": "equal-weight mean daily close-to-close return across tickers",
                "plain_english": (
                    "Risk forecast evaluation: compare realised daily market returns to VaR. "
                    "violation=1 when actual < VaR."
                ),
            },
            "caveats": [
                "CVaR is approximated as mean(VaR) × 1.2 on this Stage A pilot setup.",
            ],
        },
        preview_columns=["date", "actual", "var_95", "violation", "abs_error"],
    )
    store.add_h3b("garch_lstm", metrics, fold, calculation=calc)


def run_garch_lstm(
    daily: pd.DataFrame,
    fold: str,
    store: ResultsStore,
    test_start: pd.Timestamp | None = None,
) -> None:
    mkt = _market_daily_returns(daily)
    if test_start is None:
        test_start = mkt.index[int(len(mkt) * 0.8)]

    hist = mkt[mkt.index < test_start]
    test = mkt[mkt.index >= test_start]
    if len(hist) < 120 or test.empty:
        store.add_h3b("garch_lstm", {"var_95": None, "violation_rate": None, "cvar_95": None}, fold)
        return

    try:
        resid, cond_vol = _fit_garch_paths(hist)
        seq = 10
        X, y = [], []
        for i in range(seq, len(resid)):
            X.append(np.column_stack([resid[i - seq : i], cond_vol[i - seq : i]]).astype(np.float32))
            y.append(float(cond_vol[i]))
        if len(X) < 20:
            var_list, act_list, dates = [], [], []
            window = hist.copy()
            for dt, actual in test.items():
                sigma = float(window.iloc[-60:].std() or 0.01)
                var_list.append(-abs(sigma * 1.645))
                act_list.append(float(actual))
                dates.append(pd.Timestamp(dt).strftime("%Y-%m-%d"))
                window = pd.concat([window, pd.Series([actual], index=[dt])])
            var_arr = np.array(var_list)
            act_arr = np.array(act_list)
            frame = pd.DataFrame({
                "date": dates,
                "actual": act_arr,
                "forecast": var_arr,
                "var_95": var_arr,
                "violation": (act_arr < var_arr).astype(int),
                "abs_error": np.abs(act_arr - var_arr),
            })
            _emit(
                store,
                {
                    "var_95": float(var_arr.mean()),
                    "violation_rate": float((act_arr < var_arr).mean()),
                    "cvar_95": float(var_arr.mean() * 1.2),
                },
                frame,
                fold,
                note="historical-vol fallback (insufficient GARCH-LSTM training sequences)",
            )
            return

        X_t = torch.tensor(np.stack(X))
        y_t = torch.tensor(np.array(y, dtype=np.float32))
        model = VolLSTM()
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        for _ in range(20):
            opt.zero_grad()
            pred = model(X_t)
            loss = nn.functional.mse_loss(pred, y_t)
            loss.backward()
            opt.step()

        vars_list, actuals, dates = [], [], []
        resid_buf = list(resid[-seq:])
        vol_buf = list(cond_vol[-seq:])

        model.eval()
        for dt, actual in test.items():
            x_in = np.column_stack([resid_buf[-seq:], vol_buf[-seq:]]).astype(np.float32)
            with torch.no_grad():
                vol_pred = float(model(torch.tensor(x_in[None, ...])).item())
            realised = float(np.std(resid_buf[-20:]) or 0.0)
            vol_pred = max(abs(vol_pred), realised, 1e-4)
            var_t = -abs(vol_pred * 1.645)
            vars_list.append(var_t)
            actuals.append(float(actual))
            dates.append(pd.Timestamp(dt).strftime("%Y-%m-%d"))
            resid_buf.append(float(actual))
            vol_buf.append(0.9 * vol_buf[-1] + 0.1 * abs(float(actual)))

        var_arr = np.array(vars_list)
        act_arr = np.array(actuals)
        frame = pd.DataFrame({
            "date": dates,
            "actual": act_arr,
            "forecast": var_arr,
            "var_95": var_arr,
            "violation": (act_arr < var_arr).astype(int),
            "abs_error": np.abs(act_arr - var_arr),
        })
        _emit(
            store,
            {
                "var_95": float(var_arr.mean()),
                "violation_rate": float((act_arr < var_arr).mean()),
                "cvar_95": float(var_arr.mean() * 1.2),
            },
            frame,
            fold,
        )
    except Exception as e:
        print(f"[warn] garch_lstm failed: {e}")
        store.add_h3b("garch_lstm", {"var_95": None, "violation_rate": None, "cvar_95": None}, fold)

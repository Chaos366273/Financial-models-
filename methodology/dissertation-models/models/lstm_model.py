"""LSTM return predictor (PyTorch)."""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from data.config_loader import load_config
from data.features import get_feature_columns
from evaluation.artifact_writer import build_calculation
from evaluation.export_results import ResultsStore
from evaluation.predictive import predictive_metrics


class LSTMRegressor(nn.Module):
    def __init__(self, n_features: int, hidden: int, dropout: float = 0.0):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, batch_first=True)
        # 1-layer LSTM: apply dropout on the last hidden state (not nn.LSTM dropout)
        self.drop = nn.Dropout(p=float(dropout)) if float(dropout) > 0 else nn.Identity()
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        h = self.drop(out[:, -1, :])
        return self.fc(h).squeeze(-1)


def _months_contiguous(period_index: pd.PeriodIndex) -> bool:
    """True if each consecutive pair differs by exactly 1 month."""
    if len(period_index) <= 1:
        return True
    for j in range(1, len(period_index)):
        if (period_index[j] - period_index[j - 1]).n != 1:
            return False
    return True


def _build_sequences(
    df: pd.DataFrame,
    feature_set: str,
    seq_len: int,
    target_col: str = "target_excess_return",
    feature_cols: list[str] | None = None,
    history: pd.DataFrame | None = None,
    require_contiguous: bool = False,
):
    """Returns (X, y, cols, keys, stats).

    stats keys: n_kept, n_dropped_gap, n_candidates (eval-fold candidate windows).

    Each ticker's first seq_len rows are dropped (not enough trailing history).
    history: optional prior-fold rows prepended for warm-up (e.g. train before val).
    require_contiguous: keep only windows whose seq_len lookback months are
      calendar-consecutive (Period month gaps of exactly 1).
    """
    if feature_cols is not None:
        cols = [c for c in feature_cols if c in df.columns]
    else:
        cols = [c for c in get_feature_columns(feature_set) if c in df.columns]
    if target_col not in df.columns:
        raise KeyError(f"target_col={target_col!r} missing from panel")
    stats = {"n_kept": 0, "n_dropped_gap": 0, "n_candidates": 0}
    if not cols:
        return None, None, cols, [], stats

    hist = history if history is not None and len(history) else None
    parts_x, parts_y, parts_key = [], [], []

    for ticker, g_eval in df.sort_values("month").groupby("Ticker"):
        g_eval = g_eval.dropna(subset=[target_col])
        if g_eval.empty:
            continue
        eval_months = set(pd.to_datetime(g_eval["month"]).dt.strftime("%Y-%m-%d").values)

        if hist is not None:
            g_hist = hist.loc[hist["Ticker"].astype(str) == str(ticker)]
            if len(g_hist):
                g = (
                    pd.concat([g_hist, g_eval], axis=0, ignore_index=True)
                    .sort_values("month")
                    .drop_duplicates(subset=["month"], keep="last")
                )
            else:
                g = g_eval.sort_values("month")
        else:
            g = g_eval.sort_values("month")

        g = g.dropna(subset=[target_col])
        if len(g) <= seq_len:
            continue

        X = g[cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).values.astype(np.float32)
        y = g[target_col].values.astype(np.float32)
        months = pd.to_datetime(g["month"]).dt.strftime("%Y-%m-%d").values
        periods = pd.PeriodIndex(pd.to_datetime(g["month"]), freq="M")
        for i in range(seq_len, len(g)):
            # With history warm-up: only score months that belong to this fold
            if months[i] not in eval_months:
                continue
            stats["n_candidates"] += 1
            lookback = periods[i - seq_len : i]
            if require_contiguous and not _months_contiguous(lookback):
                stats["n_dropped_gap"] += 1
                continue
            parts_x.append(X[i - seq_len : i])
            parts_y.append(y[i])
            parts_key.append((str(ticker), months[i]))
            stats["n_kept"] += 1

    if not parts_x:
        return None, None, cols, [], stats
    return np.stack(parts_x), np.array(parts_y), cols, parts_key, stats


def fit_predict_lstm(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_te: np.ndarray,
    hidden: int,
    epochs: int,
    batch: int,
    X_va: np.ndarray | None = None,
    y_va: np.ndarray | None = None,
    patience: int = 5,
    lr: float = 1e-3,
    seed: int = 42,
    dropout: float = 0.0,
) -> np.ndarray:
    """Standardise sequence features (fit on train only) and train/predict an LSTM.

    If X_va/y_va are provided, early-stop on validation MSE and restore best weights.
    ``seed`` fixes torch/numpy RNGs so Exp grids are reproducible across runs.
    Uses CUDA + mixed precision (autocast/GradScaler) when a GPU is available.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    if use_amp:
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    n_features = X_tr.shape[2]
    flat = X_tr.reshape(-1, n_features)
    mean, std = flat.mean(axis=0), flat.std(axis=0) + 1e-9
    X_tr_s = (X_tr - mean) / std
    X_te_s = (X_te - mean) / std
    X_va_s = (X_va - mean) / std if X_va is not None else None

    model = LSTMRegressor(n_features, hidden, dropout=dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    X_t = torch.tensor(X_tr_s, device=device)
    y_t = torch.tensor(y_tr, device=device)
    X_v = torch.tensor(X_va_s, device=device) if X_va_s is not None else None
    y_v = torch.tensor(y_va, device=device) if y_va is not None else None
    g = torch.Generator()
    g.manual_seed(seed)
    # GradScaler API: torch.amp in newer torch; fall back to torch.cuda.amp
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    except (TypeError, AttributeError):
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_state = None
    best_va = float("inf")
    bad_epochs = 0

    for _ in range(epochs):
        model.train()
        perm = torch.randperm(len(y_t), generator=g)
        for i in range(0, len(y_t), batch):
            idx = perm[i : i + batch]
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                pred = model(X_t[idx])
                loss = loss_fn(pred, y_t[idx])
            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                opt.step()

        if X_v is not None and y_v is not None:
            model.eval()
            with torch.no_grad():
                with torch.autocast(device_type=device.type, enabled=use_amp):
                    va_loss = float(loss_fn(model(X_v), y_v).item())
            if va_loss < best_va - 1e-8:
                best_va = va_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= patience:
                    break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        X_te_t = torch.tensor(X_te_s, device=device)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            out = model(X_te_t)
        return out.detach().float().cpu().numpy()


def run_lstm(train, test, fold, store: ResultsStore, feature_set: str = "traditional"):
    cfg = load_config()
    seq_len = int(cfg["lstm"]["seq_len"])
    hidden = int(cfg["lstm"]["hidden_size"])
    epochs = int(cfg["lstm"]["epochs"])
    batch = int(cfg["lstm"]["batch_size"])

    X_tr, y_tr, cols, _, _ = _build_sequences(train, feature_set, seq_len)
    X_te, y_te, _, keys_te, _ = _build_sequences(test, feature_set, seq_len)
    if X_tr is None or X_te is None or len(y_te) == 0:
        store.add_h1("lstm", {"mse": None, "mae": None, "rmse": None, "r2": None, "n_obs": 0}, fold, feature_set)
        return None, None, cols

    pred = fit_predict_lstm(X_tr, y_tr, X_te, hidden, epochs, batch)
    metrics = predictive_metrics(y_te, pred)
    tickers, months = zip(*keys_te) if keys_te else ([], [])
    frame = pd.DataFrame({
        "Ticker": list(tickers),
        "month": list(months),
        "y_true": y_te,
        "y_pred": pred,
    })
    frame["error"] = frame["y_true"] - frame["y_pred"]
    frame["abs_error"] = frame["error"].abs()
    calc = build_calculation(
        run_id=store.run_id,
        model_id="lstm",
        bucket="h1",
        data_context=store._data_context,
        frame=frame,
        feature_set=feature_set,
        artifact_key=f"lstm__{feature_set}",
        model_meta={
            "family": "LSTM (PyTorch)",
            "formula": f"sequence length {seq_len} → LSTM(hidden={hidden}) → target_excess_return",
            "plain_english": (
                "A recurrent network reads the last seq_len months of features per ticker "
                "and forecasts next-month excess return. First seq_len months per ticker are excluded."
            ),
            "parameters": {
                "seq_len": seq_len,
                "hidden_size": hidden,
                "epochs": epochs,
                "batch_size": batch,
                "feature_columns": cols,
                "feature_set": feature_set,
            },
            "oos_unit": "ticker_month",
            "caveats": [
                f"Sample shrinks vs RF/XGBoost because first {seq_len} months per ticker lack history.",
            ],
        },
    )
    store.add_h1("lstm", metrics, fold, feature_set, calculation=calc)
    return None, (y_te, pred), cols


if __name__ == "__main__":
    print(
        "lstm_model.py is a pipeline module (imports data.*). Do not run it directly.\n"
        "  Full pipeline:  cd dissertation-models && python run_all.py\n"
        "  Manual script:    python \"methodology/model manul code/lstm_return.py\""
    )
    sys.exit(1)

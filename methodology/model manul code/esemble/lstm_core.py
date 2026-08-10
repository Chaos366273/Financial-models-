"""LSTM leg for stacked ensemble (standalone copy from dissertation-models/models/lstm_model.py)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from features_core import get_feature_columns


class LSTMRegressor(nn.Module):
    def __init__(self, n_features: int, hidden: int):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze(-1)


def build_sequences(
    df: pd.DataFrame,
    seq_len: int,
    feature_cols: list[str] | None = None,
    feature_set: str = "traditional",
) -> tuple[np.ndarray | None, np.ndarray | None, list[str], list[tuple[str, str]]]:
    if feature_cols is None:
        cols = [c for c in get_feature_columns(feature_set) if c in df.columns]
    else:
        cols = [c for c in feature_cols if c in df.columns]

    parts_x, parts_y, parts_key = [], [], []
    for ticker, g in df.sort_values("month").groupby("Ticker"):
        g = g.dropna(subset=["target_excess_return"])
        if len(g) <= seq_len or not cols:
            continue
        X = g[cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).values.astype(np.float32)
        y = g["target_excess_return"].values.astype(np.float32)
        months = pd.to_datetime(g["month"]).dt.strftime("%Y-%m-%d").values
        for i in range(seq_len, len(g)):
            parts_x.append(X[i - seq_len : i])
            parts_y.append(y[i])
            parts_key.append((str(ticker), months[i]))
    if not parts_x:
        return None, None, cols, []
    return np.stack(parts_x), np.array(parts_y), cols, parts_key


def fit_predict_lstm(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_te: np.ndarray,
    hidden: int,
    epochs: int,
    batch: int,
) -> np.ndarray:
    n_features = X_tr.shape[2]
    flat = X_tr.reshape(-1, n_features)
    mean, std = flat.mean(axis=0), flat.std(axis=0) + 1e-9
    X_tr_s = (X_tr - mean) / std
    X_te_s = (X_te - mean) / std

    model = LSTMRegressor(n_features, hidden)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    X_t = torch.tensor(X_tr_s)
    y_t = torch.tensor(y_tr)
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(len(y_t))
        for i in range(0, len(y_t), batch):
            idx = perm[i : i + batch]
            opt.zero_grad()
            pred = model(X_t[idx])
            loss = loss_fn(pred, y_t[idx])
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        return model(torch.tensor(X_te_s)).numpy()

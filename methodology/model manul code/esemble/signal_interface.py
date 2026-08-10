"""Architecture B signal interface: XGB/LSTM forecasts -> rank + size weights.

Trading PPO (timing) consumes ranked_signals.csv — it does NOT pick tickers.
Edit TOP_K / LONG_ONLY here; stacked_ensemble Stage 1 stays the forecast engine.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# --- Architecture B knobs ---
TOP_K = 20                 # max long candidates per month
LONG_ONLY = True           # only positive y_pred get weight
MIN_PRED = 0.0             # floor for long eligibility when LONG_ONLY
SOFTMAX_TEMP = 1.0         # >1 flattens weights; <1 sharpens


def build_ranked_signals(
    frame: pd.DataFrame,
    *,
    top_k: int = TOP_K,
    long_only: bool = LONG_ONLY,
    min_pred: float = MIN_PRED,
    temperature: float = SOFTMAX_TEMP,
) -> pd.DataFrame:
    """From ensemble predictions build per-month ranks and portfolio weights.

    Expected columns: Ticker, month (or date), y_pred
    Optional: xgb_pred, lstm_pred, y_true
    """
    df = frame.copy()
    if "month" not in df.columns and "date" in df.columns:
        df = df.rename(columns={"date": "month"})
    if "y_pred" not in df.columns:
        raise ValueError("frame must contain y_pred")

    df["month"] = pd.to_datetime(df["month"])
    df["Ticker"] = df["Ticker"].astype(str)
    df["y_pred"] = pd.to_numeric(df["y_pred"], errors="coerce").fillna(0.0)

    rows: list[dict[str, Any]] = []
    for month, g in df.groupby("month", sort=True):
        g = g.sort_values("y_pred", ascending=False).reset_index(drop=True)
        g["rank"] = np.arange(1, len(g) + 1)

        eligible = g.copy()
        if long_only:
            eligible = eligible.loc[eligible["y_pred"] > min_pred]
        eligible = eligible.head(int(top_k))

        weights = np.zeros(len(g), dtype=float)
        if not eligible.empty:
            scores = eligible["y_pred"].to_numpy(dtype=float)
            # shift so softmax is stable; negatives already filtered if long_only
            scores = np.maximum(scores, 0.0) if long_only else scores
            temp = max(float(temperature), 1e-6)
            ex = np.exp((scores - scores.max()) / temp)
            w = ex / ex.sum() if ex.sum() > 0 else np.ones(len(ex)) / len(ex)
            for i, idx in enumerate(eligible.index):
                weights[idx] = float(w[i])

        g["weight"] = weights
        g["long_candidate"] = g["weight"] > 0
        g["architecture"] = "B"
        rows.append(g)

    out = pd.concat(rows, ignore_index=True)
    cols = [
        "month", "Ticker", "y_pred", "rank", "weight", "long_candidate", "architecture",
    ]
    for opt in ("xgb_pred", "lstm_pred", "y_true"):
        if opt in out.columns:
            cols.append(opt)
    return out[cols]


def save_ranked_signals(ranked: pd.DataFrame, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ranked.to_csv(path, index=False)
    return path


def load_ranked_signals(path: Path | str) -> pd.DataFrame:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Ranked signals not found: {path}")
    df = pd.read_csv(path)
    df["month"] = pd.to_datetime(df["month"])
    return df


def month_target_weights(ranked: pd.DataFrame, month: str | pd.Timestamp) -> dict[str, float]:
    """Ticker -> weight for one month (Architecture B target book)."""
    m = pd.Timestamp(month)
    g = ranked.loc[ranked["month"] == m]
    if g.empty:
        # allow month string match on calendar day
        g = ranked.loc[ranked["month"].dt.to_period("M") == m.to_period("M")]
    return {
        str(r.Ticker): float(r.weight)
        for r in g.itertuples()
        if float(r.weight) > 0
    }


def signal_summary(ranked: pd.DataFrame) -> dict[str, Any]:
    months = ranked["month"].nunique()
    avg_longs = float(ranked.loc[ranked["long_candidate"], "Ticker"].groupby(ranked["month"]).nunique().mean()) if ranked["long_candidate"].any() else 0.0
    return {
        "architecture": "B",
        "months": int(months),
        "rows": int(len(ranked)),
        "top_k": TOP_K,
        "long_only": LONG_ONLY,
        "avg_long_candidates_per_month": avg_longs,
        "mean_weight_when_selected": float(ranked.loc[ranked["weight"] > 0, "weight"].mean()) if (ranked["weight"] > 0).any() else 0.0,
    }

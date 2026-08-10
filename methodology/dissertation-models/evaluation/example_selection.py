"""Deterministic representative OOS example selection."""
from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import pandas as pd


def _stable_rank_key(run_id: str, model_id: str, row: pd.Series) -> str:
    month = str(row.get("month", row.get("date", "")))
    ticker = str(row.get("Ticker", row.get("ticker", "")))
    return f"{run_id}|{model_id}|{month}|{ticker}"


def _hash_bucket(key: str) -> int:
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def select_examples(
    frame: pd.DataFrame,
    *,
    run_id: str,
    model_id: str,
    error_col: str = "abs_error",
    n: int = 10,
) -> pd.DataFrame:
    """Pick up to n representative rows using stratified absolute-error deciles.

    Within each occupied decile, choose the row with the smallest stable hash of
    (run_id, model_id, month/date, ticker). Final order is chronological.
    """
    if frame is None or frame.empty:
        return pd.DataFrame()

    work = frame.copy()
    if error_col not in work.columns:
        if {"y_true", "y_pred"}.issubset(work.columns):
            work[error_col] = (work["y_true"] - work["y_pred"]).abs()
        elif {"actual", "forecast"}.issubset(work.columns):
            work[error_col] = (work["actual"] - work["forecast"]).abs()
        else:
            work[error_col] = 0.0

    work = work.replace([np.inf, -np.inf], np.nan).dropna(subset=[error_col])
    if work.empty:
        return pd.DataFrame()

    if len(work) <= n:
        sort_cols = [c for c in ("month", "date", "Ticker") if c in work.columns]
        return work.sort_values(sort_cols).reset_index(drop=True)

    ranks = work[error_col].rank(method="average", pct=True)
    work["_decile"] = np.minimum((ranks * n).astype(int), n - 1)
    work["_hash"] = [
        _hash_bucket(_stable_rank_key(run_id, model_id, row))
        for _, row in work.iterrows()
    ]

    picked = (
        work.sort_values(["_decile", "_hash"])
        .groupby("_decile", as_index=False, sort=True)
        .first()
    )
    # If fewer than n deciles are occupied, top up with next-lowest hashes
    if len(picked) < n:
        remaining = work.loc[~work.index.isin(picked.index)].sort_values("_hash")
        need = n - len(picked)
        picked = pd.concat([picked, remaining.head(need)], ignore_index=True)

    sort_cols = [c for c in ("month", "date", "Ticker") if c in picked.columns]
    out = picked.drop(columns=[c for c in ("_decile", "_hash") if c in picked.columns])
    return out.sort_values(sort_cols).head(n).reset_index(drop=True)


def records_from_frame(frame: pd.DataFrame, columns: list[str] | None = None) -> list[dict[str, Any]]:
    if frame is None or frame.empty:
        return []
    cols = columns or list(frame.columns)
    cols = [c for c in cols if c in frame.columns]
    rows = []
    for _, row in frame[cols].iterrows():
        item: dict[str, Any] = {}
        for c in cols:
            val = row[c]
            if pd.isna(val):
                item[c] = None
            elif isinstance(val, (np.floating, float)):
                item[c] = float(val)
            elif isinstance(val, (np.integer, int)):
                item[c] = int(val)
            elif hasattr(val, "isoformat"):
                item[c] = val.isoformat()
            else:
                item[c] = str(val)
        rows.append(item)
    return rows

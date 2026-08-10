"""Portfolio performance metrics for H3a."""
from __future__ import annotations

import numpy as np
import pandas as pd


def max_drawdown(returns: pd.Series) -> float:
    wealth = (1 + returns.fillna(0)).cumprod()
    peak = wealth.cummax()
    dd = (wealth - peak) / peak
    return float(dd.min())


def portfolio_metrics(returns: pd.Series, rf: float = 0.0) -> dict:
    r = returns.dropna()
    if r.empty:
        return {"sharpe": 0.0, "sortino": 0.0, "max_drawdown": 0.0, "net_return_annual": 0.0}
    ann = r.mean() * 12
    vol = r.std() * np.sqrt(12)
    sharpe = (ann - rf) / vol if vol > 1e-9 else 0.0
    downside = r[r < 0]
    dvol = downside.std() * np.sqrt(12) if len(downside) else vol
    sortino = (ann - rf) / dvol if dvol > 1e-9 else 0.0
    return {
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "max_drawdown": max_drawdown(r),
        "net_return_annual": float(ann),
    }

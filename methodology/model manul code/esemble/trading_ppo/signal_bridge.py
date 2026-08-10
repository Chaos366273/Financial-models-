"""Bridge: load Architecture B ranked signals for the trading PPO.

Env wiring (narrow action space to TRADE/HOLD) is intentionally NOT here yet.
Use get_target_book(month) once FixedTradingEnv is adapted — approve that next.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pandas as pd

_ESEMBLE = Path(__file__).resolve().parent.parent
if str(_ESEMBLE) not in sys.path:
    sys.path.insert(0, str(_ESEMBLE))

from signal_interface import (  # noqa: E402
    load_ranked_signals,
    month_target_weights,
    signal_summary,
)

DEFAULT_RANKED_PATH = _ESEMBLE / "saved_data" / "ranked_signals.csv"


def load_ensemble_book(path: Path | str | None = None) -> pd.DataFrame:
    return load_ranked_signals(path or DEFAULT_RANKED_PATH)


def get_target_book(month: str | pd.Timestamp, path: Path | str | None = None) -> dict[str, float]:
    """Return {ticker: weight} for the given month from Stage 1 ranked signals."""
    ranked = load_ensemble_book(path)
    return month_target_weights(ranked, month)


def describe_signals(path: Path | str | None = None) -> dict[str, Any]:
    ranked = load_ensemble_book(path)
    return signal_summary(ranked)


if __name__ == "__main__":
    p = DEFAULT_RANKED_PATH
    if not p.is_file():
        print(f"No ranked signals yet. Run stacked_ensemble.py first.\nExpected: {p}")
    else:
        info = describe_signals(p)
        print("Architecture B signal book:")
        for k, v in info.items():
            print(f"  {k}: {v}")
        book = load_ensemble_book(p)
        latest = book["month"].max()
        print(f"\nLatest month {latest.date()} target weights:")
        for t, w in sorted(get_target_book(latest).items(), key=lambda x: -x[1]):
            print(f"  {t}: {w:.4f}")

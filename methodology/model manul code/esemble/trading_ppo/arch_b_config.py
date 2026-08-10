"""Architecture B switches for the local trading_ppo copy.

Stage 1 (stacked_ensemble) writes ranked_signals.csv.
Stage 2 uses those weights for ticker+size; PPO chooses BUY/SELL/HOLD timing.
"""
from __future__ import annotations

import os
from pathlib import Path

ARCH_B_ENABLED = True
DISABLE_SHORT = True
# When True, BUY for book names skips technical signal gating (models already ranked).
SKIP_SIGNAL_GATING = True

UNIVERSE_OVERRIDE_FROM_RANKED = True

# Prefer longest FTSE history among Arch B (weight>0) names so the shared
# monthly calendar expands. Empirically on ftse_from_final_data.csv:
#   n<=12 → ~285 months (file max) | n=15 → ~238 | n=18 → ~213 | n=20 → ~130 (cuts OOS early)
# Absolute calendar max on this monthly file is ~285 months (not 25y×12 with a fat book).
PREFER_LONG_HISTORY = os.environ.get("ESEMBLE_LONG_HISTORY", "1").strip() not in ("0", "false", "False")
MAX_UNIVERSE_SIZE = int(os.environ.get("ESEMBLE_MAX_UNIVERSE", "15"))

# Hard reset: delete saved zip before train so we do not resume a frozen BUY path.
# After ONE successful fresh run, set to False (or ESEMBLE_FORCE_FRESH=0) to allow resume.
FORCE_FRESH_TRAIN = os.environ.get("ESEMBLE_FORCE_FRESH", "1").strip() not in ("0", "false", "False")
# New seed breaks the 25→50→100k resume plateau (was 42).
TRAIN_SEED = int(os.environ.get("ESEMBLE_TRAIN_SEED", "43"))
# Always run a 1-episode stochastic sanity eval after deterministic (does not replace headline).
STOCHASTIC_SANITY_EVAL = os.environ.get("ESEMBLE_STOCH_SANITY", "1").strip() not in ("0", "false", "False")

_ESEMBLE = Path(__file__).resolve().parent.parent
_DEFAULT_RANKED = _ESEMBLE / "saved_data" / "ranked_signals.csv"
# stacked_ensemble sets ESEMBLE_RANKED_SIGNALS when auto-launching Stage 2
RANKED_SIGNALS_PATH = Path(os.environ.get("ESEMBLE_RANKED_SIGNALS", str(_DEFAULT_RANKED)))

_FTSE_CSV_CANDIDATES = (
    Path(__file__).resolve().parent / "ppo" / "saved_data" / "csv_data" / "ftse_from_final_data.csv",
    _ESEMBLE.parent / "final data for AI" / "01_market" / "ohlcv_monthly.csv",
)


def _load_ftse_month_counts() -> dict[str, int]:
    """Return {TICKER: n_unique_months} from local FTSE OHLCV if available."""
    import pandas as pd

    path = next((p for p in _FTSE_CSV_CANDIDATES if p.is_file()), None)
    if path is None:
        return {}
    try:
        df = pd.read_csv(path)
    except Exception:
        return {}
    date_col = next((c for c in df.columns if str(c).lower() in ("date", "month", "timestamp")), None)
    tic_col = next(
        (c for c in df.columns if str(c).lower() in ("company", "tic", "ticker", "symbol")),
        None,
    )
    if date_col is None or tic_col is None:
        return {}
    work = df[[date_col, tic_col]].copy()
    work["_d"] = pd.to_datetime(work[date_col], errors="coerce")
    work["_t"] = work[tic_col].astype(str).str.upper()
    work = work.dropna(subset=["_d"])
    if work.empty:
        return {}
    return work.groupby("_t")["_d"].nunique().astype(int).to_dict()


def load_ranked_universe(max_size: int | None = None) -> list[str]:
    """Tickers for Stage-2 PPO from Arch B ranked book, capped.

    Default: prefer longest FTSE history among weight>0 names (expands shared
    calendar), then by months-selected frequency. Set ESEMBLE_LONG_HISTORY=0
    to restore frequency-only ranking. Cap via ESEMBLE_MAX_UNIVERSE (default 15).
    """
    path = RANKED_SIGNALS_PATH
    if not path.is_file():
        return []
    import pandas as pd

    df = pd.read_csv(path)
    if "Ticker" not in df.columns:
        return []
    if "weight" in df.columns:
        df = df.loc[df["weight"].fillna(0) > 0]
    if df.empty:
        return []

    counts = df["Ticker"].astype(str).str.upper().value_counts()
    candidates = counts.index.tolist()
    hist = _load_ftse_month_counts() if PREFER_LONG_HISTORY else {}

    if hist:
        # Longest history first; break ties by Arch B selection frequency
        candidates = sorted(
            candidates,
            key=lambda t: (int(hist.get(t, 0)), int(counts.get(t, 0)), t),
            reverse=True,
        )
    # else: keep value_counts order (most frequently selected)

    cap = MAX_UNIVERSE_SIZE if max_size is None else int(max_size)
    if cap > 0:
        candidates = candidates[:cap]
    return candidates

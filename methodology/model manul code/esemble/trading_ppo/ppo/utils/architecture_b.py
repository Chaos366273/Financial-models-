"""Architecture B: ensemble ranked book drives ticker + size; PPO times TRADE/HOLD.

Used by FixedTradingEnv when enable_architecture_b() is called.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import pandas as pd

_ESEMBLE_ROOT = Path(__file__).resolve().parents[3]  # .../esemble
DEFAULT_RANKED = _ESEMBLE_ROOT / "saved_data" / "ranked_signals.csv"


class ArchitectureBController:
    """Maps env ticker indices <-> Stage-1 ranked weights for the active month."""

    def __init__(
        self,
        ranked_path: Path | str | None = None,
        ticker_symbols: list[str] | None = None,
        disable_short: bool = True,
    ):
        self.ranked_path = Path(ranked_path) if ranked_path else DEFAULT_RANKED
        self.disable_short = bool(disable_short)
        self.ticker_symbols = [str(t).upper() for t in (ticker_symbols or [])]
        self._symbol_to_idx = {s: i for i, s in enumerate(self.ticker_symbols)}
        self._ranked: Optional[pd.DataFrame] = None
        self._by_month: dict[pd.Period, dict[str, float]] = {}
        self._load()

    def _load(self) -> None:
        if not self.ranked_path.is_file():
            self._ranked = None
            return
        df = pd.read_csv(self.ranked_path)
        df["month"] = pd.to_datetime(df["month"])
        df["Ticker"] = df["Ticker"].astype(str).str.upper()
        df["weight"] = pd.to_numeric(df["weight"], errors="coerce").fillna(0.0)
        self._ranked = df
        self._by_month = {}
        for month, g in df.groupby(df["month"].dt.to_period("M")):
            book = {
                str(r.Ticker): float(r.weight)
                for r in g.itertuples()
                if float(r.weight) > 0
            }
            self._by_month[month] = book

    def set_tickers(self, ticker_symbols: list[str]) -> None:
        self.ticker_symbols = [str(t).upper() for t in ticker_symbols]
        self._symbol_to_idx = {s: i for i, s in enumerate(self.ticker_symbols)}

    def available(self) -> bool:
        return bool(self._by_month) and bool(self._symbol_to_idx)

    def overlap_count(self) -> int:
        if self._ranked is None:
            return 0
        sig = set(self._ranked.loc[self._ranked["weight"] > 0, "Ticker"].unique())
        return len(sig & set(self._symbol_to_idx.keys()))

    def month_from_date(self, date_like: Any) -> Optional[pd.Period]:
        if date_like is None:
            return None
        try:
            return pd.Timestamp(date_like).to_period("M")
        except Exception:
            return None

    def book_for_month(self, period: Optional[pd.Period]) -> dict[str, float]:
        if period is None or not self._by_month:
            return {}
        if period in self._by_month:
            return self._by_month[period]
        # nearest previous month with a book
        keys = sorted(self._by_month.keys())
        prev = [k for k in keys if k <= period]
        return self._by_month[prev[-1]] if prev else {}

    def target_weight(self, ticker_idx: int, period: Optional[pd.Period]) -> float:
        if ticker_idx < 0 or ticker_idx >= len(self.ticker_symbols):
            return 0.0
        sym = self.ticker_symbols[ticker_idx]
        return float(self.book_for_month(period).get(sym, 0.0))

    def candidate_indices(self, period: Optional[pd.Period]) -> list[int]:
        book = self.book_for_month(period)
        out = []
        for sym, w in book.items():
            if w > 0 and sym in self._symbol_to_idx:
                out.append(self._symbol_to_idx[sym])
        return out

    def summary(self) -> dict[str, Any]:
        return {
            "ranked_path": str(self.ranked_path),
            "months": len(self._by_month),
            "env_tickers": len(self.ticker_symbols),
            "overlap": self.overlap_count(),
            "disable_short": self.disable_short,
            "available": self.available() and self.overlap_count() > 0,
        }

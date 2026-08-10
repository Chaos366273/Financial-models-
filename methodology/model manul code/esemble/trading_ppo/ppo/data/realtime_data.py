"""
Realtime data stream compatibility layer.
Minimal wrapper to satisfy imports expecting `ppo.data.realtime_data`.
"""

from typing import Iterable, Optional, Dict, Any

__all__ = [
    "RealtimeData",
]


class RealtimeData:
    """
    Placeholder for real-time data streaming. In this project, live data is handled
    elsewhere; this class exists to preserve backward compatibility with TG4.1-style
    imports. Methods are no-ops returning safe defaults.
    """

    def __init__(self, symbols: Optional[Iterable[str]] = None) -> None:
        self.symbols = list(symbols) if symbols else []

    def connect(self) -> bool:
        return True

    def disconnect(self) -> None:
        return None

    def subscribe(self, symbols: Iterable[str]) -> None:
        self.symbols = list(set(self.symbols) | set(symbols))

    def get_latest_snapshot(self) -> Dict[str, Any]:
        return {}

    def poll(self) -> Optional[Dict[str, Any]]:
        return None



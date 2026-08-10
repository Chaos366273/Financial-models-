"""One-time (or --refresh) download of Yahoo panel data for manual models.

Run this first; lstm_return.py reuses saved_data/yahoo/ without hitting the API.

Example:
  python fetch_yahoo_data.py
  python fetch_yahoo_data.py --refresh
"""
from __future__ import annotations

import argparse

from yahoo_panel import (
    CACHE_DIR,
    DEFAULT_END,
    DEFAULT_START,
    DEFAULT_TICKERS,
    build_yahoo_panel,
    yahoo_feature_columns,
)


def main() -> None:
    p = argparse.ArgumentParser(description="Fetch and cache Yahoo Finance panel data.")
    p.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS, help="Ticker symbols")
    p.add_argument("--start", default=DEFAULT_START, help="History start (YYYY-MM-DD)")
    p.add_argument("--end", default=DEFAULT_END, help="History end (YYYY-MM-DD)")
    p.add_argument("--refresh", action="store_true", help="Ignore cache and re-download")
    args = p.parse_args()

    tickers = [t.upper() for t in args.tickers]
    panel = build_yahoo_panel(tickers, args.start, args.end, refresh=args.refresh)
    feat_cols = yahoo_feature_columns(include_alt=True, panel=panel)

    print("\nCache directory:", CACHE_DIR.resolve())
    print(f"Panel rows: {len(panel)} | tickers: {sorted(panel['Ticker'].unique())}")
    print(f"Month range: {panel['month'].min().date()} .. {panel['month'].max().date()}")
    print(f"Feature columns available ({len(feat_cols)}):")
    for c in feat_cols:
        print(f"  - {c}")
    print("\nIV4 (Bloomberg BSI): not on Yahoo — sentiment_proxy is a volume/return placeholder.")
    print("UK macro: not on Yahoo — add uk_macro.csv later if needed.")
    print("\nNext: python lstm_return.py  (DATA_SOURCE='yahoo' in that file)")


if __name__ == "__main__":
    main()

"""Export walk-forward XGB + LSTM forecasts for github private PPO feature injection.

Output CSV schema (merge on date + Ticker in PPO pipeline):
  date, Ticker, xgb_pred, lstm_pred, target_excess_return

Integration (github private):
  1. Run this script after building the monthly panel.
  2. Resample / forward-fill monthly preds to daily bars if needed.
  3. Add xgb_pred, lstm_pred to get_canonical_feature_cols() in
     ppo/utils/dynamic_config.py
  4. Merge in FeatureEngineer.create_features() or csv_loader_adapter
  5. Retrain PPO_fixed.py (obs size changes → new checkpoint)
  6. Update trained_feature_cols.pkl for TG5 paper trading

See github private/README.md and ppo/docs/SINGLE_SOURCES_OF_TRUTH.md.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.build_panel import build_panel
from data.config_loader import load_config
from models.stacked_ensemble import _lstm_frame, _pred_xgb_keyed


def export_forecasts(
    out_path: Path,
    feature_set: str = "traditional",
    include_train: bool = True,
    include_val: bool = True,
    include_test: bool = True,
) -> pd.DataFrame:
    cfg = load_config()
    _, splits = build_panel()
    train = splits["train"]
    val = splits["val"]
    test = splits["test"]

    seq_len = int(cfg["lstm"]["seq_len"])
    hidden = int(cfg["lstm"]["hidden_size"])
    epochs = int(cfg["lstm"]["epochs"])
    batch = int(cfg["lstm"]["batch_size"])

    parts: list[pd.DataFrame] = []
    split_map = [("train", train, include_train), ("val", val, include_val), ("test", test, include_test)]

    for label, frame, enabled in split_map:
        if not enabled or frame.empty:
            continue
        xgb_part = _pred_xgb_keyed(train, frame, feature_set)
        if xgb_part is None:
            continue
        lstm_part = _lstm_frame(train, frame, feature_set, seq_len, hidden, epochs, batch)
        merged = xgb_part.merge(lstm_part, on=["Ticker", "month"], how="left") if lstm_part is not None else xgb_part
        merged["split"] = label
        merged = merged.rename(columns={"month": "date"})
        parts.append(merged)

    if not parts:
        raise ValueError("No forecast rows exported — check panel splits and feature columns.")

    out = pd.concat(parts, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"])
    cols = ["date", "Ticker", "xgb_pred", "lstm_pred", "y", "split"]
    cols = [c for c in cols if c in out.columns]
    out = out[cols].rename(columns={"y": "target_excess_return"})
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Export XGB+LSTM forecasts for PPO (github private).")
    parser.add_argument(
        "--out",
        default=str(ROOT / "outputs" / "ppo_forecast_features.csv"),
        help="Output CSV path",
    )
    parser.add_argument(
        "--feature-set",
        default="traditional",
        choices=["traditional", "with_alt_data", "h1_clean", "all_indicators"],
    )
    args = parser.parse_args()

    out = export_forecasts(Path(args.out), feature_set=args.feature_set)
    print(f"Exported {len(out)} rows → {args.out}")
    print(f"  tickers: {out['Ticker'].nunique()} | splits: {sorted(out['split'].unique())}")
    print("Next: merge into github private feature_matrix (see module docstring).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

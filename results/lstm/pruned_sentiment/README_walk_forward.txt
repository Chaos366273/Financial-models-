LSTM rolling walk-forward
========================
Created: 2026-08-21T17:48:16

Settings:
  Rolling 10y train / 1y eval / 1y step
  seq=9 hidden=64 epochs=150 seed=42
  Honest OOS: NO early-stop on eval year
  Target: sector-neutral, winsor 1%/99%
  Liquidity drop: none
  Holdout 2024-2025: INCLUDED
  Warm-start: eval sequences prepend train history
  Default arms: kitchen-sink vs kitchen-sink + sentiment (H2)

Folders:
  walk_forward_baseline/   — kitchen_sink (~78 features)
  walk_forward_sentiment/  — kitchen_sink + 5 sentiment columns
  walk_forward_comparison.csv — year-by-year deltas
  shap/ + pruned_features.json — Deep SHAP persist (top-20 in >=7/12)
  shap/ under each arm — global_mean_abs_shap.csv (SHAP=ON)
  shap_baseline_vs_sentiment_comparison.csv — H2 SHAP ranks

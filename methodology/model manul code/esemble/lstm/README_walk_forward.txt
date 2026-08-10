LSTM rolling walk-forward
========================
Created: 2026-08-04T16:43:15

Settings:
  Rolling 10y train / 1y eval / 1y step
  seq=9 hidden=64 epochs=35 seed=42
  Honest OOS: NO early-stop on eval year
  Target: sector-neutral, winsor 1%/99%
  Liquidity drop: bottom 20% within month (R3a)
  Warm-start: eval sequences prepend train history
  Default arms: baseline + sentiment (H2); use --no-sentiment for baseline only

Folders:
  walk_forward_baseline/   — B_drop_bottom9 (19 features)
  walk_forward_sentiment/  — 19 + sentiment columns
  walk_forward_comparison.csv — year-by-year deltas

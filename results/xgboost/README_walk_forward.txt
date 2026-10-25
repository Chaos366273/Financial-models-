XGBoost rolling walk-forward (xgboost 2) — kitchen-sink H1
=======================================
Created: 2026-08-20T19:02:22
Log: C:\Users\Faiaz\Desktop\Courses\sem2\dessertation\Dessertation\analysis\xgboost\new run\walk_forward_run_20260820_185931.txt

Settings:
  Rolling 10y train / 1y eval / 1y step (same windows as FF5/Markov)
  HPs: depth=4, lr=0.03, mcw=8, trees=178
  Target: sector-neutral excess return, winsor 1%/99% (train bounds)
  Universe: is_member, drop Unknown sector, liquidity filter OFF
  SHAP: ON (mean |SHAP|, max 4000 rows/step)
  Prune: top-15 mean|SHAP| in >= 8 of 14 steps
  Gu R2_oos: zero benchmark (Gu, Kelly and Xiu 2020)

Folders:
  walk_forward_baseline/          — kitchen_sink (no sentiment)
  walk_forward_sentiment/         — kitchen_sink + 5 sentiment cols
  walk_forward_pruned_baseline/   — persistence-pruned kitchen_sink
  walk_forward_pruned_sentiment/  — persistence-pruned + sentiment
  walk_forward_comparison.csv
  shap_baseline_vs_sentiment_comparison.csv

figures/                          — RMSE/IC/LS/R2 charts
h1_kitchen_sink_scorecard.md

Final-test-style: ON
Golden lock: ON (--unlock-golden)

Ensemble mirror copy:
  C:\Users\Faiaz\Desktop\Courses\sem2\dessertation\methodology\model manul code\esemble\xgboost data

XGBoost rolling walk-forward (xgboost 2)
=======================================
Created: 2026-08-04T15:33:56
Log: C:\Users\Faiaz\Desktop\Courses\sem2\dessertation\methodology\model manul code\final test results\xgboost 2\walk_forward_run_20260804_153233.txt

Settings:
  Rolling 10y train / 1y eval / 1y step
  HPs: depth=4, lr=0.03, mcw=8, trees=178
  Target: sector-neutral, winsor 1%/99%
  Liquidity filter: OFF
  SHAP: ON (mean |SHAP|, max 4000 rows/step)

Also runs FINAL-TEST-STYLE (no golden lock):
  train months <= 2023-12, score 2024+  (same protocol as xgb_return --final-test)
  Outputs: final_test_style_baseline/ and final_test_style_sentiment/
  NEVER writes GOLDEN_RULE_test_already_scored.txt

Folders:
  walk_forward_baseline/   — B_drop_bottom9 (19 features)
  walk_forward_sentiment/  — 19 + sentiment columns
  final_test_style_* /     — holdout refit report (no lock)
  walk_forward_comparison.csv
  shap_baseline_vs_sentiment_comparison.csv
  shap_comparison_note.md

Ensemble mirror copy:
  C:\Users\Faiaz\Desktop\Courses\sem2\dessertation\methodology\model manul code\esemble\xgboost data

Original one-shot final test lock folder (untouched by this script):
  ../xgboost/

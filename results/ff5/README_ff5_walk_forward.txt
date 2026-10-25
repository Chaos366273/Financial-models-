Fama–French 5 — predictive walk-forward (Use B)
==============================================
Created: 2026-08-15T15:45:05
Log: C:\Users\Faiaz\Desktop\Courses\sem2\dessertation\methodology\model manul code\fama french\ff5_run_20260815_154446.txt

Primary protocol (H1):
  target = target_excess_return (next-month excess)
  features = MKT_RF, SMB, HML, RMW, CMA (month t)
  model = panel OLS (statsmodels)
  windows = same as xgb_walk_forward (10y train / 1y eval)
  universe = FTSE members (is_member) from master_panel.parquet
  hygiene = drop AIS.LN × 2014-04 (one stock-month; unadjusted corporate action)

Commands:
  python fama5.py                  # walk-forward + analysis tables
  python fama5.py --walk-forward   # same
  python fama5.py --build-h1-table # rebuild tables from saved CSVs

Do not use --classic (Yahoo single-ticker). H1 uses FTSE master_panel only.

H1 metrics: RMSE, MAE, R2, Gu R2_oos, Spearman IC, LS decile, LS Sharpe,
            Top-50 mean, Top-50 edge, hit rate.

Outputs:
  descriptive_stats.csv / .md      # mean, median, SD, min, max
  pearson_correlation.csv / .md    # Pearson of factors + target
  walk_forward_baseline/
    walk_forward_baseline_summary.csv
    pooled_oos_predictions.csv
    rolling_coefficients.csv      # OLS betas + HAC SEs per train window
    step_XX_YYYY/preview.csv | metrics.json | coefficients.json
  dissertation_table_h1.md
  h1_comparison_template.csv
  h1_rmse_by_year.csv
  h1_long_short_decile.md/.csv
  ff5_results_tracker.csv

All FF5 artefacts root: C:\Users\Faiaz\Desktop\Courses\sem2\dessertation\methodology\model manul code\fama french

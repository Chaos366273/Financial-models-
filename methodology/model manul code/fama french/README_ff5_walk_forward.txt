Fama–French 5 — predictive walk-forward (Use B)
==============================================
Created: 2026-08-07T21:51:26
Log: C:\Users\Faiaz\Desktop\Courses\sem2\dessertation\methodology\model manul code\fama french\ff5_run_20260807_215107.txt

Primary protocol (H1):
  target = target_excess_return (next-month excess)
  features = MKT_RF, SMB, HML, RMW, CMA (month t)
  model = panel OLS (statsmodels)
  windows = same as xgb_walk_forward (10y train / 1y eval)
  universe = FTSE members (is_member) from master_panel.parquet

Commands:
  python fama5.py                  # walk-forward + H1 tables
  python fama5.py --walk-forward   # same
  python fama5.py --build-h1-table # rebuild tables from saved CSVs
  python fama5.py --classic --ticker AAPL   # Use A appendix

Outputs:
  walk_forward_baseline/
    walk_forward_baseline_summary.csv
    pooled_oos_predictions.csv
    rolling_coefficients.csv      # const + factors (+ HAC SEs) across windows
    step_XX_YYYY/preview.csv | metrics.json | coefficients.json
  dissertation_table_h1.md         # paste into dissertation
  h1_comparison_template.csv
  h1_rmse_by_year.csv
  h1_long_short_decile.md/.csv     # EW + VW top/bottom 10%
  h1_2014_ais_sensitivity.json
  diagnostics/                    # 2014 plots + n_obs checks
  classic_appendix/                # Use A only
  ff5_results_tracker.csv

All FF5 artefacts root: C:\Users\Faiaz\Desktop\Courses\sem2\dessertation\methodology\model manul code\fama french

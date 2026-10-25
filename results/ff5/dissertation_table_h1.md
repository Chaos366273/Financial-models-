# Table 4.x - FF5 H1 metric scorecard (walk-forward 2012-2025)

Protocol: 10-year train -> 1-year eval; FTSE members (`is_member`); master panel.
Model: panel OLS of `target_excess_return` on MKT_RF, SMB, HML, RMW, CMA.
This is the traditional **baseline**. XGB/LSTM use the same windows, target, and metric list.

| Metric | Pooled OOS | Mean across years | Median across years |
|--------|-----------:|------------------:|--------------------:|
| RMSE | 0.1072 | 0.1049 | 0.1015 |
| MAE | 0.0689 | 0.0688 | 0.0653 |
| R² (standard) | -0.0164 | -0.0230 | -0.0160 |
| Gu R²_oos | -0.0120 | -0.0128 | -0.0111 |
| Spearman IC | - | - | - |
| Long-short decile | - | - | - |
| Long-short Sharpe | - | - | - |
| Top-50 mean return | - | - | - |
| Top-50 edge | - | - | - |
| Hit rate | - | - | - |
| n_obs | 76373 | - | - |

Ranking metrics (IC, long-short, Top-50, hit rate) are blank when every stock in a month gets the same predicted return (market-wide factors). That is expected for this FF5 spec, not a crash.
Descriptive stats and Pearson: `descriptive_stats.md`, `pearson_correlation.md`.
OLS betas (each train window): `walk_forward_baseline/rolling_coefficients.csv`.

# Table 4.x-b - Out-of-sample comparison stub (FF5 vs later models)

| Model | Features | RMSE (mean yr) | RMSE (median yr) | RMSE pooled | IC (Spearman) | Long-short EW | n_obs |
|-------|----------|----------------|------------------|-------------|---------------|---------------|-------|
| Fama-French 5 (panel OLS) | 5 factors only | 0.1049 | 0.1015 | 0.1072 | - | - | 76373 |
| XGBoost (B_drop_bottom9) | 19 features | - | - | - | 0.0399 | 0.0110 | 76373 |
| LSTM | same protocol (19 features) | 0.0999 | 0.0931 | - | 0.0105 | 0.0041 | 58622 |

Notes:
- XGBoost walk-forward summary stores IC / long-short (not RMSE).
- FF5 and XGB share the same hygiene drop: AIS.LN × April 2014 (one stock-month).
- **FF5 IC / cross-sectional long-short:** not identified under Use B with market-wide factors only (all stocks in a month receive the same `y_pred`). Compare FF5 on RMSE; use XGB/LSTM for ranking metrics.
- **Data hygiene:** one observation (AIS.LN, April 2014) showed a +21,800% excess return from an unadjusted corporate action and was excluded from the panel. AIS remaining on OOS pooled = 0. 2014 eval n = 5533. FF5 RMSE 2014 = 0.0874; pooled RMSE = 0.1072.
- Prefer **median year RMSE** as well as pooled RMSE (no AIS spike after the drop).

# Table 4.y - Walk-forward metrics by evaluation year

| Eval year | FF5 RMSE | LSTM RMSE | FF5 IC | XGB IC | LSTM IC | FF5 n | XGB n | LSTM n |
|-----------|----------|-----------|--------|--------|---------|-------|-------|--------|
| 2012 | 0.1073 | 0.0955 | - | 0.0539 | 0.0066 | 5400 | 5400 | 4132 |
| 2013 | 0.0984 | 0.0896 | - | 0.0389 | 0.0144 | 5387 | 5387 | 4179 |
| 2014 | 0.0874 | 0.0800 | - | 0.0485 | 0.0261 | 5533 | 5533 | 4211 |
| 2015 | 0.1028 | 0.0901 | - | 0.0602 | 0.0321 | 5627 | 5627 | 4259 |
| 2016 | 0.1090 | 0.1065 | - | 0.1262 | -0.0027 | 5593 | 5593 | 4262 |
| 2017 | 0.0791 | 0.0809 | - | 0.0311 | 0.0064 | 5562 | 5562 | 4303 |
| 2018 | 0.1024 | 0.0914 | - | 0.0373 | -0.0036 | 5580 | 5580 | 4279 |
| 2019 | 0.1005 | 0.0948 | - | 0.0194 | 0.0148 | 5593 | 5593 | 4272 |
| 2020 | 0.1733 | 0.1766 | - | 0.0201 | 0.0110 | 5611 | 5611 | 4234 |
| 2021 | 0.0967 | 0.0988 | - | 0.0488 | -0.0107 | 5593 | 5593 | 4301 |
| 2022 | 0.1163 | 0.1150 | - | 0.0380 | 0.0377 | 5510 | 5510 | 4243 |
| 2023 | 0.1039 | 0.0980 | - | 0.0307 | 0.0198 | 5366 | 5366 | 4185 |
| 2024 | 0.0958 | 0.0904 | - | 0.0266 | 0.0158 | 5269 | 5269 | 4117 |
| 2025 | 0.0960 | 0.0908 | - | -0.0205 | -0.0209 | 4749 | 4749 | 3645 |
| Mean | 0.1049 | 0.0999 | - | 0.0399 | 0.0105 | 76373 | 76373 | 58622 |
| Median | 0.1015 | 0.0931 | - | 0.0376 | 0.0127 | - | - | - |

See also decile long-short (EW/VW): `h1_long_short_decile.md` / `h1_long_short_decile.csv`.

## Sources
- FF5 pooled: `C:\Users\Faiaz\Desktop\Courses\sem2\dessertation\methodology\model manul code\fama french\walk_forward_baseline\pooled_oos_predictions.csv`
- FF5 summary: `C:\Users\Faiaz\Desktop\Courses\sem2\dessertation\methodology\model manul code\fama french\walk_forward_baseline\walk_forward_baseline_summary.csv`
- XGB summary: `C:\Users\Faiaz\Desktop\Courses\sem2\dessertation\methodology\model manul code\final test results\xgboost 2\walk_forward_baseline\walk_forward_baseline_summary.csv`
- LSTM summary: `C:\Users\Faiaz\Desktop\Courses\sem2\dessertation\methodology\model manul code\saved_data\lstm_walk_forward\walk_forward_baseline\summary_metrics.csv`
- Long-short: `C:\Users\Faiaz\Desktop\Courses\sem2\dessertation\methodology\model manul code\fama french\h1_long_short_decile.csv`
- Descriptive stats: `C:\Users\Faiaz\Desktop\Courses\sem2\dessertation\methodology\model manul code\fama french\descriptive_stats.csv`
- Pearson: `C:\Users\Faiaz\Desktop\Courses\sem2\dessertation\methodology\model manul code\fama french\pearson_correlation.csv`
- Rolling OLS betas: `C:\Users\Faiaz\Desktop\Courses\sem2\dessertation\methodology\model manul code\fama french\walk_forward_baseline\rolling_coefficients.csv`
- 2014 diagnostics: `C:\Users\Faiaz\Desktop\Courses\sem2\dessertation\methodology\model manul code\fama french\diagnostics`

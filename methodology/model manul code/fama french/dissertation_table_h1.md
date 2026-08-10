# Table 4.x - Out-of-sample predictive accuracy (walk-forward 2012-2025)

Protocol: 10-year train -> 1-year eval, annual steps; members-only master panel.
FF5: panel OLS of `target_excess_return` on MKT_RF, SMB, HML, RMW, CMA (Use B).
RMSE columns: **mean / median across evaluation years** (primary). `n_obs` = pooled stock-months (sum of per-year counts), not the AVERAGE-row mean.

| Model | Features | RMSE (mean yr) | RMSE (median yr) | RMSE pooled | IC (Spearman) | Long-short EW | n_obs |
|-------|----------|----------------|------------------|-------------|---------------|---------------|-------|
| Fama-French 5 (panel OLS) | 5 factors only | 0.3087 | 0.1033 | 0.7972 | - | - | 76374 |
| XGBoost (B_drop_bottom9) | 19 features | - | - | - | 0.0151 | -0.0277 | 76374 |
| LSTM | same protocol (19 features) | 0.0999 | 0.0931 | - | 0.0105 | 0.0041 | 58622 |

Notes:
- XGBoost walk-forward summary stores IC / long-short (not RMSE).
- FF5 and XGB share the same stock-month counts every year; LSTM is slightly smaller (sequence length drops early history).
- **FF5 IC / cross-sectional long-short:** not identified under Use B with market-wide factors only (all stocks in a month receive the same `y_pred`). Compare FF5 on RMSE; use XGB/LSTM for ranking metrics. See `h1_long_short_decile.md` (FF5 rows marked n/a when prediction has no cross-sectional dispersion).
- **2014 sensitivity:** `AIS.LN` on `2014-04-01` has `target_excess_return` ≈ 218.33. FF5 RMSE 2014 = 2.9360; excluding AIS = 0.0874. Pooled FF5 RMSE full = 0.7972; ex-AIS = 0.1075.
- Prefer **median year RMSE** for narrative robustness; pooled RMSE is dominated by AIS.LN.

# Table 4.y - Walk-forward metrics by evaluation year

| Eval year | FF5 RMSE | LSTM RMSE | FF5 IC | XGB IC | LSTM IC | FF5 n | XGB n | LSTM n |
|-----------|----------|-----------|--------|--------|---------|-------|-------|--------|
| 2012 | 0.1073 | 0.0955 | - | 0.0331 | 0.0066 | 5400 | 5400 | 4132 |
| 2013 | 0.0984 | 0.0896 | - | 0.0137 | 0.0144 | 5387 | 5387 | 4179 |
| 2014 | 2.9360 | 0.0800 | - | 0.0359 | 0.0261 | 5534 | 5534 | 4211 |
| 2015 | 0.1029 | 0.0901 | - | 0.0295 | 0.0321 | 5627 | 5627 | 4259 |
| 2016 | 0.1095 | 0.1065 | - | 0.0124 | -0.0027 | 5593 | 5593 | 4262 |
| 2017 | 0.0796 | 0.0809 | - | 0.0246 | 0.0064 | 5562 | 5562 | 4303 |
| 2018 | 0.1043 | 0.0914 | - | 0.0046 | -0.0036 | 5580 | 5580 | 4279 |
| 2019 | 0.1006 | 0.0948 | - | 0.0124 | 0.0148 | 5593 | 5593 | 4272 |
| 2020 | 0.1737 | 0.1766 | - | -0.0025 | 0.0110 | 5611 | 5611 | 4234 |
| 2021 | 0.0960 | 0.0988 | - | 0.0330 | -0.0107 | 5593 | 5593 | 4301 |
| 2022 | 0.1179 | 0.1150 | - | 0.0192 | 0.0377 | 5510 | 5510 | 4243 |
| 2023 | 0.1037 | 0.0980 | - | 0.0120 | 0.0198 | 5366 | 5366 | 4185 |
| 2024 | 0.0956 | 0.0904 | - | 0.0210 | 0.0158 | 5269 | 5269 | 4117 |
| 2025 | 0.0960 | 0.0908 | - | -0.0369 | -0.0209 | 4749 | 4749 | 3645 |
| Mean | 0.3087 | 0.0999 | - | 0.0151 | 0.0105 | 76374 | 76374 | 58622 |
| Median | 0.1033 | 0.0931 | - | 0.0164 | 0.0127 | - | - | - |

See also decile long-short (EW/VW): `h1_long_short_decile.md` / `h1_long_short_decile.csv`.

## Sources
- FF5 pooled: `C:\Users\Faiaz\Desktop\Courses\sem2\dessertation\methodology\model manul code\fama french\walk_forward_baseline\pooled_oos_predictions.csv`
- FF5 summary: `C:\Users\Faiaz\Desktop\Courses\sem2\dessertation\methodology\model manul code\fama french\walk_forward_baseline\walk_forward_baseline_summary.csv`
- XGB summary: `C:\Users\Faiaz\Desktop\Courses\sem2\dessertation\methodology\model manul code\final test results\xgboost 2\walk_forward_baseline\walk_forward_baseline_summary.csv`
- LSTM summary: `C:\Users\Faiaz\Desktop\Courses\sem2\dessertation\methodology\model manul code\saved_data\lstm_walk_forward\walk_forward_baseline\summary_metrics.csv`
- Long-short: `C:\Users\Faiaz\Desktop\Courses\sem2\dessertation\methodology\model manul code\fama french\h1_long_short_decile.csv`
- 2014 diagnostics: `C:\Users\Faiaz\Desktop\Courses\sem2\dessertation\methodology\model manul code\fama french\diagnostics`

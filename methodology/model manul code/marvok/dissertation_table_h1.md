# Table 4.x - Out-of-sample predictive accuracy (walk-forward 2012-2025)

Protocol: 10-year train -> 1-year eval, annual steps; members-only master panel.
Markov k=2: Hamilton bull/bear MS on MKT_RF + soft-weighted regime FF5 (primary classical).
Markov k=4: Maheu-style labels by sorted means (bear / bear_rally / bull_correction / bull) + regime FF5 (extension).
FF5: panel OLS of `target_excess_return` on MKT_RF, SMB, HML, RMW, CMA.
RMSE columns: **mean / median across evaluation years** (primary).

| Model | Features | RMSE (mean yr) | RMSE (median yr) | RMSE pooled | IC (Spearman) | Long-short EW | n_obs |
|-------|----------|----------------|------------------|-------------|---------------|---------------|-------|
| Markov regime-FF5 (k=2 Hamilton) | 2-regime MS on MKT_RF + FF5 WLS | 0.3090 | 0.1033 | 0.7973 | - | - | 76374 |
| Markov regime-FF5 (k=4 Maheu-style) | 4-regime MS (bear/rally/correction/bull) + FF5 WLS | 0.3112 | 0.1070 | 0.7976 | - | - | 76374 |
| Fama-French 5 (panel OLS) | 5 factors only | 0.3087 | 0.1033 | 0.7972 | - | - | 76374 |
| XGBoost (B_drop_bottom9) | 19 features | - | - | - | 0.0157 | -0.0262 | 76374 |
| LSTM | same protocol (19 features) | 0.3197 | 0.1077 | - | 0.0069 | 0.0034 | 73951 |

Notes:
- k=2 is the frozen H1 Hamilton baseline; k=4 is the ERP Maheu-motivated extension (unconstrained MS with mean-sorted labels — not the full hierarchical Maheu sampler).
- Markov/FF5 use market-wide factors => within-month `y_pred` has little cross-section; compare these on RMSE. Use XGB/LSTM for ranking metrics.
- Market-level Hamilton forecasts: `market_appendix/` (k=2) and `market_appendix_k4/` (k=4).
- Prefer **median year RMSE**; pooled/mean can be dominated by 2014 extremes.

# Table 4.y - Walk-forward metrics by evaluation year

| Eval year | Markov k2 RMSE | Markov k4 RMSE | FF5 RMSE | LSTM RMSE | XGB IC | LSTM IC |
|-----------|----------------|----------------|----------|-----------|--------|---------|
| 2012 | 0.1072 | 0.1137 | 0.1073 | 0.1181 | 0.0234 | 0.0148 |
| 2013 | 0.0987 | 0.0973 | 0.0984 | 0.1039 | 0.0176 | 0.0106 |
| 2014 | 2.9362 | 2.9360 | 2.9360 | 2.9948 | 0.0494 | 0.0083 |
| 2015 | 0.1027 | 0.1084 | 0.1029 | 0.1081 | 0.0235 | 0.0309 |
| 2016 | 0.1083 | 0.1098 | 0.1095 | 0.1156 | 0.0063 | 0.0071 |
| 2017 | 0.0796 | 0.0815 | 0.0796 | 0.0872 | 0.0303 | 0.0228 |
| 2018 | 0.1069 | 0.1086 | 0.1043 | 0.1073 | 0.0059 | -0.0026 |
| 2019 | 0.1004 | 0.1056 | 0.1006 | 0.1092 | 0.0192 | -0.0075 |
| 2020 | 0.1740 | 0.1734 | 0.1737 | 0.1849 | -0.0048 | 0.0372 |
| 2021 | 0.0955 | 0.1021 | 0.0960 | 0.1071 | 0.0184 | -0.0296 |
| 2022 | 0.1191 | 0.1194 | 0.1179 | 0.1283 | 0.0146 | -0.0025 |
| 2023 | 0.1039 | 0.1021 | 0.1037 | 0.1070 | 0.0221 | -0.0209 |
| 2024 | 0.0973 | 0.1023 | 0.0956 | 0.1009 | 0.0221 | 0.0091 |
| 2025 | 0.0962 | 0.0961 | 0.0960 | 0.1027 | -0.0280 | 0.0184 |
| Mean | 0.3090 | 0.3112 | 0.3087 | 0.3197 | 0.0157 | 0.0069 |
| Median | 0.1033 | 0.1070 | 0.1033 | 0.1077 | 0.0188 | 0.0087 |

# Appendix - Average regime parameters (k=2)

| mu_bear | mu_bull | persist_bear | persist_bull | dur_bear (m) | dur_bull (m) | mkt RMSE mean |
|---------|---------|--------------|--------------|--------------|--------------|---------------|
| -0.0024 | 0.0206 | 0.8715 | 0.7806 | 36.2051 | 47.5616 | 0.0433 |
Caveat: market RMSE is month-level MKT_RF forecast error — not comparable 1:1 to stock-month H1 RMSE.

# Appendix - Average regime parameters (k=4)

| mu_bear | mu_bear_rally | mu_bull_correction | mu_bull | mkt RMSE mean |
|---------|---------|---------|---------|---------|
| -0.0808 | -0.0235 | 0.0120 | 0.0781 | 0.0436 |
Caveat: market RMSE is month-level MKT_RF forecast error — not comparable 1:1 to stock-month H1 RMSE.

## Sources
- Markov k=2: `C:\Users\Faiaz\Desktop\Courses\sem2\dessertation\methodology\model manul code\marvok\walk_forward_baseline`
- Markov k=4: `C:\Users\Faiaz\Desktop\Courses\sem2\dessertation\methodology\model manul code\marvok\walk_forward_k4`
- Market appendix k=2: `C:\Users\Faiaz\Desktop\Courses\sem2\dessertation\methodology\model manul code\marvok\market_appendix`
- Market appendix k=4: `C:\Users\Faiaz\Desktop\Courses\sem2\dessertation\methodology\model manul code\marvok\market_appendix_k4`
- FF5 summary: `C:\Users\Faiaz\Desktop\Courses\sem2\dessertation\methodology\model manul code\final test results\ff5\walk_forward_baseline\walk_forward_baseline_summary.csv`
- XGB summary: `C:\Users\Faiaz\Desktop\Courses\sem2\dessertation\methodology\model manul code\final test results\xgboost 2\walk_forward_baseline\walk_forward_baseline_summary.csv`
- LSTM summary: `C:\Users\Faiaz\Desktop\Courses\sem2\dessertation\methodology\model manul code\saved_data\lstm_walk_forward\walk_forward_baseline\summary_metrics.csv`

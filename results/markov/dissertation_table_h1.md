# Table 4.x - Out-of-sample predictive accuracy (walk-forward 2012-2025)

Protocol: 10-year train -> 1-year eval, annual steps; members-only master panel.
Markov k=2: Hamilton bull/bear MS on MKT_RF + soft-weighted regime FF5 (primary classical).
Markov k=4: Maheu-style labels by sorted means (bear / bear_rally / bull_correction / bull) + regime FF5 (extension).
FF5: panel OLS of `target_excess_return` on MKT_RF, SMB, HML, RMW, CMA.
RMSE columns: **mean / median across evaluation years** (primary).

| Model | Features | RMSE (mean yr) | RMSE (median yr) | RMSE pooled | IC (Spearman) | Long-short EW | n_obs |
|-------|----------|----------------|------------------|-------------|---------------|---------------|-------|
| Markov regime-FF5 (k=2 Hamilton) | 2-regime MS on MKT_RF + FF5 WLS | 0.1049 | 0.1019 | 0.1071 | - | - | 76373 |
| Markov regime-FF5 (k=4 Maheu-style) | 4-regime MS (bear/rally/correction/bull) + FF5 WLS | 0.1073 | 0.1047 | 0.1094 | - | - | 76373 |
| Fama-French 5 (panel OLS) | 5 factors only | 0.1049 | 0.1015 | 0.1072 | - | - | 76373 |
| XGBoost (B_drop_bottom9) | 19 features | - | - | - | 0.0399 | 0.0110 | 76373 |
| LSTM | same protocol (19 features) | 0.0999 | 0.0931 | - | 0.0105 | 0.0041 | 58622 |

Notes:
- k=2 is the frozen H1 Hamilton baseline; k=4 is the ERP Maheu-motivated extension (unconstrained MS with mean-sorted labels — not the full hierarchical Maheu sampler).
- Markov/FF5 use market-wide factors => within-month `y_pred` has little cross-section; ranking (IC, long-short, Top-50, hit rate) is n/a when `ranking_identified=False`. Compare Markov vs FF5 on RMSE/MAE/R2/R2_oos. Use XGB/LSTM for ranking metrics.
- Ten H1 metrics match fama5.py: RMSE, MAE, R2, Gu R2_oos, Spearman IC, long-short mean, long-short Sharpe, Top-50 mean, Top-50 edge, hit rate.
- Regime WLS and pooled FF5 fits use HAC(4) standard errors (point estimates unchanged).
- Market-level Hamilton forecasts: `market_appendix/` (k=2) and `market_appendix_k4/` (k=4).
- Prefer **median year RMSE**; pooled/mean can be dominated by 2014 extremes.

# Table 4.y - Walk-forward metrics by evaluation year

| Eval year | Markov k2 RMSE | Markov k4 RMSE | FF5 RMSE | LSTM RMSE | XGB IC | LSTM IC |
|-----------|----------------|----------------|----------|-----------|--------|---------|
| 2012 | 0.1072 | 0.1137 | 0.1073 | 0.0955 | 0.0539 | 0.0066 |
| 2013 | 0.0987 | 0.0973 | 0.0984 | 0.0896 | 0.0389 | 0.0144 |
| 2014 | 0.0858 | 0.0882 | 0.0874 | 0.0800 | 0.0485 | 0.0261 |
| 2015 | 0.1036 | 0.1078 | 0.1028 | 0.0901 | 0.0602 | 0.0321 |
| 2016 | 0.1063 | 0.1083 | 0.1090 | 0.1065 | 0.1262 | -0.0027 |
| 2017 | 0.0795 | 0.0803 | 0.0791 | 0.0809 | 0.0311 | 0.0064 |
| 2018 | 0.1034 | 0.1054 | 0.1024 | 0.0914 | 0.0373 | -0.0036 |
| 2019 | 0.1005 | 0.1047 | 0.1005 | 0.0948 | 0.0194 | 0.0148 |
| 2020 | 0.1731 | 0.1723 | 0.1733 | 0.1766 | 0.0201 | 0.0110 |
| 2021 | 0.0964 | 0.1047 | 0.0967 | 0.0988 | 0.0488 | -0.0107 |
| 2022 | 0.1166 | 0.1172 | 0.1163 | 0.1150 | 0.0380 | 0.0377 |
| 2023 | 0.1041 | 0.1025 | 0.1039 | 0.0980 | 0.0307 | 0.0198 |
| 2024 | 0.0967 | 0.1040 | 0.0958 | 0.0904 | 0.0266 | 0.0158 |
| 2025 | 0.0962 | 0.0961 | 0.0960 | 0.0908 | -0.0205 | -0.0209 |
| Mean | 0.1049 | 0.1073 | 0.1049 | 0.0999 | 0.0399 | 0.0105 |
| Median | 0.1019 | 0.1047 | 0.1015 | 0.0931 | 0.0376 | 0.0127 |

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
- FF5 summary: `C:\Users\Faiaz\Desktop\Courses\sem2\dessertation\methodology\model manul code\fama french\walk_forward_baseline\walk_forward_baseline_summary.csv`
- XGB summary: `C:\Users\Faiaz\Desktop\Courses\sem2\dessertation\methodology\model manul code\final test results\xgboost 2\walk_forward_baseline\walk_forward_baseline_summary.csv`
- LSTM summary: `C:\Users\Faiaz\Desktop\Courses\sem2\dessertation\methodology\model manul code\saved_data\lstm_walk_forward\walk_forward_baseline\summary_metrics.csv`

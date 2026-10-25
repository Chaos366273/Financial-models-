# R4 Sensitivity — Market series for regime identification (rolling + soft + filtered)

mkt_rf (baseline): Markov fit on panel `MKT_RF` (Ken French Europe / panel factor).
ew_excess (R4): Markov fit on equal-weight member contemporaneous excess (`monthly_return - RF_FF`).
FF5 stock forecasts still use MKT_RF/SMB/HML/RMW/CMA as regressors; only the regime-state series changes. Lower RMSE is better.

| Spec | RMSE mean yr | RMSE median yr | RMSE pooled | n_obs | n_steps |
|------|--------------|----------------|-------------|-------|---------|
| k2_mkt_rf | 0.1049 | 0.1019 | 0.1071 | 76373 | 14 |
| k2_ew_excess | 0.3087 | 0.1032 | 0.7972 | 76374 | 14 |
| k4_mkt_rf | 0.1073 | 0.1047 | 0.1094 | 76373 | 14 |
| k4_ew_excess | 0.3085 | 0.1029 | 0.7973 | 76374 | 14 |

## By evaluation year

| Eval year | k2 MKT_RF | k2 EW excess | k4 MKT_RF | k4 EW excess |
|-----------|-----------|--------------|-----------|--------------|
| 2012 | 0.1072 | 0.1069 | 0.1137 | 0.1064 |
| 2013 | 0.0987 | 0.0982 | 0.0973 | 0.0984 |
| 2014 | 0.0858 | 2.9360 | 0.0882 | 2.9361 |
| 2015 | 0.1036 | 0.1025 | 0.1078 | 0.1024 |
| 2016 | 0.1063 | 0.1082 | 0.1083 | 0.1083 |
| 2017 | 0.0795 | 0.0795 | 0.0803 | 0.0798 |
| 2018 | 0.1034 | 0.1053 | 0.1054 | 0.1044 |
| 2019 | 0.1005 | 0.1005 | 0.1047 | 0.1003 |
| 2020 | 0.1731 | 0.1736 | 0.1723 | 0.1726 |
| 2021 | 0.0964 | 0.0972 | 0.1047 | 0.0960 |
| 2022 | 0.1166 | 0.1183 | 0.1172 | 0.1195 |
| 2023 | 0.1041 | 0.1040 | 0.1025 | 0.1035 |
| 2024 | 0.0967 | 0.0956 | 0.1040 | 0.0962 |
| 2025 | 0.0962 | 0.0955 | 0.0961 | 0.0956 |
| Mean | 0.1049 | 0.3087 | 0.1073 | 0.3085 |
| Median | 0.1019 | 0.1032 | 0.1047 | 0.1029 |

CSV: `market_series_sensitivity_summary.csv`, `market_series_sensitivity_by_year.csv`

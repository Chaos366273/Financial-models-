# R5 Sensitivity — Pooled FF5 fallback on/off (rolling + soft + filtered + MKT_RF)

fallback (baseline): thin/failed regimes replaced by pooled FF5; step errors can also fall back to pooled FF5.
nofallback (R5): if any regime is thin/failed, the step errors out (no silent FF5).
Lower RMSE is better. Prefer median year RMSE among completed steps.

| Spec | RMSE mean yr | RMSE median yr | RMSE pooled | n_obs | n_steps_ok | n_fallback_mode |
|------|--------------|----------------|-------------|-------|------------|-----------------|
| k2_fallback | 0.1049 | 0.1019 | 0.1071 | 76373 | 14 | 0 |
| k2_nofallback | 0.3090 | 0.1033 | 0.7973 | 76374 | 14 | 0 |
| k4_fallback | 0.1073 | 0.1047 | 0.1094 | 76373 | 14 | 0 |
| k4_nofallback | 0.3112 | 0.1070 | 0.7976 | 76374 | 14 | 0 |

## By evaluation year

| Eval year | k2 fallback | k2 nofallback | k4 fallback | k4 nofallback |
|-----------|-------------|---------------|-------------|---------------|
| 2012 | 0.1072 | 0.1072 | 0.1137 | 0.1137 |
| 2013 | 0.0987 | 0.0987 | 0.0973 | 0.0973 |
| 2014 | 0.0858 | 2.9362 | 0.0882 | 2.9360 |
| 2015 | 0.1036 | 0.1027 | 0.1078 | 0.1084 |
| 2016 | 0.1063 | 0.1083 | 0.1083 | 0.1098 |
| 2017 | 0.0795 | 0.0796 | 0.0803 | 0.0815 |
| 2018 | 0.1034 | 0.1069 | 0.1054 | 0.1086 |
| 2019 | 0.1005 | 0.1004 | 0.1047 | 0.1056 |
| 2020 | 0.1731 | 0.1740 | 0.1723 | 0.1734 |
| 2021 | 0.0964 | 0.0955 | 0.1047 | 0.1021 |
| 2022 | 0.1166 | 0.1191 | 0.1172 | 0.1194 |
| 2023 | 0.1041 | 0.1039 | 0.1025 | 0.1021 |
| 2024 | 0.0967 | 0.0973 | 0.1040 | 0.1023 |
| 2025 | 0.0962 | 0.0962 | 0.0961 | 0.0961 |
| Mean | 0.1049 | 0.3090 | 0.1073 | 0.3112 |
| Median | 0.1019 | 0.1033 | 0.1047 | 0.1070 |

CSV: `fallback_sensitivity_summary.csv`, `fallback_sensitivity_by_year.csv`

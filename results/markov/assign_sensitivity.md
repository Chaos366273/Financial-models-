# R1 Sensitivity — Soft vs Hard regime assignment (rolling only)

Soft: train months weighted by filtered regime probabilities (baseline).
Hard: each train month assigned fully to argmax regime, then FF5 OLS per regime.
OOS forecast blend still uses rolled soft probabilities. Lower RMSE is better.

| Spec | RMSE mean yr | RMSE median yr | RMSE pooled | n_obs | n_steps |
|------|--------------|----------------|-------------|-------|---------|
| k2_soft | 0.1049 | 0.1019 | 0.1071 | 76373 | 14 |
| k2_hard | 0.3090 | 0.1041 | 0.7974 | 76374 | 14 |
| k4_soft | 0.1073 | 0.1047 | 0.1094 | 76373 | 14 |
| k4_hard | 0.3284 | 0.1195 | 0.8007 | 76374 | 14 |

## By evaluation year

| Eval year | k2 soft | k2 hard | k4 soft | k4 hard |
|-----------|---------|---------|---------|---------|
| 2012 | 0.1072 | 0.1071 | 0.1137 | 0.1321 |
| 2013 | 0.0987 | 0.0997 | 0.0973 | 0.0985 |
| 2014 | 0.0858 | 2.9362 | 0.0882 | 2.9360 |
| 2015 | 0.1036 | 0.1028 | 0.1078 | 0.1147 |
| 2016 | 0.1063 | 0.1076 | 0.1083 | 0.1282 |
| 2017 | 0.0795 | 0.0797 | 0.0803 | 0.0850 |
| 2018 | 0.1034 | 0.1078 | 0.1054 | 0.1134 |
| 2019 | 0.1005 | 0.1004 | 0.1047 | 0.1175 |
| 2020 | 0.1731 | 0.1740 | 0.1723 | 0.1789 |
| 2021 | 0.0964 | 0.0944 | 0.1047 | 0.0997 |
| 2022 | 0.1166 | 0.1178 | 0.1172 | 0.1350 |
| 2023 | 0.1041 | 0.1055 | 0.1025 | 0.2310 |
| 2024 | 0.0967 | 0.0957 | 0.1040 | 0.1216 |
| 2025 | 0.0962 | 0.0966 | 0.0961 | 0.1066 |
| Mean | 0.1049 | 0.3090 | 0.1073 | 0.3284 |
| Median | 0.1019 | 0.1041 | 0.1047 | 0.1195 |

CSV: `assign_sensitivity_summary.csv`, `assign_sensitivity_by_year.csv`

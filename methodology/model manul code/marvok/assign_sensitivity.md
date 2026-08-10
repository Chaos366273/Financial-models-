# R1 Sensitivity — Soft vs Hard regime assignment (rolling only)

Soft: train months weighted by filtered regime probabilities (baseline).
Hard: each train month assigned fully to argmax regime, then FF5 OLS per regime.
OOS forecast blend still uses rolled soft probabilities. Lower RMSE is better.

| Spec | RMSE mean yr | RMSE median yr | RMSE pooled | n_obs | n_steps |
|------|--------------|----------------|-------------|-------|---------|
| k2_soft | 0.3090 | 0.1033 | 0.7973 | 76374 | 14 |
| k2_hard | 0.3090 | 0.1041 | 0.7974 | 76374 | 14 |
| k4_soft | 0.3112 | 0.1070 | 0.7976 | 76374 | 14 |
| k4_hard | 0.3284 | 0.1195 | 0.8007 | 76374 | 14 |

## By evaluation year

| Eval year | k2 soft | k2 hard | k4 soft | k4 hard |
|-----------|---------|---------|---------|---------|
| 2012 | 0.1072 | 0.1071 | 0.1137 | 0.1321 |
| 2013 | 0.0987 | 0.0997 | 0.0973 | 0.0985 |
| 2014 | 2.9362 | 2.9362 | 2.9360 | 2.9360 |
| 2015 | 0.1027 | 0.1028 | 0.1084 | 0.1147 |
| 2016 | 0.1083 | 0.1076 | 0.1098 | 0.1282 |
| 2017 | 0.0796 | 0.0797 | 0.0815 | 0.0850 |
| 2018 | 0.1069 | 0.1078 | 0.1086 | 0.1134 |
| 2019 | 0.1004 | 0.1004 | 0.1056 | 0.1175 |
| 2020 | 0.1740 | 0.1740 | 0.1734 | 0.1789 |
| 2021 | 0.0955 | 0.0944 | 0.1021 | 0.0997 |
| 2022 | 0.1191 | 0.1178 | 0.1194 | 0.1350 |
| 2023 | 0.1039 | 0.1055 | 0.1021 | 0.2310 |
| 2024 | 0.0973 | 0.0957 | 0.1023 | 0.1216 |
| 2025 | 0.0962 | 0.0966 | 0.0961 | 0.1066 |
| Mean | 0.3090 | 0.3090 | 0.3112 | 0.3284 |
| Median | 0.1033 | 0.1041 | 0.1070 | 0.1195 |

CSV: `assign_sensitivity_summary.csv`, `assign_sensitivity_by_year.csv`

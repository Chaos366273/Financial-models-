# R2 Sensitivity — Filtered vs Smoothed train probabilities (rolling + soft)

Filtered (baseline): train FF5 weights use filtered regime probs.
Smoothed (R2): train FF5 weights use smoothed regime probs.
OOS always: end-of-train filtered probs rolled with P (no look-ahead).
Lower RMSE is better.

| Spec | RMSE mean yr | RMSE median yr | RMSE pooled | n_obs | n_steps |
|------|--------------|----------------|-------------|-------|---------|
| k2_filtered | 0.1049 | 0.1019 | 0.1071 | 76373 | 14 |
| k2_smoothed | 0.3089 | 0.1034 | 0.7973 | 76374 | 14 |
| k4_filtered | 0.1073 | 0.1047 | 0.1094 | 76373 | 14 |
| k4_smoothed | 0.3115 | 0.1078 | 0.7976 | 76374 | 14 |

## By evaluation year

| Eval year | k2 filtered | k2 smoothed | k4 filtered | k4 smoothed |
|-----------|-------------|-------------|-------------|-------------|
| 2012 | 0.1072 | 0.1077 | 0.1137 | 0.1132 |
| 2013 | 0.0987 | 0.0974 | 0.0973 | 0.0969 |
| 2014 | 0.0858 | 2.9362 | 0.0882 | 2.9360 |
| 2015 | 0.1036 | 0.1029 | 0.1078 | 0.1080 |
| 2016 | 0.1063 | 0.1072 | 0.1083 | 0.1096 |
| 2017 | 0.0795 | 0.0795 | 0.0803 | 0.0809 |
| 2018 | 0.1034 | 0.1070 | 0.1054 | 0.1077 |
| 2019 | 0.1005 | 0.1004 | 0.1047 | 0.1079 |
| 2020 | 0.1731 | 0.1736 | 0.1723 | 0.1731 |
| 2021 | 0.0964 | 0.0955 | 0.1047 | 0.1037 |
| 2022 | 0.1166 | 0.1191 | 0.1172 | 0.1208 |
| 2023 | 0.1041 | 0.1039 | 0.1025 | 0.1029 |
| 2024 | 0.0967 | 0.0984 | 0.1040 | 0.1020 |
| 2025 | 0.0962 | 0.0963 | 0.0961 | 0.0980 |
| Mean | 0.1049 | 0.3089 | 0.1073 | 0.3115 |
| Median | 0.1019 | 0.1034 | 0.1047 | 0.1078 |

CSV: `prob_sensitivity_summary.csv`, `prob_sensitivity_by_year.csv`

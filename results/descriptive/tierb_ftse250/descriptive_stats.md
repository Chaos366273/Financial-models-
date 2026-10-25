# Table B.1b — FTSE 250 descriptive statistics

Series: Bloomberg `ftse250_level` in `macro_wide.csv`. Returns: r_t = ln(P_t/P_{t-1}). DV2: sqrt(252) x 30-day sample SD. Complete cases (return + RV): n = 6,539 days; 2000-02-15 to 2025-12-31. Same recipe as Table B.1 (FTSE 100); do not copy-paste 100 numbers.

| Variable | n | Mean | Median | SD | Min | Max |
|----------|--:|-----:|-------:|---:|----:|----:|
| Daily log return (`ret`) | 6539 | 0.0002 | 0.0007 | 0.0106 | -0.0982 | 0.0804 |
| Absolute daily return (`abs_ret`) | 6539 | 0.0074 | 0.0053 | 0.0076 | 0.0000 | 0.0982 |
| Squared daily return (`ret_sq`) | 6539 | 0.0001 | 0.0000 | 0.0003 | 0.0000 | 0.0096 |
| 30-day realised vol (annualised, DV2) (`rv_30_ann`) | 6539 | 0.1485 | 0.1259 | 0.0784 | 0.0514 | 0.6959 |
| FTSE 250 index level (`ftse250_level`) | 6539 | 13539.5477 | 12958.1400 | 5777.9453 | 3802.0000 | 24250.8300 |

Percentiles:

| Variable | P25 | P75 |
|----------|----:|----:|
| Daily log return | -0.0048 | 0.0057 |
| Absolute daily return | 0.0024 | 0.0097 |
| Squared daily return | 0.0000 | 0.0001 |
| 30-day realised vol (annualised, DV2) | 0.0987 | 0.1739 |
| FTSE 250 index level | 7867.2500 | 19161.7700 |

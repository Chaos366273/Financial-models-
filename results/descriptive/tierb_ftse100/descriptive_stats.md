# Table B.1 — FTSE 100 descriptive statistics

Series: Bloomberg `ftse100_level` in `macro_wide.csv`. Returns: r_t = ln(P_t/P_{t-1}). DV2: sqrt(252) x 30-day sample SD. Complete cases (return + RV): n = 6,539 days; 2000-02-15 to 2025-12-31.

| Variable | n | Mean | Median | SD | Min | Max |
|----------|--:|-----:|-------:|---:|----:|----:|
| Daily log return (`ret`) | 6539 | 0.0001 | 0.0005 | 0.0113 | -0.1151 | 0.0938 |
| Absolute daily return (`abs_ret`) | 6539 | 0.0077 | 0.0053 | 0.0083 | 0.0000 | 0.1151 |
| Squared daily return (`ret_sq`) | 6539 | 0.0001 | 0.0000 | 0.0004 | 0.0000 | 0.0133 |
| 30-day realised vol (annualised, DV2) (`rv_30_ann`) | 6539 | 0.1572 | 0.1312 | 0.0894 | 0.0526 | 0.7420 |
| FTSE 100 index level (`ftse100_level`) | 6539 | 6266.5433 | 6280.1200 | 1222.4222 | 3287.0000 | 9940.7100 |

Percentiles:

| Variable | P25 | P75 |
|----------|----:|----:|
| Daily log return | -0.0049 | 0.0056 |
| Absolute daily return | 0.0024 | 0.0101 |
| Squared daily return | 0.0000 | 0.0001 |
| 30-day realised vol (annualised, DV2) | 0.0994 | 0.1852 |
| FTSE 100 index level | 5428.9250 | 7186.1900 |

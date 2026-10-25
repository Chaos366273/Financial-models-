# Champion — realistic_costs

Live book: IC-blend XGB+LSTM views into ML-BL, then GARCH(1,1) size.
RF omitted. LSTM VaR is not used as s. Top-50 Ensemble is not this book.

- XGB previews: 14 under `C:\Users\Faiaz\Desktop\Courses\sem2\dessertation\Dessertation\analysis\xgboost\walk_forward_baseline`
- LSTM previews: 14 under `C:\Users\Faiaz\Desktop\Courses\sem2\dessertation\Dessertation\analysis\lstm\Pruned 25, 14 step No Sentiment\walk_forward_baseline`
- mean_w_xgb=0.6955 mean_w_lstm=0.3045
- ML-BL weights: `C:\Users\Faiaz\Desktop\Courses\sem2\dessertation\Dessertation\analysis\tier c\ftse100\08_ml_black_litterman\ml_black_litterman_weights.csv`
- GARCH: `C:\Users\Faiaz\Desktop\Courses\sem2\dessertation\Dessertation\analysis\tier b\garch only\pooled_oos_daily.csv`

| n | Sharpe | Sortino | Net ann. | Excess ann. | Gross ann. | MDD | Turnover | Mean s | n skip |
|--:|-------:|--------:|---------:|-----------:|-----------:|----:|---------:|-------:|-------:|
| 167 | 0.2371 | 0.3915 | 0.0499 | 0.0365 | 0.2047 | -0.2906 | 0.7232 | 1.1756 | 1 |

## By evaluation year

| Year | n | Sharpe | Net ann. | Mean s |
|-----:|--:|-------:|---------:|-------:|
| 2012 | 11 | 1.1641 | 0.1396 | 1.3370 |
| 2013 | 12 | 0.7947 | 0.1401 | 1.2625 |
| 2014 | 12 | -0.7041 | -0.0914 | 1.0873 |
| 2015 | 12 | -0.8474 | -0.1444 | 1.3220 |
| 2016 | 12 | 2.5175 | 0.4058 | 0.9089 |
| 2017 | 12 | 2.3828 | 0.2633 | 1.2919 |
| 2018 | 12 | -0.1117 | -0.0140 | 0.9037 |
| 2019 | 12 | 0.8602 | 0.1641 | 1.3837 |
| 2020 | 12 | -0.2578 | -0.0515 | 1.1919 |
| 2021 | 12 | 0.1741 | 0.0275 | 1.2392 |
| 2022 | 12 | -0.2299 | -0.0321 | 0.9778 |
| 2023 | 12 | -0.7436 | -0.0608 | 1.1515 |
| 2024 | 12 | -1.0361 | -0.0485 | 1.1084 |
| 2025 | 12 | -0.3340 | 0.0083 | 1.3065 |

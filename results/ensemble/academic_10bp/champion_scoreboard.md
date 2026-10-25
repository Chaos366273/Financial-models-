# Champion — academic_10bp

Live book: IC-blend XGB+LSTM views into ML-BL, then GARCH(1,1) size.
RF omitted. LSTM VaR is not used as s. Top-50 Ensemble is not this book.

- XGB previews: 14 under `C:\Users\Faiaz\Desktop\Courses\sem2\dessertation\Dessertation\analysis\xgboost\walk_forward_baseline`
- LSTM previews: 14 under `C:\Users\Faiaz\Desktop\Courses\sem2\dessertation\Dessertation\analysis\lstm\Pruned 25, 14 step No Sentiment\walk_forward_baseline`
- mean_w_xgb=0.6955 mean_w_lstm=0.3045
- ML-BL weights: `C:\Users\Faiaz\Desktop\Courses\sem2\dessertation\Dessertation\analysis\tier c\ftse100\08_ml_black_litterman\ml_black_litterman_weights.csv`
- GARCH: `C:\Users\Faiaz\Desktop\Courses\sem2\dessertation\Dessertation\analysis\tier b\garch only\pooled_oos_daily.csv`

| n | Sharpe | Sortino | Net ann. | Excess ann. | Gross ann. | MDD | Turnover | Mean s | n skip |
|--:|-------:|--------:|---------:|-----------:|-----------:|----:|---------:|-------:|-------:|
| 167 | 1.1327 | 2.1275 | 0.1874 | 0.1740 | 0.2047 | -0.1928 | 0.7232 | 1.1756 | 1 |

## By evaluation year

| Year | n | Sharpe | Net ann. | Mean s |
|-----:|--:|-------:|---------:|-------:|
| 2012 | 11 | 2.0532 | 0.2447 | 1.3370 |
| 2013 | 12 | 1.5685 | 0.2628 | 1.2625 |
| 2014 | 12 | 0.1929 | 0.0410 | 1.0873 |
| 2015 | 12 | 0.1329 | 0.0261 | 1.3220 |
| 2016 | 12 | 3.5127 | 0.6032 | 0.9089 |
| 2017 | 12 | 3.4703 | 0.3766 | 1.2919 |
| 2018 | 12 | 0.5760 | 0.0907 | 0.9037 |
| 2019 | 12 | 1.7187 | 0.3132 | 1.3837 |
| 2020 | 12 | 0.6268 | 0.1505 | 1.1919 |
| 2021 | 12 | 1.1197 | 0.1653 | 1.2392 |
| 2022 | 12 | 0.6989 | 0.1132 | 0.9778 |
| 2023 | 12 | 0.2531 | 0.0701 | 1.1515 |
| 2024 | 12 | 0.0801 | 0.0602 | 1.1084 |
| 2025 | 12 | 0.5956 | 0.1102 | 1.3065 |

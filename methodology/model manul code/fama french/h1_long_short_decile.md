# Table 4.z - Decile long-short (economic magnitude)

Each month: long top 10% / short bottom 10% by predicted next-month excess return.
EW = equal-weight within each leg. VW = market-cap weights within each leg.
Reported value = mean monthly long-short spread (excess return units).

| Model | EW long-short | EW std | VW long-short | VW std | n_months (EW) | note |
|-------|---------------|--------|---------------|--------|---------------|------|
| Fama-French 5 (panel OLS) | - | - | - | - | 0 | n/a for ranking: market-wide factors => constant cross-section y_pred; months without pred dispersion skipped |
| LSTM | 0.0041 | 0.0385 | -0.0034 | 0.0495 | 167 | scored on LSTM preview stock-months |
| XGBoost (B_drop_bottom9) | -0.0277 | - | - | - | - | EW from walk-forward summary AVERAGE; VW N/A (full preview.csv not saved) |

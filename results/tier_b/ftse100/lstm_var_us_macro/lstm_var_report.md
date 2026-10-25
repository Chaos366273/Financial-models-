# LSTM quantile VaR (pinball)

Target = VaR (not realised vol). Features = GARCH-t / EGARCH / EWMA variance + lagged return, |r|, I(r<0), z + lag-1 daily `us_vix`, `|d us_10y|`, Baa–10y spread, `|d log brent|`, `|d log gbp|`.
Scaler and GARCH/EGARCH/EWMA fits use the train window only.
Features: garch_var, egarch_var, ewma_var, r_lag, abs_r_lag, neg_lag, z_lag, us_vix, abs_d_us10y, us_baa_spread, abs_d_log_brent, abs_d_log_gbp

| Model | n | VR | hit rate | Kupiec p | Christoffersen p |
|-------|--:|---:|---------:|---------:|-----------------:|
| LSTM-95 | 3537 | 1.0178 | 0.0509 | 0.8085 | 0.0096 |
| LSTM-99 | 3537 | 1.2157 | 0.0122 | 0.2123 | 0.3850 |

# LSTM quantile VaR (pinball)

Target = VaR (not realised vol). Features = GARCH-t / EGARCH / EWMA variance + lagged return, |r|, I(r<0), z + lagged monthly CSD (`csd_ret`, `pct_negative`, `cs_skew`) + Hamilton OOS `prob_bear`/`prob_bull`.
Scaler and GARCH/EGARCH/EWMA fits use the train window only.
Features: garch_var, egarch_var, ewma_var, r_lag, abs_r_lag, neg_lag, z_lag, csd_ret, pct_negative, cs_skew, prob_bear, prob_bull

| Model | n | VR | hit rate | Kupiec p | Christoffersen p |
|-------|--:|---:|---------:|---------:|-----------------:|
| LSTM-95 | 3537 | 1.3118 | 0.0656 | 0.0000 | 0.0000 |
| LSTM-99 | 3537 | 1.1309 | 0.0113 | 0.4434 | 0.5788 |

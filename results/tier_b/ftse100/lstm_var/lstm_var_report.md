# LSTM quantile VaR (pinball)

Target = VaR (not realised vol). Features = GARCH-t / EGARCH / EWMA variance + lagged return, |r|, I(r<0), z.
Scaler and GARCH/EGARCH/EWMA fits use the train window only.

| Model | n | VR | hit rate | Kupiec p | Christoffersen p |
|-------|--:|---:|---------:|---------:|-----------------:|
| LSTM-95 | 3537 | 1.0235 | 0.0512 | 0.7497 | 0.0499 |
| LSTM-99 | 3537 | 1.1592 | 0.0116 | 0.3534 | 0.4019 |

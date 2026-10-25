# LSTM quantile VaR (pinball)

Target = VaR (not realised vol). Features = GARCH-t / EGARCH / EWMA variance + lagged return, |r|, I(r<0), z.
Scaler and GARCH/EGARCH/EWMA fits use the train window only.

| Model | n | VR | hit rate | Kupiec p | Christoffersen p |
|-------|--:|---:|---------:|---------:|-----------------:|
| LSTM-95 | 3537 | 1.0744 | 0.0537 | 0.3159 | 0.1231 |
| LSTM-99 | 3537 | 1.3288 | 0.0133 | 0.0613 | 0.0643 |

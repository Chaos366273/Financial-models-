# XGB stack VaR (GARCH-t + EGARCH + EWMA + GEW avg)

No LSTM. Meta: XGBoost quantile. 2012 = GEW-average VaR passthrough.

| Model | n | VR | hit rate | Kupiec p | Christoffersen p |
|-------|--:|---:|---------:|---------:|-----------------:|
| Stack-95 | 3537 | 1.3797 | 0.0690 | 0.0000 | 0.0000 |
| Stack-99 | 3537 | 2.7142 | 0.0271 | 0.0000 | 0.0000 |

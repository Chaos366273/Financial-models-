# GEW walk-forward (Gaussian VaR z)

GARCH-t (vol only), EGARCH(1,1,1), EWMA lambda=0.94, average of the three daily variances.
VaR uses 1.645 / 2.326 on every model. Frozen Normal GARCH is Benchmark A.

| Model | n | RMSE | MAE | QLIKE | VaR95 VR | Kupiec95 p | VaR99 VR | Kupiec99 p |
|-------|--:|-----:|----:|------:|---------:|-----------:|---------:|-----------:|
| garch_t | 3537 | 0.0351 | 0.0233 | 0.0898 | 1.0404 | 0.5836 | 2.1487 | 0.0000 |
| egarch | 3537 | 0.0436 | 0.0275 | 0.1301 | 1.0008 | 0.9908 | 1.8094 | 0.0000 |
| ewma | 3537 | 0.0176 | 0.0116 | 0.0255 | 1.2101 | 0.0055 | 2.4032 | 0.0000 |
| gew_avg | 3537 | 0.0290 | 0.0189 | 0.0584 | 1.0970 | 0.1924 | 2.1204 | 0.0000 |

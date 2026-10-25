# GEW walk-forward (Gaussian VaR z)

GARCH-t (vol only), EGARCH(1,1,1), EWMA lambda=0.94, average of the three daily variances.
VaR uses 1.645 / 2.326 on every model. Frozen Normal GARCH is Benchmark A.

| Model | n | RMSE | MAE | QLIKE | VaR95 VR | Kupiec95 p | VaR99 VR | Kupiec99 p |
|-------|--:|-----:|----:|------:|---------:|-----------:|---------:|-----------:|
| garch_t | 3537 | 0.0378 | 0.0240 | 0.0853 | 1.2214 | 0.0035 | 2.2335 | 0.0000 |
| egarch | 3537 | 0.0480 | 0.0297 | 0.1357 | 1.0122 | 0.8685 | 1.6681 | 0.0003 |
| ewma | 3537 | 0.0196 | 0.0125 | 0.0258 | 1.1309 | 0.0799 | 2.4314 | 0.0000 |
| gew_avg | 3537 | 0.0320 | 0.0203 | 0.0588 | 1.1366 | 0.0680 | 2.0639 | 0.0000 |

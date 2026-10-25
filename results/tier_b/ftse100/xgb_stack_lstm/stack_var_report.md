# XGB stack VaR (EGARCH + EWMA + LSTM)

Meta-learner: XGBoost quantile (`reg:quantileerror`).
Features: `egarch_sigma`, `ewma_sigma`, `lstm_sigma` (implied from champion LSTM VaR).
2012 has no prior OOS years → LSTM passthrough. Later years train on previous OOS years only.

| Model | n | VR | hit rate | Kupiec p | Christoffersen p |
|-------|--:|---:|---------:|---------:|-----------------:|
| Stack-95 | 3537 | 1.3458 | 0.0673 | 0.0000 | 0.0000 |
| Stack-99 | 3537 | 2.3183 | 0.0232 | 0.0000 | 0.0000 |

Compare to frozen LSTM: 95 VR=1.023 Kupiec p=0.75; 99 VR=1.159 Kupiec p=0.35.

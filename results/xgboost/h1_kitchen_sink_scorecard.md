# XGBoost H1 — full characteristic set (14-step walk-forward)

Frozen HPs: 178 trees, depth 4, lr 0.03, min_child_weight 8, subsample/colsample 0.80, seed 42.
Target: `target_excess_return` (sector-neutral + 1/99 winsor on train for fitting only; score on raw y).
AIS.LN 2014-04 dropped as known data error.

| Metric | No sentiment | + sentiment |
|---|---:|---:|
| RMSE | 0.105 | 0.105 |
| MAE | 0.068 | 0.068 |
| R2 | -0.016 | -0.015 |
| Gu R2_oos | -0.006 | -0.004 |
| IC | 0.037 | 0.037 |
| LS / month | 0.0113 | 0.0112 |
| LS Sharpe | 1.19 | 1.28 |
| Top-50 edge | 0.0047 | 0.0048 |
| Hit rate | 0.604 | 0.586 |

See `figures/` for year-by-year charts.

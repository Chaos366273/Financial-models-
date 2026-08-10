# H1 comparability findings (2026-08-03)

## 1. The n_obs mismatch is mostly a table bug

| Model | What the H1 table showed | What the walk-forward actually used |
|-------|--------------------------|-------------------------------------|
| FF5 | 76,374 | 76,374 pooled stock-months |
| XGBoost | 5,455 | **76,374** (= sum of per-year `n_eval_rows`) |
| LSTM | 5,282 | **73,951** (= sum of per-year `n_eval_rows`) |

`5,455` / `5,282` are the **AVERAGE** row’s mean `n_eval_rows` per step, not the evaluation sample size.

Year-by-year: **FF5 n == XGB n every year** (difference = 0).  
LSTM is ~100–240 rows lighter per year because `seq_len=9` drops early history per ticker.

See `n_obs_by_year.csv`.

## 2. 2014 is one broken observation

- Max `target_excess_return` in 2014: **218.33** for **AIS.LN / 2014-04**
- That single point is **~99.9% of 2014 SSE**
- Top-5 extremes ≈ 99.9% of 2014 SSE (dominated by AIS)
- After dropping 1%/99% tails, 2014 RMSE falls from ~2.94 → **~0.07**

Plot: `2014_target_distribution.png`  
List: `2014_extreme_observations.csv`

## 3. Aligned sample (FF5 ∩ LSTM preview rows)

- Intersection size: **73,951** (all LSTM OOS rows)
- `y_true` match rate FF5 vs LSTM: **1.000** (identical target)
- Aligned pooled RMSE: FF5 **0.810** vs LSTM **0.811**
- Mean year RMSE: FF5 **0.313** vs LSTM **0.320**
- Median year RMSE: FF5 **0.103** vs LSTM **0.108**

See `aligned_ff5_lstm_rmse_by_year.csv`.

XGB full `preview.csv` was **not saved** (only `top50_picks.csv`), so a 3-way key intersection needs an XGB re-run that writes previews — but year counts already prove XGB used the same stock-months as FF5.

## 4. Target / leakage / scaling

| Item | FF5 | XGB / LSTM |
|------|-----|------------|
| OOS scored target | `target_excess_return` | same `target_excess_return` |
| Training target | raw factors → OLS on raw `target_excess_return` | `target_model` = sector-neutral + winsor 1/99 |
| Leakage | factors at t → next-month y | same timing; train winsor uses train window only |
| Scaling | none | train-only winsor; predictions scored on raw y |

Conclusion: fair on **target definition** for OOS metrics. Training objective differs (neutralisation), which is intentional for ML ranking, not leakage.

## 5. Long-short currently

Existing metric = equal-weight **top 10% minus bottom 10%** by `y_pred` each month (decile-style).  
Not yet value-weighted. Needs `market_cap` / `log_market_cap` join for VW top/bottom decile.

## Recommended next code changes (need approval)

1. Fix H1 table `n_obs` to pooled sums (76374 / 76374 / 73951).
2. Report year-mean **and** year-median RMSE; footnote 2014 / AIS.LN.
3. Optionally score all models on LSTM-eligible keys only (73,951) for strict equality.
4. Re-run XGB walk-forward once with full `preview.csv` saved for true 3-way join.
5. Add explicit EW + VW decile long-short table under `fama french/`.

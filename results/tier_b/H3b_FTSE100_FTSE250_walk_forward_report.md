# H3b walk-forward — FTSE 100 vs FTSE 250 volatility / VaR

**Analysis by:** Faiaz  
**Folder:** `Dessertation/analysis/tier b/h3b_walk_forward_report/`  
**Figures:** `figures/` (20 charts, generated 2026-08-19 from frozen CSVs; no source models were edited)

**Update 2026-08-19 (afternoon):** Section 5 expanded from a one-row robustness table to the full **how it was wired** + **14-year results** for US/UK daily macros, Markov+CSD hybrid, and both XGB VaR stacks. Champion LSTM is unchanged. FTSE 250 never ran `--macro-daily` or `--markov-csd` (those flags were FTSE 100 only).

This is the Tier-B counterpart of the XGBoost H1 14-step report. Same calendar grid. Different object.

| Item | Locked choice |
|---|---|
| Hypothesis | **H3b:** a GARCH–LSTM hybrid improves *risk estimation* vs GARCH(1,1) on UK index stress |
| Series | Index **levels** → daily log returns (`ftse100_level`, `ftse250_level`) |
| DV2 (vol) | \(\sqrt{252}\times\) 30-day SD of that index’s log returns (`rv_30_ann`). **Not** Bloomberg stock `volatility_30d` |
| Windows | 10-year train → 1-year eval, rolled annually, **2012–2025** (14 steps). Pooled OOS **n = 3,537** days on both indices |
| Vol metrics | RMSE, MAE, **QLIKE on variance** (Patton) |
| VaR metrics | exceptions, hit rate, **violation ratio**, Kupiec 95/99, Christoffersen independence / CC |
| VaR z (parametric) | Gaussian \(1.645\) / \(2.326\) for every GARCH/EGARCH/EWMA/GEW model (fair vol race) |
| LSTM | Pinball quantile net. Target = **VaR**, not RV. Champion = 7 features |

**Do not quote RMSE for LSTM.** The LSTM is not a realised-vol forecaster. Quote VR / Kupiec / Christoffersen.

Frozen source folders (read-only; this report does not overwrite them):

- FTSE 100: `garch only\`, `gew\`, `lstm var\`, plus robustness `markov garch lstm hybrid\`, `lstm var us macro\`, `xgb stack *`
- FTSE 250: `ftse 250\{garch only, gew, lstm var}`

---

## What was run (plain English)

Each eval year the last 10 years of **index** daily returns are used to fit a frozen recipe, then that recipe is filtered / scored on the next calendar year. That is done **14 times**.

| Spec | Role | What it outputs |
|---|---|---|
| Gaussian GARCH(1,1) | **Benchmark A** | one-step \(\sigma_t\), parametric VaR |
| GARCH-t vol + EGARCH(1,1,1) + EWMA \(\lambda=0.94\) + variance average | **GEW horse race** | same |
| LSTM quantile VaR (champion 7-feat) | **H3b main** | \(\mathrm{VaR}_{0.05}\), \(\mathrm{VaR}_{0.01}\) |
| LSTM + Markov/CSD; LSTM + US daily macros | Robustness (FTSE 100 only) | VaR |
| XGB VaR stacks | Failed meta-models (FTSE 100 only) | VaR |

Champion LSTM features (lag-1, train-window scaler): `garch_var`, `egarch_var`, `ewma_var`, `r_lag`, `abs_r_lag`, `neg_lag`, `z_lag`. Architecture: lookback 30, layers 32/16, dropout 0.2.

---

## How to read the metrics

- **Violation ratio (VR):** realised exceptions / expected exceptions. **1.00 is perfect.** VR \(> 1\) = too many hits (VaR too tight). VR \(< 1\) = too few hits (VaR too conservative).
- **Kupiec (UC):** tests whether the hit rate equals 5% or 1%. **Do not reject if \(p > 0.05\).** Year-by-year 99% tests are under-powered (~2–3 expected hits). **Pooled UC is the H3b coverage test.**
- **Christoffersen (IND / CC):** tests clustering of hits. A model can pass Kupiec and still fail independence (exceptions bunch in crashes).
- **RMSE vs 30-day RV:** EWMA will look “best” because \(\lambda=0.94\) is a smoother of the same object. That is **not** a VaR win.
- **2020:** the stress year. Every model is allowed to look bad here; the question is *how much* and whether LSTM still improves **99%** tails.

---

## Headline pooled table (quote this)

Pooled OOS 2012–2025, **3,537** days.

### Volatility accuracy (parametric family only)

![Pooled RMSE and QLIKE](figures/pooled_rmse_qlike.png)

| Model | FTSE 100 RMSE | FTSE 250 RMSE | FTSE 100 QLIKE | FTSE 250 QLIKE |
|---|---:|---:|---:|---:|
| GARCH(1,1) | 0.0367 | 0.0390 | 0.0977 | 0.0943 |
| GARCH-t (vol) | 0.0351 | 0.0378 | 0.0898 | 0.0853 |
| EGARCH | 0.0436 | 0.0480 | 0.1301 | 0.1357 |
| **EWMA** | **0.0176** | **0.0196** | **0.0255** | **0.0258** |
| GEW avg | 0.0290 | 0.0320 | 0.0584 | 0.0588 |

EWMA wins the *level* race on both indices for the same mechanical reason.

### 95% VaR coverage

![Pooled 95% VR](figures/pooled_vr95_bars.png)

| Model | FTSE 100 exceptions | VR | Kupiec p | Christ. p | FTSE 250 exceptions | VR | Kupiec p | Christ. p |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| GARCH(1,1) | 184 | 1.040 | 0.584 | 0.014 | 208 | 1.176 | **0.019** | 0.000 |
| GARCH-t | 184 | 1.040 | 0.584 | 0.014 | 216 | 1.221 | **0.003** | 0.000 |
| **EGARCH** | 177 | **1.001** | **0.991** | 0.007 | 179 | **1.012** | **0.869** | 0.009 |
| EWMA | 214 | 1.210 | **0.005** | 0.000 | 200 | 1.131 | 0.080 | 0.000 |
| GEW avg | 194 | 1.097 | 0.192 | 0.000 | 201 | 1.137 | 0.068 | 0.001 |
| **LSTM-95 (champion)** | **181** | **1.023** | **0.750** | 0.050 | **190** | **1.074** | **0.316** | 0.123 |

### 99% VaR coverage (this is where LSTM earns H3b)

![Pooled 99% VR](figures/pooled_vr99_bars.png)

| Model | FTSE 100 exceptions | VR | Kupiec p | Christ. p | FTSE 250 exceptions | VR | Kupiec p | Christ. p |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| GARCH(1,1) | 76 | 2.149 | 0.000 | 0.000 | 75 | 2.120 | 0.000 | 0.000 |
| GARCH-t | 76 | 2.149 | 0.000 | 0.000 | 79 | 2.234 | 0.000 | 0.000 |
| EGARCH | 64 | 1.809 | 0.000 | 0.000 | 59 | 1.668 | 0.000 | 0.001 |
| EWMA | 85 | 2.403 | 0.000 | 0.000 | 86 | 2.431 | 0.000 | 0.000 |
| GEW avg | 75 | 2.120 | 0.000 | 0.000 | 73 | 2.064 | 0.000 | 0.000 |
| **LSTM-99 (champion)** | **41** | **1.159** | **0.353** | **0.402** | **47** | **1.329** | **0.061** | **0.064** |

![Kupiec p-values](figures/kupiec_p_pooled.png)

**Read it once:**

1. **95%:** EGARCH is the closest VR to 1.0 on both indices. Champion LSTM-95 is statistically fine (Kupiec not rejected) and slightly conservative-to-calibrated. Gaussian GARCH **fails Kupiec on FTSE 250 95%** (p = 0.019).
2. **99%:** every parametric Gaussian-z model **fails** pooled Kupiec (VR ≈ 1.7–2.4). Champion LSTM **passes** on FTSE 100 (VR 1.16, p = 0.35) and is **borderline** on FTSE 250 (VR 1.33, p = 0.061). That is the H3b coverage result.
3. **Independence:** GARCH/EGARCH 95% still cluster hits (Christoffersen p small). LSTM-95 on FTSE 100 is on the 5% knife-edge (p = 0.0499). LSTM-99 independence is acceptable.

---

## 1. FTSE 100 — GARCH(1,1) baseline

Fit on train, freeze \((\mu,\omega,\alpha,\beta)\), one-step filter on the test year.

![GARCH σ vs RV, FTSE 100](figures/ftse100_garch_sigma_vs_rv.png)

| Year | n | RMSE | MAE | QLIKE | VaR95 VR | Kupiec95 p | VaR99 VR | Kupiec99 p | persistence |
|---:|--:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2012 | 252 | 0.0170 | 0.0141 | 0.0282 | 1.270 | 0.345 | 1.190 | 0.768 | 0.996 |
| 2013 | 253 | 0.0190 | 0.0153 | 0.0453 | 1.344 | 0.232 | 1.976 | 0.168 | 0.994 |
| 2014 | 253 | 0.0199 | 0.0157 | 0.0546 | 1.265 | 0.352 | 3.162 | 0.006 | 0.992 |
| 2015 | 253 | 0.0288 | 0.0214 | 0.0487 | 1.344 | 0.232 | 3.162 | 0.006 | 0.993 |
| 2016 | 253 | 0.0319 | 0.0229 | 0.0768 | 0.949 | 0.850 | 1.581 | 0.392 | 0.987 |
| 2017 | 252 | 0.0250 | 0.0224 | 0.1178 | 0.397 | 0.013 | 0.794 | 0.733 | 0.983 |
| 2018 | 253 | 0.0209 | 0.0165 | 0.0437 | 1.502 | 0.087 | 2.372 | 0.062 | 0.987 |
| 2019 | 253 | 0.0217 | 0.0171 | 0.0536 | 0.632 | 0.151 | 2.372 | 0.062 | 0.976 |
| **2020** | 254 | **0.0886** | 0.0587 | 0.1760 | 1.575 | 0.052 | **4.724** | 0.000 | 0.951 |
| 2021 | 253 | 0.0271 | 0.0219 | 0.0758 | 0.791 | 0.428 | 2.767 | 0.020 | 0.965 |
| 2022 | 250 | 0.0431 | 0.0323 | 0.1740 | 1.440 | 0.133 | 2.800 | 0.019 | 0.945 |
| 2023 | 251 | 0.0300 | 0.0214 | 0.1045 | 1.116 | 0.680 | 1.992 | 0.164 | 0.947 |
| 2024 | 254 | 0.0229 | 0.0201 | 0.0801 | 0.630 | 0.148 | 0.394 | 0.268 | 0.940 |
| 2025 | 253 | 0.0501 | 0.0355 | 0.2896 | 0.316 | 0.004 | 0.791 | 0.728 | 0.936 |
| **Pooled** | **3537** | **0.0367** | **0.0240** | **0.0977** | **1.040** | **0.584** | **2.149** | **0.000** | |

Persistence falls from 0.996 (2012 window) to 0.936 (2025 window). The 2020 window is the only year with RMSE 0.089. 2025 95% VR 0.32 is *over*-conservative (too few hits), not a crash miss.

---

## 2. FTSE 100 — GEW horse race (year story in charts)

![FTSE 100 RMSE by year](figures/ftse100_rmse_by_year.png)
![FTSE 100 95% VR by year](figures/ftse100_vr95_by_year.png)
![FTSE 100 99% VR by year](figures/ftse100_vr99_by_year.png)

Pooled (repeat for the write-up):

| Model | RMSE | MAE | QLIKE | VaR95 VR | Kupiec95 p | VaR99 VR | Kupiec99 p |
|---|---:|---:|---:|---:|---:|---:|---:|
| garch_t | 0.0351 | 0.0233 | 0.0898 | 1.040 | 0.584 | 2.149 | 0.000 |
| egarch | 0.0436 | 0.0275 | 0.1301 | **1.001** | **0.991** | 1.809 | 0.000 |
| ewma | **0.0176** | **0.0116** | **0.0255** | 1.210 | 0.005 | 2.403 | 0.000 |
| gew_avg | 0.0290 | 0.0189 | 0.0584 | 1.097 | 0.192 | 2.120 | 0.000 |

EGARCH 2020 95% VR = **1.89** (worse than GARCH 1.57 that year). Leverage helps *average* 95% calibration; it does not save the crash year by itself.

---

## 3. FTSE 100 — champion LSTM quantile VaR

![LSTM 95% exceptions, FTSE 100](figures/ftse100_lstm95_exceptions.png)
![2020 zoom LSTM vs GARCH](figures/ftse100_2020_var_zoom.png)

| Year | n | LSTM-95 exceptions | VR95 | Kupiec95 p | LSTM-99 exceptions | VR99 | Kupiec99 p |
|---:|--:|---:|---:|---:|---:|---:|---:|
| 2012 | 252 | 13 | 1.032 | 0.908 | 1 | 0.397 | 0.273 |
| 2013 | 253 | 11 | 0.870 | 0.627 | 6 | 2.372 | 0.062 |
| 2014 | 253 | 12 | 0.949 | 0.850 | 3 | 1.186 | 0.773 |
| 2015 | 253 | 20 | 1.581 | 0.050 | 4 | 1.581 | 0.392 |
| 2016 | 253 | 9 | 0.711 | 0.268 | 1 | 0.395 | 0.271 |
| 2017 | 252 | 4 | 0.317 | 0.004 | 2 | 0.794 | 0.733 |
| 2018 | 253 | 22 | 1.739 | 0.014 | 4 | 1.581 | 0.392 |
| 2019 | 253 | 11 | 0.870 | 0.627 | 3 | 1.186 | 0.773 |
| **2020** | 254 | **32** | **2.520** | **0.000** | **11** | **4.331** | **0.000** |
| 2021 | 253 | 8 | 0.632 | 0.151 | 3 | 1.186 | 0.773 |
| 2022 | 250 | 14 | 1.120 | 0.669 | 1 | 0.400 | 0.278 |
| 2023 | 251 | 9 | 0.717 | 0.280 | 1 | 0.398 | 0.276 |
| 2024 | 254 | 10 | 0.787 | 0.420 | **0** | **0.000** | — |
| 2025 | 253 | 6 | 0.474 | 0.033 | 1 | 0.395 | 0.271 |
| **Pooled** | **3537** | **181** | **1.023** | **0.750** | **41** | **1.159** | **0.353** |

2024 LSTM-99 has **zero** exceptions (Kupiec LR undefined). Treat as conservative that year, not as a model crash.

**2020 is still the break:** LSTM-95 VR 2.52 vs GARCH 1.57. LSTM is **not** a 2020 miracle. The H3b claim is **pooled 99% tails**, not crash-year 95%.

---

## 4. SHAP — what the champion LSTM used (FTSE 100 only)

GradientExplainer on CPU, mean |SHAP| over batch and time (same squeeze rule as H1 LSTM). Retrain-per-window; weights were never saved.

![Pooled SHAP LSTM-95](figures/shap_lstm95_pooled.png)
![Pooled SHAP LSTM-99](figures/shap_lstm99_pooled.png)
![SHAP heatmap LSTM-95](figures/shap_lstm95_heatmap.png)

**Pooled mean |SHAP|**

| Rank | LSTM-95 | |SHAP| | LSTM-99 | |SHAP| |
|---:|---|---:|---|---:|
| 1 | `r_lag` | 0.000125 | `r_lag` | 0.000148 |
| 2 | `egarch_var` | 0.000112 | `neg_lag` | 0.000136 |
| 3 | `z_lag` | 0.000103 | `garch_var` | 0.000123 |
| 4 | `garch_var` | 0.000093 | `z_lag` | 0.000121 |
| 5 | `neg_lag` | 0.000093 | `egarch_var` | 0.000121 |
| 6 | `abs_r_lag` | 0.000077 | `abs_r_lag` | 0.000088 |
| 7 | `ewma_var` | 0.000061 | `ewma_var` | 0.000084 |

The net is **not** “just copying EWMA”. Lagged return and EGARCH variance dominate 95%. The 99% head uses **sign of yesterday’s return** (`neg_lag`) harder — that is the tail / leverage channel. EWMA is last in both heads: useful as a level feature, not the story the network trades on.

---

## 5. How extras were integrated (FTSE 100 only)

Same LSTM (lookback 30, 32/16, dropout 0.2, pinball). **Do not combine** `--markov-csd` and `--macro-daily`. `--shap` is champion-only.

![95% VR robustness](figures/robustness_vr95_by_year.png)
![99% VR robustness](figures/robustness_vr99_by_year.png)

### 5.0 Wiring (what was actually concatenated)

Champion always has 7 columns: `garch_var`, `egarch_var`, `ewma_var`, `r_lag`, `abs_r_lag`, `neg_lag`, `z_lag`. GARCH-t / EGARCH / EWMA are **fit on the train window only**, then filtered on test.

| Arm | CLI | Extra columns (lag-1, no look-ahead) | Feature count | Log | Out folder |
|---|---|---|---:|---|---|
| Champion | (default) | none | 7 | `lstm var\` | `lstm var\` |
| US/UK macros | `--macro-daily` | `us_vix`, `abs_d_us10y`, `us_baa_spread`, `abs_d_log_brent`, `abs_d_log_gbp` | **12** | `lstm_var_run_20260819_000704.txt` | `lstm var us macro\` |
| Markov + CSD | `--markov-csd` | `csd_ret`, `pct_negative`, `cs_skew`, `prob_bear`, `prob_bull` | **12** | `lstm_var_run_20260818_230118.txt` | `markov garch lstm hybrid\` |

**US/UK macros — construction**

- US FRED file: `data\final data for AI\us data\us_macro_daily.csv`  
  `us_vix` = VIXCLS, `us_10y_treasury` = **DGS10 (US 10y, not UK gilt)**, `us_baa_10y_spread` = BAA10Y.
- Bloomberg overlay: `data\final data for AI\05_macro\macro_wide.csv` → `brent_crude`, `gbp_usd`.
- Transforms, all **shifted by 1 calendar day** so day *t* uses information known at *t−1*:
  - `us_vix_t` ← VIX_{t−1}
  - `abs_d_us10y_t` ← |Δ DGS10|_{t−1}
  - `us_baa_spread_t` ← BAA10Y_{t−1}
  - `abs_d_log_brent_t` ← |Δ log Brent|_{t−1}
  - `abs_d_log_gbp_t` ← |Δ log GBPUSD|_{t−1}
- 6,871 macro rows after merge. Not a UK 10y gilt / VFTSE / UK credit spread.

**Markov + CSD — construction**

- Markov: `methodology\model manul code\marvok\market_appendix\pooled_market_forecast.csv`  
  Hamilton **OOS** `regime_prob_bear` (monthly). Mapped onto calendar days of that month. `prob_bull = 1 − prob_bear`. Missing days filled **0.5**.  
  This is a **P-roll with no updating on the test month** — the filter does not re-estimate inside March 2020, so `prob_bear` does **not** spike when the crash hits.
- CSD: `07_panel\master_panel.parquet` monthly stock returns (AIS.LN Apr 2014 dropped). Cross-section per month: `std`, share negative, skew. Then **`shift(1)` one month** so January 2020 uses December 2019 CSD. 284 CSD months, 5,083 Markov daily days. No daily stock parquet on this machine.

Pooled comparison (same 3,537 FTSE 100 days):

| Spec | 95 hits | VR95 | Kupiec95 p | CC95 p | 99 hits | VR99 | Kupiec99 p | CC99 p | Verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Champion LSTM | 181 | **1.023** | **0.750** | 0.050 | 41 | **1.159** | **0.353** | 0.402 | **Keep** |
| + US/UK daily macros | 180 | 1.018 | 0.809 | 0.010 | 43 | 1.216 | 0.212 | 0.385 | 95 similar; **99 worse** |
| + Markov OOS + CSD | 232 | 1.312 | **0.000** | 0.000 | 40 | 1.131 | 0.443 | 0.579 | **Hurts 95% badly** |
| XGB stack LSTM+EGARCH+EWMA | 238 | 1.346 | 0.000 | 0.000 | 82 | 2.318 | 0.000 | 0.000 | Fail |
| XGB stack parametric GEW | 244 | 1.380 | 0.000 | 0.000 | 96 | 2.714 | 0.000 | 0.000 | Fail |

---

### 5.1 LSTM + US/UK daily macros — year by year

Log: `lstm var us macro\lstm_var_run_20260819_000704.txt`. Device CUDA. Features listed in that log match the 12-column list above.

| Year | n | ex95 | VR95 | Kupiec95 p | ex99 | VR99 | Kupiec99 p |
|---:|--:|---:|---:|---:|---:|---:|---:|
| 2012 | 252 | 8 | 0.635 | 0.155 | 6 | 2.381 | 0.061 |
| 2013 | 253 | 13 | 1.028 | 0.920 | 3 | 1.186 | 0.773 |
| 2014 | 253 | 16 | 1.265 | 0.352 | 4 | 1.581 | 0.392 |
| 2015 | 253 | 17 | 1.344 | 0.232 | 4 | 1.581 | 0.392 |
| 2016 | 253 | 8 | 0.632 | 0.151 | 5 | 1.976 | 0.168 |
| 2017 | 252 | 3 | 0.238 | 0.001 | 1 | 0.397 | 0.273 |
| 2018 | 253 | 21 | 1.660 | 0.027 | 2 | 0.791 | 0.728 |
| 2019 | 253 | 10 | 0.791 | 0.428 | 3 | 1.186 | 0.773 |
| **2020** | 254 | **41** | **3.228** | **0.000** | **9** | **3.543** | **0.002** |
| 2021 | 253 | 7 | 0.553 | 0.076 | 2 | 0.791 | 0.728 |
| 2022 | 250 | 5 | 0.400 | 0.014 | 1 | 0.400 | 0.278 |
| 2023 | 251 | 9 | 0.717 | 0.280 | 1 | 0.398 | 0.276 |
| 2024 | 254 | 13 | 1.024 | 0.931 | 0 | 0.000 | — |
| 2025 | 253 | 9 | 0.711 | 0.268 | 2 | 0.791 | 0.728 |
| **Pooled** | **3537** | **180** | **1.018** | **0.809** | **43** | **1.216** | **0.212** |

Vs champion in **2020**: macro 95% VR **3.23** (champion 2.52). VIX/credit/FX **did not** buy a better crash week; they made 95% **worse**. 99% 2020 VR 3.54 vs champion 4.33 — a small 99% improvement in the crash year that is **not** enough to beat pooled 99% of the 7-feature net (1.22 vs 1.16).

Christoffersen 95% pooled p = **0.010** (independence fail). Champion was already on the 0.05 knife-edge; macros make clustering worse.

---

### 5.2 LSTM + Markov OOS + monthly CSD — year by year

Log: `markov garch lstm hybrid\lstm_var_run_20260818_230118.txt`. 2012 train sequences = 2,415 (slightly fewer than 2,497 because of CSD/Markov NA at the start of the panel).

| Year | n | ex95 | VR95 | Kupiec95 p | ex99 | VR99 | Kupiec99 p |
|---:|--:|---:|---:|---:|---:|---:|---:|
| 2012 | 252 | 14 | 1.111 | 0.691 | 0 | 0.000 | — |
| **2013** | 253 | **38** | **3.004** | **0.000** | 2 | 0.791 | 0.728 |
| 2014 | 253 | 19 | 1.502 | 0.087 | 0 | 0.000 | — |
| 2015 | 253 | 14 | 1.107 | 0.702 | 2 | 0.791 | 0.728 |
| 2016 | 253 | 14 | 1.107 | 0.702 | 3 | 1.186 | 0.773 |
| 2017 | 252 | 6 | 0.476 | 0.034 | 0 | 0.000 | — |
| 2018 | 253 | 18 | 1.423 | 0.146 | 4 | 1.581 | 0.392 |
| 2019 | 253 | 8 | 0.632 | 0.151 | 3 | 1.186 | 0.773 |
| **2020** | 254 | **50** | **3.937** | **0.000** | **18** | **7.087** | **0.000** |
| 2021 | 253 | 6 | 0.474 | 0.033 | 2 | 0.791 | 0.728 |
| 2022 | 250 | 20 | 1.600 | 0.044 | 2 | 0.800 | 0.742 |
| 2023 | 251 | 6 | 0.478 | 0.035 | 1 | 0.398 | 0.276 |
| 2024 | 254 | 8 | 0.630 | 0.148 | 2 | 0.787 | 0.724 |
| 2025 | 253 | 11 | 0.870 | 0.627 | 1 | 0.395 | 0.271 |
| **Pooled** | **3537** | **232** | **1.312** | **0.000** | **40** | **1.131** | **0.443** |

This is the diagnostic, not a better model:

- **2013** already blows 95% (VR 3.00) — the extra slow-moving monthly features destabilise a calm year.
- **2020 95% VR 3.94** (champion 2.52; macros 3.23). **2020 99% VR 7.09** (champion 4.33). The hybrid is the **worst** crash-year VaR of the three LSTMs.
- Pooled 99% *looks* fine (VR 1.13, p = 0.44) because many years have **zero** 99% hits (2012, 2014, 2017) — the net became too conservative in quiet years and then dumped 18 nines in 2020. That is not a well-calibrated tail model.

**Why Markov could not help 2020:** OOS Hamilton probabilities are produced with a fixed transition matrix rolled forward. They do not update on March daily returns. CSD is last month’s cross-section. The LSTM therefore sees **stale** “regime” and **stale** breadth into the crash.

---

### 5.3 XGB VaR stacks — year by year (FTSE 100)

Expanding OOS meta. **2012 = passthrough** (not enough history to fit XGB). Not a replacement for the champion.

**Stack A** — LSTM + EGARCH + EWMA → XGB quantile (`xgb stack lstm egarch ewma\`, log `stack_var_run_20260819_003749.txt`). 2012 meta = `lstm_passthrough`.

| Year | meta | VR95 | Kupiec95 p | VR99 | Kupiec99 p |
|---:|---|---:|---:|---:|---:|
| 2012 | lstm_passthrough | 1.032 | 0.908 | 0.397 | 0.273 |
| 2013 | xgb | 0.949 | 0.850 | 1.976 | 0.168 |
| 2014 | xgb | 1.186 | 0.510 | 1.976 | 0.168 |
| 2015 | xgb | 2.213 | 0.000 | 5.929 | 0.000 |
| 2016 | xgb | 2.134 | 0.000 | 4.348 | 0.000 |
| 2017 | xgb | 0.476 | 0.034 | 0.397 | 0.273 |
| 2018 | xgb | 1.502 | 0.087 | 1.581 | 0.392 |
| 2019 | xgb | 0.791 | 0.428 | 1.976 | 0.168 |
| 2020 | xgb | 2.835 | 0.000 | 7.480 | 0.000 |
| 2021 | xgb | 0.870 | 0.627 | 2.372 | 0.062 |
| 2022 | xgb | 1.760 | 0.012 | 2.000 | 0.162 |
| 2023 | xgb | 1.275 | 0.337 | 1.195 | 0.763 |
| 2024 | xgb | 0.866 | 0.617 | 0.000 | — |
| 2025 | xgb | 0.949 | 0.850 | 0.791 | 0.728 |
| **Pooled** | | **1.346** | **0.000** | **2.318** | **0.000** |

**Stack B** — GARCH-t + EGARCH + EWMA + GEW avg, no LSTM (`xgb stack garch egarch ewma gew\`, log `stack_var_run_20260819_004817.txt`). 2012 meta = `gew_avg_passthrough`.

| Year | meta | VR95 | Kupiec95 p | VR99 | Kupiec99 p |
|---:|---|---:|---:|---:|---:|
| 2012 | gew_avg_passthrough | 1.190 | 0.500 | 1.190 | 0.768 |
| 2013 | xgb | 1.028 | 0.920 | 1.976 | 0.168 |
| 2014 | xgb | 1.265 | 0.352 | 2.767 | 0.020 |
| 2015 | xgb | 1.897 | 0.003 | 6.719 | 0.000 |
| 2016 | xgb | 1.976 | 0.002 | 7.115 | 0.000 |
| 2017 | xgb | 0.476 | 0.034 | 0.397 | 0.273 |
| 2018 | xgb | 1.502 | 0.087 | 1.581 | 0.392 |
| 2019 | xgb | 0.870 | 0.627 | 1.186 | 0.773 |
| 2020 | xgb | 2.913 | 0.000 | 8.268 | 0.000 |
| 2021 | xgb | 1.028 | 0.920 | 2.372 | 0.062 |
| 2022 | xgb | 1.760 | 0.012 | 2.000 | 0.162 |
| 2023 | xgb | 1.355 | 0.220 | 1.195 | 0.763 |
| 2024 | xgb | 1.260 | 0.360 | 0.394 | 0.268 |
| 2025 | xgb | 0.791 | 0.428 | 0.791 | 0.728 |
| **Pooled** | | **1.380** | **0.000** | **2.714** | **0.000** |

Both stacks over-hit in 2015–16 and explode in 2020. Meta-learning on ~250 points/year is too thin for 1% VaR.

**FTSE 250:** these three robustness arms were **not** run. Only GARCH / GEW / champion LSTM exist for 250.

---

## 6. FTSE 250 — GARCH(1,1) baseline

Same recipe, `ftse250_level`. (The auto-written `garch11_walk_forward_report.md` in that folder still says “FTSE 100” in the header; the **numbers below are FTSE 250**.)

![GARCH σ vs RV, FTSE 250](figures/ftse250_garch_sigma_vs_rv.png)

| Year | n | RMSE | MAE | QLIKE | VaR95 VR | Kupiec95 p | VaR99 VR | Kupiec99 p | persistence |
|---:|--:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2012 | 252 | 0.0182 | 0.0146 | 0.0279 | 1.032 | 0.908 | 1.587 | 0.388 | 0.989 |
| 2013 | 253 | 0.0218 | 0.0180 | 0.0589 | 1.107 | 0.702 | 2.372 | 0.062 | 0.990 |
| 2014 | 253 | 0.0195 | 0.0155 | 0.0427 | 1.265 | 0.352 | 2.372 | 0.062 | 0.988 |
| 2015 | 253 | 0.0228 | 0.0169 | 0.0470 | 1.107 | 0.702 | 2.372 | 0.062 | 0.984 |
| 2016 | 253 | 0.0524 | 0.0319 | 0.1171 | 1.186 | 0.510 | 3.162 | 0.006 | 0.979 |
| 2017 | 252 | 0.0298 | 0.0281 | 0.1866 | 0.397 | 0.013 | 0.397 | 0.273 | 0.976 |
| 2018 | 253 | 0.0232 | 0.0185 | 0.0619 | 1.502 | 0.087 | 1.976 | 0.168 | 0.988 |
| 2019 | 253 | 0.0258 | 0.0205 | 0.0758 | 0.870 | 0.627 | 1.581 | 0.392 | 0.978 |
| **2020** | 254 | **0.0926** | 0.0621 | 0.1777 | 1.575 | 0.052 | **3.543** | 0.002 | 0.956 |
| 2021 | 253 | 0.0234 | 0.0191 | 0.0613 | 1.107 | 0.702 | 1.976 | 0.168 | 0.967 |
| 2022 | 250 | 0.0469 | 0.0350 | 0.1492 | **1.840** | **0.006** | 2.800 | 0.019 | 0.947 |
| 2023 | 251 | 0.0334 | 0.0265 | 0.1057 | 1.195 | 0.491 | 1.594 | 0.384 | 0.957 |
| 2024 | 254 | 0.0219 | 0.0164 | 0.0525 | 1.181 | 0.519 | 1.575 | 0.396 | 0.963 |
| 2025 | 253 | 0.0421 | 0.0275 | 0.1565 | 1.107 | 0.702 | 2.372 | 0.062 | 0.965 |
| **Pooled** | **3537** | **0.0390** | **0.0250** | **0.0943** | **1.176** | **0.019** | **2.120** | **0.000** | |

Mid-cap 2016 RMSE is worse than FTSE 100 (0.052 vs 0.032) — Brexit hits 250 harder. 2022 95% coverage **fails** on 250 (VR 1.84, p = 0.006) while 100 was still acceptable (VR 1.44, p = 0.13).

---

## 7. FTSE 250 — GEW + champion LSTM

![FTSE 250 RMSE by year](figures/ftse250_rmse_by_year.png)
![FTSE 250 95% VR by year](figures/ftse250_vr95_by_year.png)
![FTSE 250 99% VR by year](figures/ftse250_vr99_by_year.png)
![LSTM 95% exceptions, FTSE 250](figures/ftse250_lstm95_exceptions.png)

**GEW pooled**

| Model | RMSE | MAE | QLIKE | VaR95 VR | Kupiec95 p | VaR99 VR | Kupiec99 p |
|---|---:|---:|---:|---:|---:|---:|---:|
| garch_t | 0.0378 | 0.0240 | 0.0853 | 1.221 | 0.003 | 2.234 | 0.000 |
| egarch | 0.0480 | 0.0297 | 0.1357 | **1.012** | **0.869** | 1.668 | 0.000 |
| ewma | **0.0196** | **0.0125** | **0.0258** | 1.131 | 0.080 | 2.431 | 0.000 |
| gew_avg | 0.0320 | 0.0203 | 0.0588 | 1.137 | 0.068 | 2.064 | 0.000 |

**LSTM by year**

| Year | n | LSTM-95 VR | Kupiec95 p | LSTM-99 VR | Kupiec99 p |
|---:|--:|---:|---:|---:|---:|
| 2012 | 252 | 1.667 | 0.026 | 1.587 | 0.388 |
| 2013 | 253 | 0.395 | 0.012 | 0.791 | 0.728 |
| 2014 | 253 | 0.711 | 0.268 | 0.791 | 0.728 |
| 2015 | 253 | 0.949 | 0.850 | 0.791 | 0.728 |
| 2016 | 253 | 1.344 | 0.232 | 1.976 | 0.168 |
| 2017 | 252 | 0.159 | 0.000 | 0.397 | 0.273 |
| 2018 | 253 | 0.949 | 0.850 | 1.186 | 0.773 |
| 2019 | 253 | 1.186 | 0.510 | 1.581 | 0.392 |
| **2020** | 254 | **2.677** | **0.000** | **4.331** | **0.000** |
| 2021 | 253 | 0.791 | 0.428 | 0.791 | 0.728 |
| 2022 | 250 | 1.280 | 0.329 | 0.800 | 0.742 |
| 2023 | 251 | 0.797 | 0.445 | 0.000 | — |
| 2024 | 254 | 1.102 | 0.713 | 1.575 | 0.396 |
| 2025 | 253 | 1.028 | 0.920 | 1.976 | 0.168 |
| **Pooled** | **3537** | **1.074** | **0.316** | **1.329** | **0.061** |

2023 LSTM-99: zero exceptions (same caveat as FTSE 100 2024).

---

## 8. Window-by-window story (use VR, not RMSE)

- **2012–2015.** Quiet-to-normal. GARCH 95% a bit hot (VR ~1.3 on 100). LSTM-95 close to 1 on 100; 250 LSTM-95 is noisy in 2012–13 (small-year power).
- **2016 (Brexit).** FTSE 250 GARCH RMSE **0.052** vs 100 **0.032**. LSTM-95 on 250 VR 1.34 (acceptable). 99% still thin.
- **2017.** Too *few* 95% hits everywhere (VR ~0.16–0.40). Quiet year + high persistence = conservative VaR. Not a model bug.
- **2018–19.** LSTM-95 on 100 is hot in 2018 (VR 1.74, Kupiec p = 0.014) then calm in 2019.
- **2020 (COVID).** RMSE jumps to ~0.09 on both indices. **Every** 99% VR > 3. LSTM-95 VR 2.52 (100) / 2.68 (250). GARCH 95% VR 1.57 both. The hybrid does **not** de-risk the crash week in time. State that plainly.
- **2021–24.** Coverage comes back. LSTM-99 is often *conservative* (VR < 1).
- **2022 (rate shock).** GARCH 95% fails on **250** (VR 1.84) but not on 100. LSTM-95 on 250 VR 1.28 (passes). Mid-cap tails are the harder object.
- **2025.** GARCH 95% on 100 is over-conservative (VR 0.32). LSTM-95 also low (0.47). 250 GARCH 95% is fine (1.11).

---

## 9. FTSE 100 vs FTSE 250 (replication, not a second hypothesis)

| Question | FTSE 100 | FTSE 250 |
|---|---|---|
| GARCH 95% pooled Kupiec | Pass (p = 0.58) | **Fail** (p = 0.019) |
| EGARCH 95% VR | 1.001 | 1.012 |
| LSTM-95 pooled VR | 1.023 | 1.074 |
| LSTM-99 pooled VR | **1.159 (p = 0.35)** | **1.329 (p = 0.061)** |
| Parametric 99% | Fail both (VR ~2.1) | Fail both (VR ~2.1) |
| EWMA RMSE | Lowest | Lowest |
| 2020 | Break year | Break year (slightly worse RMSE) |

The ranking **replicates**: EGARCH for 95% calibration, LSTM for 99% tails, EWMA for RV tracking, 2020 unsolved. 250 is a tougher 95% problem for Gaussian GARCH.

---

## Bottom line for the write-up

| Question | Answer from these runs |
|---|---|
| Does GARCH(1,1) track 30-day RV? | Adequately in calm years. **2020 RMSE ~0.09** on both indices. |
| Is EWMA the “best vol model”? | **Best RMSE/QLIKE**, worst-ish 99% VaR. Do not use RMSE to pick a risk model. |
| Best *parametric* 95% coverage? | **EGARCH** (VR ≈ 1.00 / 1.01). |
| Does LSTM beat GARCH on 95%? | Coverage is similar on 100; LSTM **rescues 95% Kupiec on 250** (GARCH fails, LSTM passes). |
| Does LSTM beat GARCH on 99%? | **Yes, pooled.** 100: VR 1.16 vs 2.15. 250: VR 1.33 vs 2.12. This is the H3b sentence. |
| Did LSTM “see 2020 coming”? | **No.** 95% VR 2.52 / 2.68. Same break as H1 RMSE in 2020. |
| Macros / Markov / XGB stack? | Full year tables in §5. Macros: 2020 95% VR **3.23** (worse). Markov: 2020 95% VR **3.94**, 99% **7.09**. Stacks fail UC. Champion stays 7-feat. |
| What the LSTM listens to | Yesterday’s return + EGARCH variance (95%); sign of return + GARCH variance (99%). Not EWMA. |
| Index vs stock | All of this is **one index series**. It can scale a book (vol overlay). It cannot fill a 100-name covariance diagonal. |

**H3b table to quote:** Section “Headline pooled table”, champion LSTM-99 vs GARCH(1,1), both indices.  
**Do not quote:** EWMA RMSE as proof of better *risk* forecasts; XGB-stack VRs; Markov 95% VR 1.31.

---

## File map (frozen inputs)

```
Dessertation/analysis/tier b/
  garch only/pooled_oos_daily.csv
  gew/pooled_{garch_t,egarch,ewma,gew_avg}_daily.csv
  lstm var/pooled_lstm{95,99}_daily.csv
  lstm var/shap/
  lstm var us macro/          -- --macro-daily (12 feat); log 20260819_000704
  markov garch lstm hybrid/   -- --markov-csd (12 feat); log 20260818_230118
  xgb stack lstm egarch ewma/
  xgb stack garch egarch ewma gew/
  ftse 250/garch only/walk_forward_baseline/pooled_oos_daily.csv
  ftse 250/gew/pooled_*_daily.csv
  ftse 250/lstm var/pooled_lstm{95,99}_daily.csv
  h3b_walk_forward_report/          ← this report (new)
```

# Ensemble Attribution Report (2026-08-05T16:23:30)

Architecture **B**: XGB + LSTM → blend `y_pred` → ranked book → trading PPO times BUY/SELL/HOLD.

## 1. Blend weight on XGB (`w` in `y_pred = w·xgb + (1-w)·lstm`)

- Rows: **58622**
- Mean `w_xgb`: **0.591** (median 0.591, std 0.000)
- XGB-heavy (`w>0.7`): 0.0%
- Mid (`0.3–0.7`): 100.0%
- LSTM-heavy (`w<0.3`): 0.0%

### Histogram of `w_xgb`

| Bin | Share |
|-----|-------|
| 0.0-0.2 | 0.0% |
| 0.2-0.4 | 0.0% |
| 0.4-0.6 | 100.0% |
| 0.6-0.8 | 0.0% |
| 0.8-1.0 | 0.0% |

## 2. Forecast quality (same ranked rows)

| Leg | MSE | R² | Dir acc | N |
|-----|-----|----|---------|---|
| XGB | 0.009246 | -0.0105 | 0.506 | 58622 |
| LSTM | 0.010533 | -0.1511 | 0.507 | 58622 |
| Blend | 0.009386 | -0.0257 | 0.509 | 58622 |

## 3. Counterfactual monthly long books

Equal-weight top-10 with `score > 0`, plus Arch B weighted book.

| Book | Mean monthly | Cum (compound) | Hit rate | Months |
|------|--------------|----------------|----------|--------|
| XGB-only | 0.8% | 166.5% | 54.5% | 167 |
| LSTM-only | 0.5% | 32.5% | 52.1% | 167 |
| Blend `y_pred` | 0.7% | 91.6% | 53.9% | 167 |
| Arch B weighted | 0.9% | 199.1% | 52.7% | 167 |

On Arch B selected longs (n=3340): blend ≈ **59.1% XGB / 40.9% LSTM**.

## 4. Trading PPO (Stage 2)

- Total return: **-1.9%**
- Sharpe: **0.040**
- Max DD: **2.1%**
- Win rate: **33.3%**
- Trades / steps: 21 / 31

> Monthly book cum returns span Stage-1 test months. Trading PPO return is a short daily execution window (often 1k-timestep smoke) — compare directionally, not as like-for-like total return.

## 5. Trade ↔ Stage-1 tags

- Trades tagged: **21** (BUY=12, closes=9)
- BUY avg Stage-1 book weight: **0.0060**
- BUY avg blend mix: **59.1% XGB / 40.9% LSTM**
- Close PnL sum: **-1846.67**
- CSV: `C:\Users\Faiaz\Desktop\Courses\sem2\dessertation\methodology\model manul code\esemble\saved_data\trade_attribution.csv`

> Trades have no market calendar date; tags use per-ticker averages from Stage-1 ranked_signals (not the exact month of the fill).

## Interpretation

- y_pred = w*xgb + (1-w)*lstm. High mean_w_xgb => book is XGB-driven; low => LSTM-driven. Trading PPO does not choose the leg.
- Monthly book cum returns span Stage-1 test months. Trading PPO return is a short daily execution window (often 1k-timestep smoke) — compare directionally, not as like-for-like total return.

JSON: `C:\Users\Faiaz\Desktop\Courses\sem2\dessertation\methodology\model manul code\esemble\saved_data\attribution_report.json`

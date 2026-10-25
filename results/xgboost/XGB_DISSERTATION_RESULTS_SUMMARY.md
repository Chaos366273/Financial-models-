# XGBoost dissertation results summary

**Model:** B_drop_bottom9 (19 features from h1_clean)  
**Hyper-parameters:** depth=4, learning_rate=0.03, min_child_weight=8, trees=178  
**Target:** Sector-neutral excess return, winsorised at 1%/99%  
**Liquidity (primary):** no filter (`LIQUIDITY_FILTER=False`)  
**Test status:** LOCKED (run `run_xgb.bat --final-test` once when ready)  

## Table 1 — Main ranking / portfolio metrics

| Metric                               | Validation (2023)   | Test (2024–2025)     |
|:-------------------------------------|:--------------------|:---------------------|
| Spearman IC (monthly mean)           | 0.0444              | Locked / not yet run |
| Long–short (top–bottom 10%)          | 2.60%               | Locked / not yet run |
| Top-50 edge vs equal-weight market   | 1.62%               | Locked / not yet run |
| Top-50 hit rate (months beat market) | 91.7%               | Locked / not yet run |
| Top-50 avg monthly turnover          | 62%                 | Locked / not yet run |
| RMSE (level)                         | 0.1011              | Locked / not yet run |

## Table 2 — Validation robustness (do not retune on these)

| Check                          | Spec / result                                                        |
|:-------------------------------|:---------------------------------------------------------------------|
| Feature set                    | B_drop_bottom9 (19 feats from h1_clean); A_top16 secondary only      |
| Hyper-parameters               | depth=4, lr=0.03, min_child_weight=8, trees=178                      |
| Target (frozen)                | Sector-neutral excess return; winsor 1%/99%                          |
| Exp1 market-neutral Y only     | IC 0.0503; Top-50 edge 0.99%; hit 66.7% → reject for book            |
| Exp2 tighter winsor 0.5%/99.5% | IC noisier; imp. stability weaker → keep 1%/99%                      |
| Exp3 no demean                 | IC 0.0236; Top-50 edge ~0; hit 41.7% → demeaning required            |
| Liquidity (primary)            | LIQUIDITY_FILTER=False (full universe)                               |
| Exp4 drop bottom 10%           | IC 0.0338; Top-50 edge 0.81%; hit 66.7% (robustness only)            |
| Exp5 drop bottom 20%           | IC 0.0299; Top-50 edge 0.47%; hit 50.0% (robustness only)            |
| Top-50 size/liquidity (val)    | Smaller & less liquid than universe (log mcap −0.62; turnover 0.50×) |
| Sector concentration (val)     | Max avg sector weight Real Estate 20.2% (diversified)                |
| Top-50 risk (val)              | Mean +1.46%/mo; vol 5.34%; Sharpe 0.27; max DD −7.9%                 |
| Factor corr of y_pred (val)    | MOM +0.21; SMB −0.19; HML −0.14; log mcap ~0                         |

## Table 3 — Transaction costs (val Top-50; turnover 62%)

| Scenario                     | Round-trip cost / month   | Net Top-50 edge / month   |
|:-----------------------------|:--------------------------|:--------------------------|
| Optimistic (15 bp one-way)   | 0.19%                     | 1.43%                     |
| Base (25 bp one-way)         | 0.31%                     | 1.31%                     |
| Base (30 bp one-way)         | 0.37%                     | 1.25%                     |
| Conservative (40 bp one-way) | 0.50%                     | 1.12%                     |
| Stress (50 bp one-way)       | 0.62%                     | 1.00%                     |

### Capacity note (illustrative)
Under a simple square-root impact model (£5m ADV assumption), practical AUM is roughly
**£50–100m** before the 10% ADV participation rule is breached. See `cost_and_capacity_analysis.py`.

### Files
- Main metrics CSV: `XGB_DISSERTATION_RESULTS_SUMMARY.csv`
- Robustness CSV: `XGB_DISSERTATION_ROBUSTNESS_SUMMARY.csv`
- Cost CSV: `XGB_DISSERTATION_COST_SUMMARY.csv`
- Tracker: `xgb_results_tracker.csv`
- Frozen spec: `xgb_frozen_spec.json`

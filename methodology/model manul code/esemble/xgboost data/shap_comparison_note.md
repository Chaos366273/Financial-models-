# SHAP comparison — baseline vs sentiment (walk-forward average)

Importance metric: **mean |SHAP|** averaged across all 14 rolling steps.
Eval sample capped at 4000 rows/step for TreeSHAP.

## Does sentiment rank high?
- `news_sentiment`: rank **16** (share 2.9% of total mean|SHAP|)
- `sentiment_proxy`: rank **20** (share 1.9% of total mean|SHAP|)
- `twitter_publication_count`: rank **22** (share 1.6% of total mean|SHAP|)
- `news_publication_count`: rank **23** (share 1.2% of total mean|SHAP|)
- `twitter_sentiment`: rank **24** (share 0.8% of total mean|SHAP|)

**Answer:** No — sentiment features do **not** enter the global top 15; traditional / technical signals dominate.

## Do traditional factors lose importance?
- `MOM`: baseline rank 11 → sentiment model rank 10 (share Δ -0.2 pp)
- `SMB`: baseline rank 7 → sentiment model rank 9 (share Δ -1.0 pp)
- `HML`: baseline rank 10 → sentiment model rank 14 (share Δ -0.9 pp)
- `RMW`: baseline rank 18 → sentiment model rank 19 (share Δ -0.5 pp)
- `CMA`: baseline rank 12 → sentiment model rank 12 (share Δ -0.4 pp)
- `log_market_cap`: baseline rank 5 → sentiment model rank 5 (share Δ +0.4 pp)

## Top 15 — baseline

1. `price_to_sma_20` — 0.001632
2. `volatility_30d` — 0.001433
3. `volume_trend` — 0.001067
4. `sma_20` — 0.000983
5. `log_market_cap` — 0.000777
6. `debt_equity` — 0.000642
7. `SMB` — 0.000629
8. `bb_lower` — 0.000621
9. `atr` — 0.000566
10. `HML` — 0.000532
11. `MOM` — 0.000515
12. `CMA` — 0.000500
13. `revenue_growth` — 0.000490
14. `sma_15` — 0.000486
15. `macd_signal` — 0.000465

## Top 15 — sentiment model

1. `price_to_sma_20` — 0.001604
2. `volatility_30d` — 0.001309
3. `volume_trend` — 0.001018
4. `sma_20` — 0.000963
5. `log_market_cap` — 0.000864
6. `bb_lower` — 0.000619
7. `debt_equity` — 0.000582
8. `atr` — 0.000544
9. `SMB` — 0.000528
10. `MOM` — 0.000507
11. `revenue_growth` — 0.000485
12. `CMA` — 0.000461
13. `macd_signal` — 0.000453
14. `HML` — 0.000428
15. `sma_15` — 0.000406

# SHAP comparison — baseline vs sentiment (walk-forward average)

Importance metric: **mean |SHAP|** averaged across all 14 rolling steps.
Eval sample capped at 4000 rows/step for TreeSHAP.

## Does sentiment rank high?
- `news_sentiment`: rank **22** (share 1.7% of total mean|SHAP|)
- `sentiment_proxy`: rank **38** (share 0.8% of total mean|SHAP|)
- `twitter_publication_count`: rank **39** (share 0.8% of total mean|SHAP|)
- `news_publication_count`: rank **46** (share 0.6% of total mean|SHAP|)
- `twitter_sentiment`: rank **50** (share 0.4% of total mean|SHAP|)

**Answer:** No — sentiment features do **not** enter the global top 15; traditional / technical signals dominate.

## Do traditional factors lose importance?
- `MOM`: baseline rank 39 → sentiment model rank 44 (share Δ -0.0 pp)
- `SMB`: baseline rank 40 → sentiment model rank 43 (share Δ +0.0 pp)
- `HML`: baseline rank 29 → sentiment model rank 31 (share Δ -0.1 pp)
- `RMW`: baseline rank 38 → sentiment model rank 40 (share Δ -0.0 pp)
- `CMA`: baseline rank 32 → sentiment model rank 29 (share Δ +0.1 pp)
- `log_market_cap`: baseline rank 34 → sentiment model rank 33 (share Δ +0.1 pp)

## Top 15 — baseline

1. `price_to_sma_20` — 0.001589
2. `gbp_usd` — 0.001202
3. `volatility_30d` — 0.001138
4. `volume_trend` — 0.000717
5. `bb_upper` — 0.000617
6. `roe` — 0.000545
7. `bb_lower` — 0.000505
8. `atr` — 0.000485
9. `rsi` — 0.000484
10. `bb_middle` — 0.000458
11. `debt_equity` — 0.000436
12. `px_bid` — 0.000436
13. `macd_signal` — 0.000431
14. `amihud_illiquidity` — 0.000387
15. `pe_ratio` — 0.000367

## Top 15 — sentiment model

1. `price_to_sma_20` — 0.001600
2. `volatility_30d` — 0.001026
3. `gbp_usd` — 0.000890
4. `volume_trend` — 0.000709
5. `bb_upper` — 0.000614
6. `roe` — 0.000537
7. `atr` — 0.000536
8. `rsi` — 0.000509
9. `bb_lower` — 0.000489
10. `px_bid` — 0.000440
11. `debt_equity` — 0.000412
12. `amihud_illiquidity` — 0.000411
13. `revenue_growth` — 0.000372
14. `pe_ratio` — 0.000356
15. `macd_signal` — 0.000350

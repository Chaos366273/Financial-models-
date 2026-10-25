# SHAP vs Pearson — XGB H1 baseline (78 features, 14 walk-forward steps)

SHAP source: `analysis/xgboost/walk_forward_baseline/shap/global_mean_abs_shap.csv`
Pearson source: `pearson_with_target_h1.csv` (119,541 stock-months)

Spearman rank correlation (|Pearson r| rank vs mean|SHAP| rank): **0.214**

## Side-by-side (sorted by SHAP rank)

| SHAP | Feature | mean\|SHAP\| | Pearson r | \|r\| rank | Rank gap |
|---:|---|---:|---:|---:|---:|
| 1 | price_to_sma_20 | 0.001589 | +0.0107 | 35 | +34 |
| 2 | gbp_usd | 0.001202 | -0.0291 | 18 | +16 |
| 3 | volatility_30d | 0.001138 | +0.0160 | 30 | +27 |
| 4 | volume_trend | 0.000717 | -0.0039 | 54 | +50 |
| 5 | bb_upper | 0.000617 | -0.0180 | 22 | +17 |
| 6 | roe | 0.000545 | +0.0033 | 62 | +56 |
| 7 | bb_lower | 0.000505 | -0.0178 | 27 | +20 |
| 8 | atr | 0.000485 | -0.0201 | 21 | +13 |
| 9 | rsi | 0.000484 | +0.0056 | 46 | +37 |
| 10 | bb_middle | 0.000458 | -0.0179 | 24 | +14 |
| 11 | debt_equity | 0.000436 | -0.0084 | 39 | +28 |
| 12 | px_bid | 0.000436 | -0.0162 | 29 | +17 |
| 13 | macd_signal | 0.000431 | +0.0018 | 70 | +57 |
| 14 | amihud_illiquidity | 0.000387 | +0.0039 | 55 | +41 |
| 15 | pe_ratio | 0.000367 | -0.0036 | 59 | +44 |
| 16 | revenue_growth | 0.000366 | -0.0037 | 56 | +40 |
| 17 | ema_12 | 0.000359 | -0.0178 | 26 | +9 |
| 18 | volume | 0.000352 | -0.0007 | 75 | +57 |
| 19 | cur_mkt_cap | 0.000303 | -0.0059 | 45 | +26 |
| 20 | pb_ratio | 0.000300 | -0.0052 | 49 | +29 |
| 21 | ftse100_level | 0.000295 | -0.0616 | 5 | -16 |
| 22 | px_ask | 0.000277 | -0.0163 | 28 | +6 |
| 23 | macd | 0.000272 | +0.0035 | 60 | +37 |
| 24 | MKT_RF | 0.000268 | +0.0512 | 8 | -16 |
| 25 | sma_15 | 0.000245 | -0.0180 | 23 | -2 |
| 26 | bid_ask_spread | 0.000243 | +0.0160 | 31 | +5 |
| 27 | uk_gdp_qoq | 0.000210 | -0.0018 | 71 | +44 |
| 28 | eqy_turnover | 0.000209 | -0.0071 | 42 | +14 |
| 29 | HML | 0.000208 | -0.0105 | 36 | +7 |
| 30 | market_cap | 0.000196 | -0.0052 | 48 | +18 |
| 31 | earnings_growth | 0.000186 | -0.0103 | 37 | +6 |
| 32 | CMA | 0.000180 | -0.0634 | 3 | -29 |
| 33 | gold_usd | 0.000178 | +0.0081 | 40 | +7 |
| 34 | log_market_cap | 0.000162 | -0.0108 | 34 | +0 |
| 35 | brent_crude | 0.000161 | -0.0641 | 2 | -33 |
| 36 | ftse250_level | 0.000149 | -0.0305 | 17 | -19 |
| 37 | us_3m_yield | 0.000140 | -0.0324 | 15 | -22 |
| 38 | RMW | 0.000137 | -0.0006 | 76 | +38 |
| 39 | MOM | 0.000120 | -0.0577 | 6 | -33 |
| 40 | SMB | 0.000116 | -0.0029 | 64 | +24 |
| 41 | silver_usd | 0.000106 | +0.0043 | 52 | +11 |
| 42 | month_fe_12 | 0.000104 | -0.0152 | 32 | -10 |
| 43 | uk_cpi_yoy | 0.000099 | -0.0492 | 9 | -34 |
| 44 | uk_unemployment_rate | 0.000089 | +0.0488 | 10 | -34 |
| 45 | sma_20 | 0.000078 | -0.0179 | 24 | -21 |
| 46 | RF_FF | 0.000046 | -0.0331 | 14 | -32 |
| 47 | uk_bank_rate | 0.000043 | -0.0557 | 7 | -40 |
| 48 | us_fed_funds_target | 0.000033 | -0.0396 | 13 | -35 |
| 49 | sector__Basic_Resources | 0.000029 | -0.0000 | 78 | +29 |
| 50 | sector__Real_Estate | 0.000027 | -0.0076 | 41 | -9 |
| 51 | month_fe_02 | 0.000018 | -0.0270 | 19 | -32 |
| 52 | month_fe_10 | 0.000012 | +0.0108 | 33 | -19 |
| 53 | month_fe_03 | 0.000011 | +0.0726 | 1 | -52 |
| 54 | sector__Retail | 0.000010 | -0.0071 | 43 | -11 |
| 55 | month_fe_05 | 0.000009 | -0.0623 | 4 | -51 |
| 56 | sector__Industrial_Goods_and_Services | 0.000008 | +0.0042 | 53 | -3 |
| 57 | sector__Telecommunications | 0.000007 | +0.0049 | 50 | -7 |
| 58 | month_fe_11 | 0.000006 | +0.0482 | 11 | -47 |
| 59 | sector__Energy | 0.000006 | -0.0095 | 38 | -21 |
| 60 | month_fe_08 | 0.000006 | -0.0463 | 12 | -48 |
| 61 | month_fe_07 | 0.000006 | +0.0037 | 57 | -4 |
| 62 | sector__Media | 0.000005 | -0.0033 | 61 | -1 |
| 63 | sector__Travel_and_Leisure | 0.000005 | -0.0014 | 73 | +10 |
| 64 | month_fe_04 | 0.000004 | -0.0023 | 66 | +2 |
| 65 | sector__Technology | 0.000004 | +0.0065 | 44 | -21 |
| 66 | sector__Financial_Services | 0.000004 | +0.0002 | 77 | +11 |
| 67 | month_fe_09 | 0.000004 | -0.0202 | 20 | -47 |
| 68 | month_fe_06 | 0.000003 | +0.0307 | 16 | -52 |
| 69 | sector__Food__Beverage_and_Tobacco | 0.000003 | +0.0018 | 69 | +0 |
| 70 | sector__Consumer_Products_and_Services | 0.000003 | +0.0030 | 63 | -7 |
| 71 | sector__Personal_Care__Drug_and_Grocery_Stores | 0.000003 | +0.0021 | 68 | -3 |
| 72 | sector__Health_Care | 0.000002 | +0.0016 | 72 | +0 |
| 73 | sector__Banks | 0.000001 | -0.0023 | 67 | -6 |
| 74 | sector__Construction_and_Materials | 0.000000 | +0.0025 | 65 | -9 |
| 75 | sector__Insurance | 0.000000 | +0.0044 | 51 | -24 |
| 76 | sector__Utilities | 0.000000 | +0.0037 | 58 | -18 |
| 77 | sector__Chemicals | 0.000000 | +0.0011 | 74 | -3 |
| 77 | sector__Automobiles_and_Parts | 0.000000 | -0.0054 | 47 | -30 |

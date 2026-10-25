# Table 4.2 — Pearson correlation: full H1 characteristic set (78) + target

Panel: **119,541** stock-months | **807** tickers | filters: `is_member`, Unknown sector dropped, AIS.LN 2014-04 dropped.

Features: **78** kitchen-sink predictors (`get_feature_columns('kitchen_sink')`). Method: pairwise-complete Pearson on raw values (before walk-forward train-median fill).

Files:
- `pearson_with_target_h1.csv` — each predictor vs `target_excess_return`, sorted by |r|
- `pearson_correlation_h1.csv` — full **79×79** matrix (78 + target)

## Correlation with next-month excess return (all 78 predictors)

| Rank | Variable | Pearson r |
|---:|---|---:|
| 1 | month_fe_03 | 0.0726 |
| 2 | brent_crude | -0.0641 |
| 3 | CMA | -0.0634 |
| 4 | month_fe_05 | -0.0623 |
| 5 | ftse100_level | -0.0616 |
| 6 | MOM | -0.0577 |
| 7 | uk_bank_rate | -0.0557 |
| 8 | MKT_RF | 0.0512 |
| 9 | uk_cpi_yoy | -0.0492 |
| 10 | uk_unemployment_rate | 0.0488 |
| 11 | month_fe_11 | 0.0482 |
| 12 | month_fe_08 | -0.0463 |
| 13 | us_fed_funds_target | -0.0396 |
| 14 | RF_FF | -0.0331 |
| 15 | us_3m_yield | -0.0324 |
| 16 | month_fe_06 | 0.0307 |
| 17 | ftse250_level | -0.0305 |
| 18 | gbp_usd | -0.0291 |
| 19 | month_fe_02 | -0.0270 |
| 20 | month_fe_09 | -0.0202 |
| 21 | atr | -0.0201 |
| 22 | bb_upper | -0.0180 |
| 23 | sma_15 | -0.0180 |
| 24 | bb_middle | -0.0179 |
| 25 | sma_20 | -0.0179 |
| 26 | ema_12 | -0.0178 |
| 27 | bb_lower | -0.0178 |
| 28 | px_ask | -0.0163 |
| 29 | px_bid | -0.0162 |
| 30 | volatility_30d | 0.0160 |
| 31 | bid_ask_spread | 0.0160 |
| 32 | month_fe_12 | -0.0152 |
| 33 | month_fe_10 | 0.0108 |
| 34 | log_market_cap | -0.0108 |
| 35 | price_to_sma_20 | 0.0107 |
| 36 | HML | -0.0105 |
| 37 | earnings_growth | -0.0103 |
| 38 | sector__Energy | -0.0095 |
| 39 | debt_equity | -0.0084 |
| 40 | gold_usd | 0.0081 |
| 41 | sector__Real_Estate | -0.0076 |
| 42 | eqy_turnover | -0.0071 |
| 43 | sector__Retail | -0.0071 |
| 44 | sector__Technology | 0.0065 |
| 45 | cur_mkt_cap | -0.0059 |
| 46 | rsi | 0.0056 |
| 47 | sector__Automobiles_and_Parts | -0.0054 |
| 48 | market_cap | -0.0052 |
| 49 | pb_ratio | -0.0052 |
| 50 | sector__Telecommunications | 0.0049 |
| 51 | sector__Insurance | 0.0044 |
| 52 | silver_usd | 0.0043 |
| 53 | sector__Industrial_Goods_and_Services | 0.0042 |
| 54 | volume_trend | -0.0039 |
| 55 | amihud_illiquidity | 0.0039 |
| 56 | revenue_growth | -0.0037 |
| 57 | month_fe_07 | 0.0037 |
| 58 | sector__Utilities | 0.0037 |
| 59 | pe_ratio | -0.0036 |
| 60 | macd | 0.0035 |
| 61 | sector__Media | -0.0033 |
| 62 | roe | 0.0033 |
| 63 | sector__Consumer_Products_and_Services | 0.0030 |
| 64 | SMB | -0.0029 |
| 65 | sector__Construction_and_Materials | 0.0025 |
| 66 | month_fe_04 | -0.0023 |
| 67 | sector__Banks | -0.0023 |
| 68 | sector__Personal_Care__Drug_and_Grocery_Stores | 0.0021 |
| 69 | sector__Food__Beverage_and_Tobacco | 0.0018 |
| 70 | macd_signal | 0.0018 |
| 71 | uk_gdp_qoq | -0.0018 |
| 72 | sector__Health_Care | 0.0016 |
| 73 | sector__Travel_and_Leisure | -0.0014 |
| 74 | sector__Chemicals | 0.0011 |
| 75 | volume | -0.0007 |
| 76 | RMW | -0.0006 |
| 77 | sector__Financial_Services | 0.0002 |
| 78 | sector__Basic_Resources | -0.0000 |

## Top 10 by absolute correlation

- **month_fe_03**: r = 0.0726
- **brent_crude**: r = -0.0641
- **CMA**: r = -0.0634
- **month_fe_05**: r = -0.0623
- **ftse100_level**: r = -0.0616
- **MOM**: r = -0.0577
- **uk_bank_rate**: r = -0.0557
- **MKT_RF**: r = 0.0512
- **uk_cpi_yoy**: r = -0.0492
- **uk_unemployment_rate**: r = 0.0488

## Note vs Table 4.2b (SHAP top-15 subset)

`pearson_shap_top15.csv` remains the compact SHAP-focused subset. This file is the **full** H1 set requested for the data section.


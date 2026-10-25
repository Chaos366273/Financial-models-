# Table 4.1b — Descriptive statistics: XGBoost / LSTM characteristic panel

**XGB and LSTM share this table.** Both models use the same stock-month characteristics.
LSTM does **not** get a second set of descriptives because the eval window is 9 months.
The 9-month lookback only stacks nine rows of *these same variables*. Mean/SD of `rsi` (etc.) do not change.

## Sample

- Stock-months: **119,541** | tickers: **807** | months: **284** (2002-04-01 → 2025-11-01)
- Filters: `is_member`; Unknown sector dropped; AIS.LN × 2014-04 dropped (`Ticker` + month).
- H1 predictors: **78** (full characteristic set). H2 adds 5 sentiment columns (**83**).
- Target: `target_excess_return` (raw; models winsorise 1/99 on **train** for fitting only).
- LSTM seq=9, `contiguous=False`: at most **113,216** sequences on this panel (each sequence = 9 stock-months). XGB uses each stock-month once.
- Stats are **pairwise complete per column** (not one complete-case drop across all 78). Missingness is Table 4.1c.

## Why LSTM 9-month eval does not change Table 4.1b

| | XGBoost | LSTM |
|---|---|---|
| Variables | 78 characteristics + y | **same 78 + y** |
| Frequency | stock-month | stock-month, stacked into length-9 windows |
| What a row in this table is | one characteristic over stock-months | **the same** |
| What differs | n_eval ≈ all member-months in the year | n_eval sequences ≈ year months minus lookback |

Do **not** publish 9 × 78 = 702 descriptive rows. That would double-count the same series.

### Target

| Variable | n | Mean | Median | SD | Min | Max |
|---|---|---|---|---|---|---|
| target_excess_return | 119,541 | 0.0062 | 0.0050 | 0.1120 | -0.9314 | 3.3992 |

### Technical

| Variable | n | Mean | Median | SD | Min | Max |
|---|---|---|---|---|---|---|
| rsi | 119,540 | 51.1267 | 51.1502 | 16.7986 | 11.3636 | 89.5042 |
| macd | 119,541 | 0.0080 | 0.0032 | 0.1547 | -0.6807 | 0.6387 |
| macd_signal | 119,541 | 0.0082 | 0.0034 | 0.1488 | -0.6556 | 0.6120 |
| atr | 119,541 | 0.1469 | 0.0622 | 0.2372 | 0.0024 | 1.4326 |
| bb_upper | 119,541 | 5.9549 | 2.5882 | 9.2017 | 0.1208 | 52.7478 |
| bb_middle | 119,541 | 5.6178 | 2.4335 | 8.6986 | 0.1100 | 49.8581 |
| bb_lower | 119,541 | 5.2784 | 2.2795 | 8.2000 | 0.0892 | 46.9699 |
| price_to_sma_20 | 119,541 | 1.0014 | 1.0029 | 0.0490 | 0.8275 | 1.1527 |
| volume_trend | 119,541 | 2,727,495.98 | 662,666.50 | 6,485,063.66 | 3,825.65 | 46,780,326.65 |
| sma_20 | 119,541 | 5.6178 | 2.4335 | 8.6986 | 0.1100 | 49.8581 |
| ema_12 | 119,541 | 5.6216 | 2.4330 | 8.7098 | 0.1094 | 49.9483 |
| sma_15 | 119,541 | 5.6185 | 2.4331 | 8.6992 | 0.1092 | 49.8210 |
| volatility_30d | 119,518 | 32.1254 | 26.5230 | 20.7121 | 6.5046 | 130.9903 |

### Liquidity / price levels

| Variable | n | Mean | Median | SD | Min | Max |
|---|---|---|---|---|---|---|
| amihud_illiquidity | 119,515 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| bid_ask_spread | 119,488 | 0.0052 | 0.0021 | 0.0089 | 0.0002 | 0.0589 |
| eqy_turnover | 119,518 | 15,028,543.08 | 2,270,014.00 | 44,776,813.53 | 0.0000 | 3,213,367,000.00 |
| volume | 119,541 | 66,189,596.77 | 15,855,047.00 | 226,968,257.11 | 0.0000 | 8,204,084,942.00 |
| px_bid | 119,488 | 6.2493 | 2.4330 | 22.6724 | 0.0020 | 2,460.22 |
| px_ask | 119,488 | 6.2678 | 2.4450 | 22.7622 | 0.0020 | 2,475.59 |
| cur_mkt_cap | 98,795 | 6,181.49 | 1,278.09 | 17,164.32 | 0.5477 | 222,492.49 |

### Fundamentals

| Variable | n | Mean | Median | SD | Min | Max |
|---|---|---|---|---|---|---|
| pe_ratio | 119,541 | 26.7451 | 16.4979 | 52.1708 | 1.7412 | 491.3383 |
| pb_ratio | 119,541 | 3.3181 | 2.0663 | 4.9684 | 0.2037 | 40.4071 |
| roe | 119,541 | 13.4697 | 12.8928 | 24.4440 | -91.3306 | 139.9483 |
| debt_equity | 119,541 | 94.0340 | 48.2492 | 177.6201 | 0.0000 | 1,352.58 |
| revenue_growth | 119,541 | 0.0996 | 0.0591 | 0.3090 | -0.6395 | 2.2167 |
| earnings_growth | 119,541 | 0.2621 | 0.1016 | 2.9912 | -15.3022 | 17.4895 |
| market_cap | 96,792 | 6,100.94 | 1,240.19 | 17,077.54 | 0.8285 | 207,964.51 |
| log_market_cap | 96,792 | 7.2868 | 7.1230 | 1.5570 | 3.4491 | 11.4952 |

### Factors (incl. MOM, RF_FF)

| Variable | n | Mean | Median | SD | Min | Max |
|---|---|---|---|---|---|---|
| MKT_RF | 119,541 | 0.0061 | 0.0085 | 0.0511 | -0.1507 | 0.1311 |
| SMB | 119,541 | 0.0012 | 0.0014 | 0.0181 | -0.0465 | 0.0468 |
| HML | 119,541 | 0.0018 | 0.0012 | 0.0242 | -0.0498 | 0.0753 |
| RMW | 119,541 | 0.0022 | 0.0032 | 0.0160 | -0.0488 | 0.0352 |
| CMA | 119,541 | 0.0007 | 0.0001 | 0.0150 | -0.0393 | 0.0521 |
| MOM | 119,541 | 0.0075 | 0.0093 | 0.0373 | -0.2609 | 0.1365 |
| RF_FF | 119,541 | 0.0013 | 0.0006 | 0.0015 | 0.0000 | 0.0048 |

### Macro / markets

| Variable | n | Mean | Median | SD | Min | Max |
|---|---|---|---|---|---|---|
| uk_bank_rate | 119,541 | 1.9907 | 0.5000 | 2.0264 | 0.1000 | 5.7500 |
| uk_cpi_yoy | 119,541 | 2.7604 | 2.4000 | 2.1538 | -0.1000 | 10.5000 |
| uk_unemployment_rate | 119,541 | 5.4500 | 5.1000 | 1.4254 | 3.7000 | 8.4000 |
| uk_gdp_qoq | 39,805 | 0.4013 | 0.5000 | 3.0206 | -19.9000 | 17.0000 |
| us_3m_yield | 119,541 | 1.5877 | 0.9068 | 1.8113 | 0.0000 | 5.0727 |
| us_fed_funds_target | 119,541 | 1.7679 | 1.0000 | 1.8238 | 0.2500 | 5.2500 |
| brent_crude | 119,541 | 73.1081 | 71.4100 | 25.3874 | 24.4500 | 125.8900 |
| gold_usd | 119,541 | 1,291.22 | 1,293.53 | 485.1005 | 314.4500 | 1,937.44 |
| silver_usd | 119,541 | 19.4261 | 17.8523 | 8.2437 | 4.4600 | 39.8925 |
| gbp_usd | 119,541 | 1.5061 | 1.5060 | 0.2273 | 1.1622 | 2.0473 |
| ftse100_level | 119,541 | 6,332.98 | 6,462.22 | 1,071.28 | 3,655.60 | 7,678.20 |
| ftse250_level | 119,541 | 14,795.73 | 16,273.72 | 5,333.89 | 4,037.60 | 23,106.61 |

### Sentiment (H2 only; not in XGB/LSTM H1 78-feature spec)

| Variable | n | Mean | Median | SD | Min | Max |
|---|---|---|---|---|---|---|
| news_sentiment | 62,053 | 0.0889 | 0.0300 | 0.2101 | -1.0000 | 0.9999 |
| news_publication_count | 55,398 | 17.9848 | 7.2753 | 79.1130 | 0.0000 | 9,230.48 |
| twitter_sentiment | 33,290 | 0.0386 | 0.0041 | 0.1550 | -0.9826 | 0.9991 |
| twitter_publication_count | 47,371 | 9.4354 | 2.0000 | 46.3878 | 0.0000 | 3,447.00 |
| sentiment_proxy | 62,053 | -0.0007 | -0.0817 | 0.9312 | -2.8780 | 2.8497 |

### Calendar and sector dummies (share = mean of 0/1)

| Block | Variable | n | Share |
|---|---|---|---|
| month_fe | month_fe_02 | 119,541 | 0.0813 |
| month_fe | month_fe_03 | 119,541 | 0.0818 |
| month_fe | month_fe_04 | 119,541 | 0.0837 |
| month_fe | month_fe_05 | 119,541 | 0.0835 |
| month_fe | month_fe_06 | 119,541 | 0.0845 |
| month_fe | month_fe_07 | 119,541 | 0.0842 |
| month_fe | month_fe_08 | 119,541 | 0.0839 |
| month_fe | month_fe_09 | 119,541 | 0.0850 |
| month_fe | month_fe_10 | 119,541 | 0.0846 |
| month_fe | month_fe_11 | 119,541 | 0.0843 |
| month_fe | month_fe_12 | 119,541 | 0.0817 |
| sector_dummy | sector__Automobiles_and_Parts | 119,541 | 0.0039 |
| sector_dummy | sector__Banks | 119,541 | 0.0262 |
| sector_dummy | sector__Basic_Resources | 119,541 | 0.0415 |
| sector_dummy | sector__Chemicals | 119,541 | 0.0131 |
| sector_dummy | sector__Construction_and_Materials | 119,541 | 0.0198 |
| sector_dummy | sector__Consumer_Products_and_Services | 119,541 | 0.0368 |
| sector_dummy | sector__Energy | 119,541 | 0.0435 |
| sector_dummy | sector__Financial_Services | 119,541 | 0.2314 |
| sector_dummy | sector__Food__Beverage_and_Tobacco | 119,541 | 0.0314 |
| sector_dummy | sector__Health_Care | 119,541 | 0.0296 |
| sector_dummy | sector__Industrial_Goods_and_Services | 119,541 | 0.1562 |
| sector_dummy | sector__Insurance | 119,541 | 0.0376 |
| sector_dummy | sector__Media | 119,541 | 0.0272 |
| sector_dummy | sector__Personal_Care__Drug_and_Grocery_Stores | 119,541 | 0.0238 |
| sector_dummy | sector__Real_Estate | 119,541 | 0.0702 |
| sector_dummy | sector__Retail | 119,541 | 0.0530 |
| sector_dummy | sector__Technology | 119,541 | 0.0447 |
| sector_dummy | sector__Telecommunications | 119,541 | 0.0236 |
| sector_dummy | sector__Travel_and_Leisure | 119,541 | 0.0620 |
| sector_dummy | sector__Utilities | 119,541 | 0.0244 |

Source CSVs in this folder: `descriptive_stats.csv`, `descriptive_missingness.csv`, `dummy_shares.csv`, `pearson_shap_top15.csv`, `sample_definition.csv`.

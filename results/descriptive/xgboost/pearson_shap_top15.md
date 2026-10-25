# Table 4.2b — Pearson correlation with next-month excess return

Not a 78×78 matrix. Variables = XGB global SHAP top 15 + target, on the shared stock-month panel.

| Variable | Pearson r with target_excess_return |
|---|---:|
| gbp_usd | -0.0291 |
| atr | -0.0201 |
| bb_upper | -0.0180 |
| bb_middle | -0.0179 |
| bb_lower | -0.0178 |
| px_bid | -0.0162 |
| volatility_30d | 0.0160 |
| price_to_sma_20 | 0.0107 |
| debt_equity | -0.0084 |
| rsi | 0.0056 |
| volume_trend | -0.0039 |
| amihud_illiquidity | 0.0039 |
| pe_ratio | -0.0036 |
| roe | 0.0033 |
| macd_signal | 0.0018 |

Full 16×16 matrix: `pearson_shap_top15.csv`.

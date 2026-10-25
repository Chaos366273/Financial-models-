# Table 4.1c — Missingness on the XGB/LSTM panel

Same 119k-row filtered panel. XGB/LSTM fill remaining holes with **train-median** at each walk-forward step; this table is **before** that fill.

| Block | Variable | n raw | n missing | % missing |
|---|---|---|---|---|
| sentiment_h2 | twitter_sentiment | 119,541 | 86,251 | 72.15 |
| macro | uk_gdp_qoq | 119,541 | 79,736 | 66.70 |
| sentiment_h2 | twitter_publication_count | 119,541 | 72,170 | 60.37 |
| sentiment_h2 | news_publication_count | 119,541 | 64,143 | 53.66 |
| sentiment_h2 | sentiment_proxy | 119,541 | 57,488 | 48.09 |
| sentiment_h2 | news_sentiment | 119,541 | 57,488 | 48.09 |
| fundamentals | market_cap | 119,541 | 22,749 | 19.03 |
| fundamentals | log_market_cap | 119,541 | 22,749 | 19.03 |
| liquidity_price | cur_mkt_cap | 119,541 | 20,746 | 17.35 |
| liquidity_price | px_bid | 119,541 | 53 | 0.04 |
| liquidity_price | bid_ask_spread | 119,541 | 53 | 0.04 |
| liquidity_price | px_ask | 119,541 | 53 | 0.04 |
| liquidity_price | amihud_illiquidity | 119,541 | 26 | 0.02 |
| liquidity_price | eqy_turnover | 119,541 | 23 | 0.02 |
| technical | volatility_30d | 119,541 | 23 | 0.02 |
| technical | rsi | 119,541 | 1 | 0.00 |
| target | target_excess_return | 119,541 | 0 | 0.00 |
| technical | atr | 119,541 | 0 | 0.00 |
| technical | macd_signal | 119,541 | 0 | 0.00 |
| technical | macd | 119,541 | 0 | 0.00 |
| technical | bb_upper | 119,541 | 0 | 0.00 |
| liquidity_price | volume | 119,541 | 0 | 0.00 |
| technical | bb_middle | 119,541 | 0 | 0.00 |
| technical | bb_lower | 119,541 | 0 | 0.00 |
| technical | price_to_sma_20 | 119,541 | 0 | 0.00 |
| technical | volume_trend | 119,541 | 0 | 0.00 |
| technical | sma_20 | 119,541 | 0 | 0.00 |
| technical | ema_12 | 119,541 | 0 | 0.00 |
| technical | sma_15 | 119,541 | 0 | 0.00 |
| factors | MKT_RF | 119,541 | 0 | 0.00 |
| factors | SMB | 119,541 | 0 | 0.00 |
| fundamentals | pe_ratio | 119,541 | 0 | 0.00 |
| fundamentals | pb_ratio | 119,541 | 0 | 0.00 |
| fundamentals | debt_equity | 119,541 | 0 | 0.00 |
| fundamentals | roe | 119,541 | 0 | 0.00 |
| fundamentals | earnings_growth | 119,541 | 0 | 0.00 |
| fundamentals | revenue_growth | 119,541 | 0 | 0.00 |
| macro | uk_bank_rate | 119,541 | 0 | 0.00 |
| factors | RF_FF | 119,541 | 0 | 0.00 |
| factors | MOM | 119,541 | 0 | 0.00 |
| factors | CMA | 119,541 | 0 | 0.00 |
| factors | RMW | 119,541 | 0 | 0.00 |
| factors | HML | 119,541 | 0 | 0.00 |
| macro | uk_unemployment_rate | 119,541 | 0 | 0.00 |
| macro | uk_cpi_yoy | 119,541 | 0 | 0.00 |
| macro | silver_usd | 119,541 | 0 | 0.00 |
| macro | gold_usd | 119,541 | 0 | 0.00 |
| macro | brent_crude | 119,541 | 0 | 0.00 |
| macro | us_fed_funds_target | 119,541 | 0 | 0.00 |
| macro | us_3m_yield | 119,541 | 0 | 0.00 |
| macro | ftse250_level | 119,541 | 0 | 0.00 |
| macro | gbp_usd | 119,541 | 0 | 0.00 |
| macro | ftse100_level | 119,541 | 0 | 0.00 |

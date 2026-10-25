Markov + GARCH + LSTM hybrid (quantile VaR)

Does not overwrite:
  Dessertation\analysis\tier b\lstm var
  Dessertation\analysis\tier b\ftse 250\lstm var

Results save here:
  C:\Users\Faiaz\Desktop\Courses\sem2\dessertation\Dessertation\analysis\tier b\markov garch lstm hybrid

Expected files after a run:
  lstm_var_run_YYYYMMDD_HHMMSS.txt
  lstm_var_report.md
  lstm_var_pooled_metrics.csv
  walk_forward_lstm95_summary.csv
  walk_forward_lstm99_summary.csv
  pooled_lstm95_daily.csv
  pooled_lstm99_daily.csv

Features added vs frozen LSTM VaR:
  - prob_bear / prob_bull from marvok\market_appendix\pooled_market_forecast.csv
    (Hamilton OOS roll, no updating on test months)
  - csd_ret, pct_negative, cs_skew from master_panel monthly_return
    (lagged 1 month; this machine has no daily stock parquet)

Run (FTSE 100):
cd "C:\Users\Faiaz\Desktop\Courses\sem2\dessertation\Dessertation\analysis\tier b\lstm var"
C:/Users/Faiaz/AppData/Local/Programs/Python/Python311/python.exe lstm_var_quantile.py --markov-csd

FTSE 250 (optional, same folder unless you pass --out-dir):
C:/Users/Faiaz/AppData/Local/Programs/Python/Python311/python.exe lstm_var_quantile.py --markov-csd --index ftse250 --out-dir "C:\Users\Faiaz\Desktop\Courses\sem2\dessertation\Dessertation\analysis\tier b\markov garch lstm hybrid\ftse250"

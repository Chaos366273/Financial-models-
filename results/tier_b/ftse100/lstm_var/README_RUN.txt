LSTM quantile VaR — run when ready. Does not overwrite GARCH/GEW/H1 LSTM files.

LOOKBACK = 30. OOS GEW filters: GARCH dist=t, EGARCH p=1,o=1,q=1 (leverage).
The script aborts on import if those filters do not match the in-sample specs.

cd "C:\Users\Faiaz\Desktop\Courses\sem2\dessertation\Dessertation\analysis\tier b\lstm var"
C:/Users/Faiaz/AppData/Local/Programs/Python/Python311/python.exe lstm_var_quantile.py

Outputs in this folder:
  walk_forward_lstm95_summary.csv
  walk_forward_lstm99_summary.csv
  pooled_lstm95_daily.csv
  pooled_lstm99_daily.csv
  lstm_var_pooled_metrics.csv
  lstm_var_report.md
  lstm_var_run_*.txt

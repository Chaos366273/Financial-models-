LSTM VaR + lag-1 daily macros (no Markov, no monthly CSD)

GARCH-t / EGARCH / EWMA + return lags + US VIX, |d US 10y|, Baa-10y spread, |d log Brent|, |d log GBP|.

Does not overwrite:
  Dessertation\analysis\tier b\lstm var
  Dessertation\analysis\tier b\markov garch lstm hybrid
  Dessertation\analysis\tier b\ftse 250

Results save here:
  C:\Users\Faiaz\Desktop\Courses\sem2\dessertation\Dessertation\analysis\tier b\lstm var us macro

Run:
cd "C:\Users\Faiaz\Desktop\Courses\sem2\dessertation\Dessertation\analysis\tier b\lstm var"
C:/Users/Faiaz/AppData/Local/Programs/Python/Python311/python.exe lstm_var_quantile.py --macro-daily

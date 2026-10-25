FTSE 250 — same pipeline as FTSE 100 (Gaussian GARCH → GEW → LSTM quantile VaR).
Does not overwrite Dessertation\analysis\tier b\garch only, gew, or lstm var.

Python:
  C:/Users/Faiaz/AppData/Local/Programs/Python/Python311/python.exe

Run in order. Each job is ~14 walk-forward windows (2012–2025).

1) Standard Normal GARCH(1,1)
cd "C:\Users\Faiaz\Desktop\Courses\sem2\dessertation\methodology\model manul code\tier b"
C:/Users/Faiaz/AppData/Local/Programs/Python/Python311/python.exe garch11.py --index ftse250 --out-dir "C:\Users\Faiaz\Desktop\Courses\sem2\dessertation\Dessertation\analysis\tier b\ftse 250\garch only"

2) GEW (GARCH-t vol + EGARCH + EWMA + average; Gaussian VaR z)
cd "C:\Users\Faiaz\Desktop\Courses\sem2\dessertation\Dessertation\analysis\tier b\gew"
C:/Users/Faiaz/AppData/Local/Programs/Python/Python311/python.exe garch_gew_standalone.py --index ftse250 --out-dir "C:\Users\Faiaz\Desktop\Courses\sem2\dessertation\Dessertation\analysis\tier b\ftse 250\gew"

3) LSTM quantile VaR hybrid
cd "C:\Users\Faiaz\Desktop\Courses\sem2\dessertation\Dessertation\analysis\tier b\lstm var"
C:/Users/Faiaz/AppData/Local/Programs/Python/Python311/python.exe lstm_var_quantile.py --index ftse250 --out-dir "C:\Users\Faiaz\Desktop\Courses\sem2\dessertation\Dessertation\analysis\tier b\ftse 250\lstm var"

No flags = FTSE 100, original folders (leave those frozen).
Index series: data\final data for AI\05_macro\macro_wide.csv → ftse250_level.

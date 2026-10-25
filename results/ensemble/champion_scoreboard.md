# FTSE 100 champion — GARCH x ML-BL

Views = expanding IC blend of frozen XGB (78, no sentiment) and LSTM (pruned-25, no sentiment).
Size = GARCH(1,1). RF omitted. LSTM VaR not in s.

| Cost world | n | Sharpe | Net ann. | Mean s |
|---|--:|-------:|---------:|-------:|
| Academic 10 bp | 167 | 1.1327 | 0.1874 | 1.1756 |
| Realistic + borrow | 167 | 0.2371 | 0.0499 | 1.1756 |

Comparators copied under `comparators/` (read-only sources). Hurdle: unscaled realistic 1/N Sharpe 0.50.
Analysis root (not written): `C:\Users\Faiaz\Desktop\Courses\sem2\dessertation\Dessertation\analysis`

# Summary

UK equity dissertation, Manchester Metropolitan University. FTSE 100 and FTSE 250, monthly, 2002–2025. Walk-forward test years are 2012–2025.

This repo has the model code, the aggregate result tables, and the graphs. The Bloomberg firm-month panel is not included.

## Models

| Tier | What it does | Models | Code |
|---|---|---|---|
| A | Monthly excess-return forecast | FF5, Markov-switching, XGBoost, LSTM, XGB + LSTM ensemble | `methodology/model manul code/` |
| B | Volatility and VaR | GARCH(1,1), GARCH-t, GEW, LSTM-VaR, Markov–GARCH–LSTM, XGB volatility stacks | `methodology/dissertation-models/` and `results/tier_b/` |
| C | Portfolios | 1/N, minimum variance, mean variance, Black–Litterman, top-k XGB / LSTM / ensemble, ML Black–Litterman, volatility overlay | `methodology/model manul code/tier c/` |

## Results

`results/` holds pooled and by-year tables: RMSE, IC, long-short, VaR hit rates, and portfolio scoreboards. These are summaries, not stock-level forecasts.

| Folder | Contents |
|---|---|
| `results/ff5` | FF5 walk-forward metrics and rolling coefficients |
| `results/markov` | Markov walk-forward and sensitivity tables |
| `results/xgboost` | XGBoost walk-forward, SHAP comparison, cost and robustness summaries |
| `results/lstm` | Full-78 and pruned LSTM runs, with and without sentiment |
| `results/ensemble` | Champion scoreboard and GARCH-scaled ML Black–Litterman |
| `results/tier_b` | Volatility models, FTSE 100 and FTSE 250 |
| `results/tier_c` | Portfolio scoreboards, academic and realistic costs |
| `results/descriptive` | Sample size, missingness, and correlations |

## Graphs

| Folder | Contents |
|---|---|
| `figures/thesis` | Chapter 3 and appendix figures |
| `figures/ff5`, `figures/markov`, `figures/lstm` | Model charts |
| `figures/comparison` | Accuracy, ranking, FTSE 100 vs 250, Gu-style comparisons |
| `figures/submission` | Figures used in the submitted draft |
| `figures/html` | Interactive cost, turnover, and volatility charts |
| `figures/descriptive_ff5`, `figures/descriptive_tierb` | Descriptive charts |
| `figures/h3b` | Volatility-overlay dumbbells |

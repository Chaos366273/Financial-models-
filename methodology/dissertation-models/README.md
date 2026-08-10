# Dissertation Models — Stage A

Python engine for all **13 dissertation models**. Outputs Lab-compatible JSON; the HTML Lab only displays results.

**Location:** `methodology/dissertation-models/`

## Setup

**First time only** (installs into Python 3.11 — not the default `python` 3.13 on this PC):

```bat
cd methodology\dissertation-models
install_deps.bat
```

Or manually:

```bat
C:\Users\Faiaz\AppData\Local\Programs\Python\Python311\python.exe -m pip install -r requirements.txt
```

## Run

```bat
python run_all.py --model all --export outputs\lab_results_bundle.json
python run_all.py --model ff5
run_model.bat all
run_model.bat xgboost
```

| Flag | Description |
|------|-------------|
| `--model` | `all`, `ff5`, `markov_switching`, `random_forest`, `xgboost`, `lstm`, `stacked_ensemble`, `h2`, `garch_var`, `garch_lstm`, or any h3a strategy id |
| `--export` | Path for `lab_results_bundle.json` (default: `outputs/lab_results_bundle.json`) |

## Bloomberg data (default)

Paths in `config.yaml` point to:

- `../../data/final data for AI/07_panel/master_panel.parquet` (**primary** monthly panel)
- `../../data/final data for AI/02_fundamentals/Fundamental_Data_Full.csv`
- `../../data/final data for AI/05_macro/risk_free_monthly.csv`
- `../../data/bloomberg/trading_data_features.csv` (optional daily feed for H3b GARCH only)

When `master_panel` exists, `build_panel.py` loads it directly (no daily rebuild). Feature sets:

| Set | Contents |
|-----|----------|
| `traditional` | tech + fundamentals + FF5 |
| `h1_clean` | improved H1 (no month FE / FX levels; + MOM, liquidity, size) |
| `with_alt_data` | traditional + news/twitter sentiment |
| `all_indicators` | full master-panel predictors |

Env overrides: `DISSLAB_MASTER_PANEL`, `DISSLAB_FEATURES_CSV`, `DISSLAB_FUNDAMENTALS_CSV`, `DISSLAB_RISK_FREE_CSV`, `DISSLAB_PANEL_CACHE`.

## Output schema

`outputs/lab_results_bundle.json` matches Dissertation Lab `storage.js`:

- `results.h1` — predictive metrics (6 models)
- `results.h2` — traditional vs alt-data (xgboost, lstm, stacked_ensemble)
- `results.h3a` — portfolio strategies (5)
- `results.h3b` — VaR models (2)
- `results.shap` — TreeExplainer feature importance
- `model_status_updates` — sets matching cards to `pilot_done`
- `data_context` — exact CSV paths plus train/validation/test date ranges, row counts, and ticker count
- `summary` — bucket counts and 13-model coverage
- `calculation` (per result, schemaVersion 2) — inputs, transformations, target, model formula/parameters, metric definitions, 10-row `oos.examples_preview`, and artifact CSV path
- Sidecar CSVs under `outputs/artifacts/<run_id>/<model_or_feature_key>/predictions.csv`

Every individual result also carries the same `data_context`, making its training source and unseen test period explicit.

**Pilot / Bloomberg caveats surfaced in calculation.caveats:** FF5 factors are cross-sectional proxies (not Kenneth French), Markov scores market-level `MKT_RF` (not per-stock), H3b CVaR uses a Stage A approximation, and H2 still uses `sentiment_proxy` until Bloomberg BSI is wired. Excess returns use `risk_free_monthly.csv` when present (scalar `risk_free_rate` only as month-level fallback).

## Train/test discipline (Stage A)

- FF5 / Markov / ML models: fit on pre-test data only; metrics are OOS on the test fold
- Fundamentals: `fundamentals_lag_months` (default 6) — FY(t) usable from mid-(t+1)
- Winsorise / sentiment z-score: bounds fit on train window, applied forward
- H3a: mu/covariance from train+val only; all strategies share the same ticker universe
- H3b: rolling one-step GARCH VaR over the full test window (not a static last-20 check)
- FF5 factor cross-section: equities only (ETFs stay in the portfolio universe)

## Wire to Lab

1. Run models (above)
2. Open **Dissertation Lab** → **Settings** → **Import results bundle**
3. Select `outputs/lab_results_bundle.json` (check **Replace existing results** after a full run)

When using `Start Live Server.bat`, the browser performs these steps and imports the bundle automatically. Individual CLI/browser runs upsert their model result into the existing bundle; a full `all` run rebuilds and validates all 13 trackable outputs before replacing it.

## Package layout

```
dissertation-models/
├── config.yaml
├── run_all.py
├── run_model.bat
├── models/         ff5, markov, RF, XGB, LSTM, XGB+LSTM→PPO stack, portfolios, GARCH
├── data/           build_panel.py, features.py, export_ppo_forecasts.py
├── evaluation/     metrics + export_results.py
└── outputs/        gitignored JSON
```

## Change log

### 2026-08-01 — Manual LSTM golden-rule parity with XGB

- Manual `lstm_return.py`: `h1_clean` / `B_drop_bottom9` features, members-only, sector-neutral + winsor target, val-only default, `--final-test` lock, IC / long-short / top-K metrics, optional `--tune` grid.
- Lab `lstm_model.py`: `_build_sequences` accepts `target_col` / `feature_cols` / `history` warm-start; `fit_predict_lstm` supports early stopping on validation.
- Cross-fold warm-start: val sequences prepend train history; test prepends train+val (targets stay fold-local) so `seq_len=12` works on a 12-month val window.
- `fit_predict_lstm(..., seed=42)` + `lstm_return.SEED=42` for reproducible Exp grids (torch/numpy + batch shuffle generator).

### 2026-07-30 — XGB H1 improvements (manual + Lab)

- Manual `xgb_return.py`: FTSE `is_member` filter, `h1_clean` features, month-demean + winsorise target, val-grid + early stopping, Spearman IC + long-short metrics, no StandardScaler.
- Lab `xgboost_model.py` / `predictive.py` / `config.yaml` aligned for members_only + ranking metrics.

### 2026-07-30 — Final-data master panel wired as default

- `config.yaml` / `build_panel.py` load `data/final data for AI/07_panel/master_panel.parquet` when present.
- Feature sets expanded: `with_alt_data` uses News/Twitter sentiment; `all_indicators` connects full IV stack.
- Manual runners (`xgb_return.py`, `random_forest.py`, `lstm_return.py`) retargeted to master panel.

### 2026-07-28 — Bloomberg FTSE panel as Stage A default

- `config.yaml` defaults to `data/bloomberg/` features, fundamentals, and monthly risk-free.
- `build_panel.py` merges `risk_free_monthly.csv` into excess returns and caches `panel_monthly.parquet`.
- Manual runners under `model manul code/` retargeted to the same Bloomberg paths (`TICKERS = None`).

### 2026-07-27 — Stacked ensemble: RF/Ridge → PPO meta (XGB + LSTM)

- `stacked_ensemble.py` now uses **XGBoost + LSTM → PPO** (Random Forest and Ridge removed from the stack).
- `models/ppo_meta.py` trains a blend policy on validation forecasts; `config.yaml` adds `ppo_meta` hyperparameters.
- `data/export_ppo_forecasts.py` exports `xgb_pred` / `lstm_pred` CSV for injection into **github private** PPO (`PPO_fixed.py` / TG5).
- Standalone **random_forest** H1 model unchanged (still runnable via `--model random_forest`).

### 2026-07-20 — Calculation breakdowns and prediction artifacts

- Extended result schema to v2 with per-model `calculation` manifests.
- Exported deterministic 10-row OOS examples plus full prediction/backtest CSVs under `outputs/artifacts/`.
- Captured FF5 coefficients, Markov regime forecasts, ML feature lists, H3a weights, and H3b VaR day-level series.
- H2 nests traditional vs with_alt_data calculation blocks without overwriting H1 artifacts.

### 2026-07-20 — Dataset and result-integrity fixes

- Restored `data/pilot/` as the explicit default training source.
- Added environment-only data overrides for browser uploads.
- Added required-column, file, and non-empty chronological-split validation.
- Added per-result dataset and train/validation/test attribution.
- Added full-run 13-model completeness validation.
- Individual runs now merge by model ID instead of erasing unrelated results.
- Individual H3a commands now emit only the requested portfolio strategy.

# UK Equity Return Prediction — Dissertation Code & Data

Manchester Metropolitan University dissertation project: FTSE 100/250 monthly excess-return forecasting (2002–2025), comparing traditional econometric models (FF5, Markov-switching) with machine-learning approaches (XGBoost, LSTM, stacked ensemble), plus volatility and portfolio extensions in the Lab pipeline.

---

## Repository layout

```
github dessertation/
├── README.md
├── requirements.txt
├── .gitignore
├── final data for AI/          # LOCAL ONLY — not pushed (Bloomberg licence)
└── methodology/
    ├── dissertation-models/    # Lab pipeline (Tier B/C + shared data utilities)
    └── model manul code/       # Tier-A manual runners (FF5, Markov, XGB, LSTM, ensemble)
```

> **Public repo = code only.** Place `final data for AI/` locally next to `methodology/` (see [Data setup](#data-setup-local-only)).

### Data (`final data for AI/`)

| Path | Contents |
|---|---|
| `07_panel/master_panel.parquet` | Main firm–month panel (953 tickers, Apr 2002–Dec 2025) |
| `02_fundamentals/` | Annual accounting CSVs |
| `01_factors/` | Ken French Europe FF5 + momentum (CSV) |
| `05_macro/` | Macro wide + risk-free monthly |
| `04_sentiment/` | Bloomberg news/Twitter sentiment (from 2007) |
| `06_universe_reference/` | FTSE membership + security reference |
| `reports/` | Data inventory & validation |

See `final data for AI/README.md` for field-level lookup.

### Code (`methodology/model manul code/`)

| Script | Model / purpose |
|---|---|
| `fama5.py` | FF5 OLS walk-forward (H1 baseline) |
| `marvok/markov_switching.py` | Regime-dependent FF5 walk-forward |
| `xgb_return.py` | XGBoost (tuning + golden-rule final test) |
| `xgb_walk_forward.py` | XGBoost walk-forward + H2 sentiment arm |
| `lstm_return.py` | LSTM walk-forward + H2 sentiment arm |
| `random_forest.py` | Random Forest (Lab split; supplementary) |
| `esemble/stacked_ensemble.py` | XGB + LSTM blend → ranked signals |

Walk-forward protocol (Tier A): **10-year train → 1-year test**, rolling **2012–2025** (14 steps).

---

## Data setup (local only)

The processed Bloomberg panel is **not** in the public GitHub repo. After cloning, copy or symlink your local folder:

```
final data for AI/
└── 07_panel/master_panel.parquet   # required
```

From your dissertation project:

```
source: .../dessertation/data/final data for AI/
target: .../github dessertation/final data for AI/
```

Ken French factors in `01_factors/` are public and may be included locally or downloaded from the [Ken French Data Library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html).

---

## Before first run — required setup

### 1. Python environment

```powershell
cd "C:\Users\Faiaz\Desktop\Courses\sem2\dessertation\github dessertation"
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

Python 3.11+ recommended. PyTorch install may need the [official wheel index](https://pytorch.org) on some machines.

### 2. Shared library

Manual scripts import `methodology/dissertation-models/` (included in this repo) for `build_panel`, `prepare_xy`, and Lab Tier B/C models.

### 3. Data path

Scripts expect the panel at:

```
final data for AI/07_panel/master_panel.parquet
```

(relative to repo root via `../../data/final data for AI` from `model manul code`, or env var `DISSLAB_MASTER_PANEL`).

---

## Quick start (Tier A)

From `methodology/model manul code/`:

```powershell
# FF5 walk-forward
python fama5.py

# Markov-switching walk-forward
python marvok/markov_switching.py

# XGBoost walk-forward (baseline + sentiment H2)
python xgb_walk_forward.py

# LSTM walk-forward
python lstm_return.py

# Stacked ensemble (uses saved XGB/LSTM forecasts)
python esemble/stacked_ensemble.py
```

Golden-rule final test (XGB / LSTM only — run **once** after tuning):

```powershell
python xgb_return.py --final-test
python lstm_return.py --final-test
```

---

## Tier B & C (Lab pipeline)

After copying `dissertation-models/`:

```powershell
cd methodology/dissertation-models
python run_all.py
```

Includes: GARCH(1,1)–VaR, GARCH–LSTM, 1/N, mean–variance, min-variance, Black–Litterman, ML–Black–Litterman.

Lab split: train ≤ 2022-12, validation 2023, test 2024+ (see `config.yaml`).

---

## Validation design (summary)

| Layer | Applies to |
|---|---|
| Walk-forward 2012–2025 | FF5, Markov, XGB, LSTM; ensemble uses saved WF forecasts |
| Single split tuning (train ≤2022, val 2023) | XGB, LSTM (before frozen HPs) |
| Golden-rule lock (2024+ scored once) | XGB, LSTM only |
| Sector-neutral target | XGB, LSTM only |

Tier B/C use separate Lab protocols (not walk-forward).

---

## Data licence notice

- **Ken French factors**: public research data — included.
- **Bloomberg-derived fields** in `final data for AI/`: accessed under Manchester Metropolitan University institutional licence. **Do not redistribute** beyond permitted academic use. For a public GitHub repo, consider publishing **code only** and sharing data privately with examiners.
- Do not commit API keys (Alpaca, Bloomberg, etc.). `.gitignore` blocks common secret patterns.

---

## Known gaps

- [ ] `final data for AI/` — local only (not on GitHub)
- [ ] `data/bloomberg-extract/` — optional raw pipeline not included
- [ ] S&P 500 robustness tickers — universe reference only

---

## Author

Faiaz — MSc Dissertation, Manchester Metropolitan University, 2025/26.

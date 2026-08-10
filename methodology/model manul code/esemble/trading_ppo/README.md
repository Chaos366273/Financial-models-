# Trading PPO (dissertation FTSE ensemble — Architecture B)

Self-contained copy under `model manul code/esemble/trading_ppo`.
**Do not import from `github private` / Pathrise at runtime.**

## Role

```
Stage 1 (esemble/)     saved XGB + LSTM FTSE forecasts → ranked_signals.csv
Stage 2 (this folder)  Trading PPO → BUY / HOLD / SELL timing on that book
```

XGB forecasts (default): `esemble/xgboost data/walk_forward_baseline/`  
LSTM forecasts (default): `esemble/lstm/walk_forward_baseline/`

Market data: `data/final data for AI` (FTSE). Prefer daily technical files; else month bars from `07_panel/master_panel.parquet`.

## Control panel

Edit [`../stacked_ensemble.py`](../stacked_ensemble.py) CONFIG:

- `STAGE1_MODE = "saved_forecasts"` — load walk-forward outputs (default)
- `TRADING_PPO_TIMESTEPS` — single Stage-2 train length (e.g. 600_000)
- `STAGE2_REWARD_OVERRIDES` — applied into local `ppo/utils/dynamic_config.py` at launch
- `AUTO_START_TRADING_PPO`

Full reward knobs live in local [`ppo/utils/dynamic_config.py`](ppo/utils/dynamic_config.py).

## Methodology docs (copied for write-up)

Under `ppo/docs/`:

- `REWARD_CHANGELOG.md`
- `REWARD_PLAN_REGIME_INDICATORS_2026-02-28.md`
- `SINGLE_SOURCES_OF_TRUTH.md`
- `SYSTEM_CONFIGURATION.md`

Optional SHAP debug: `ppo/tests/test_shap_diagnostic.py`

## Already local (do not re-copy over Arch B)

SHAP (`agent_explainability.py`), unified reports, regime controller, BC modules, scaling, `architecture_b.py` / `arch_b_config.py` / `signal_bridge.py`.

## Run

From `esemble/`:

```bat
python stacked_ensemble.py
```

Stage 1 writes `saved_data/ranked_signals.csv`. Stage 2 launches only if `AUTO_START_TRADING_PPO=True`.

## Status

- [x] Local PPO (no Pathrise runtime)
- [x] Saved FTSE XGB/LSTM Stage 1
- [x] Stage-2 timestep / reward overrides from stacked_ensemble
- [x] FTSE loader from `final data for AI`
- [ ] Full daily technical parquet when available (falls back to master_panel months)

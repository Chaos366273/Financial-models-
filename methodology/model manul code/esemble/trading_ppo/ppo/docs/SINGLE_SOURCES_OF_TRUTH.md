# Single Sources of Truth

**Purpose**: Ensure the agent never calls the wrong file; one canonical implementation per concern.

| Concern | Single source |
|--------|----------------|
| **Adaptive hyperparameters (LR + entropy)** | `AdaptiveHyperparameterCallback` in `PPO_fixed.py` + `ppo/utils/dynamic_config.py` (`get_dynamic_config()`) |
| **Evaluation** | `ppo/evaluation/performance_evaluator.py` (`evaluate_model`, `PerformanceEvaluator`) |
| **Training entry** | `PPO_fixed.py` → `ppo/training/ppo_trainer.py` (`create_ppo_agent`, `train_ppo_agent`) |
| **Env** | `ppo/models/trading_env.FixedTradingEnv` via `ppo/models/env_factory.make_trading_env` |

**Do not use**: `ppo/training/adaptive_lr_fix.py` (deprecated); `evaluation.performance_evaluator` (wrong import path; use `ppo.evaluation.performance_evaluator`).

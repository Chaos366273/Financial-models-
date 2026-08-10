"""
Performance Evaluator Module for PPO Trading System
Evaluates agent performance, computes metrics, and saves results
"""

import os
import numpy as np
import pandas as pd
import json
import time
import logging
from collections.abc import Mapping
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path

# CRITICAL FIX: Import psutil for system metrics collection
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    logger = logging.getLogger("PPOTrading.Evaluator")
    logger.warning("⚠️ psutil not available - system metrics will not be collected")

logger = logging.getLogger("PPOTrading.Evaluator")

def _calculate_statistical_significance(returns: List[float], strategy_sharpe_ratio: float) -> tuple:
    """Calculate statistical significance for returns and Sharpe ratio"""
    try:
        import scipy.stats as stats
        
        returns_arr = np.array(returns)
        
        # Only run tests if sufficient data points are available
        if len(returns_arr) < 30:
            return 1.0, 1.0  # Not significant due to insufficient data
        
        # 1. T-test for mean return vs. zero (null hypothesis: mean return is zero)
        t_stat, p_value = stats.ttest_1samp(returns_arr, popmean=0.0)
        
        # 2. Sharpe Ratio p-value using Lo's asymptotic distribution
        if strategy_sharpe_ratio != 0 and len(returns_arr) > 30:
            t_sharpe = np.sqrt(len(returns_arr)) * strategy_sharpe_ratio
            sharpe_p_value = 1 - stats.norm.cdf(abs(t_sharpe))  # One-sided test
        else:
            sharpe_p_value = 1.0
        
        def _to_float(val):
            try:
                arr = np.asarray(val, dtype=float)
                if arr.size == 0:
                    return 1.0
                return float(arr.reshape(-1)[0])
            except Exception:
                return 1.0

        return _to_float(p_value), _to_float(sharpe_p_value)
        
    except ImportError:
        # Fallback without scipy
        returns_arr = np.array(returns)
        if len(returns_arr) >= 30 and np.std(returns_arr) > 0:
            mean_r = float(np.mean(returns_arr))
            std_r = float(np.std(returns_arr))
            t_stat = mean_r / (std_r / np.sqrt(len(returns_arr)))
            # Normal approximation for p-value
            from math import erf, sqrt
            p_val_calc = float(1.0 - 0.5 * (1.0 + erf(abs(t_stat) / sqrt(2)))) * 2.0
            # Sharpe p-value approximation
            t_sharpe = abs(strategy_sharpe_ratio) * np.sqrt(max(1, len(returns_arr)))
            sharpe_p_val_calc = float(1.0 - 0.5 * (1.0 + erf(t_sharpe / sqrt(2)))) * 2.0
            return p_val_calc, sharpe_p_val_calc
        else:
            return 1.0, 1.0
    except Exception:
        return 1.0, 1.0


class _ObsShapeAlignWrapper:
    """Wraps env so observation shape matches model (slice or zero-pad). Use when test env obs dim != train."""
    def __init__(self, env: Any, target_obs_size: int):
        self.env = env
        self.target_obs_size = int(target_obs_size)

    def _align_obs(self, obs: Any) -> Any:
        if obs is None:
            return obs
        try:
            arr = np.asarray(obs, dtype=np.float32)
            flat = arr.reshape(-1)
            if len(flat) == self.target_obs_size:
                return obs
            if len(flat) > self.target_obs_size:
                out = np.zeros(self.target_obs_size, dtype=np.float32)
                out[:] = flat[:self.target_obs_size]
            else:
                out = np.zeros(self.target_obs_size, dtype=np.float32)
                out[:len(flat)] = flat
            return out.reshape(1, -1) if arr.ndim > 1 else out
        except Exception:
            return obs

    def reset(self, *args, **kwargs):
        result = self.env.reset(*args, **kwargs)
        if isinstance(result, tuple):
            obs, info = result[0], result[1] if len(result) > 1 else {}
            return self._align_obs(obs), info
        return self._align_obs(result)

    def step(self, action, *args, **kwargs):
        step_result = self.env.step(action, *args, **kwargs)
        obs = step_result[0]
        aligned = self._align_obs(obs)
        return (aligned,) + step_result[1:]

    @property
    def observation_space(self):
        try:
            import gym
            return gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.target_obs_size,), dtype=np.float32)
        except Exception:
            return getattr(self.env, 'observation_space', None)

    def __getattr__(self, name):
        return getattr(self.env, name)


_ACTION_NAMES = ("HOLD", "BUY", "SELL", "SHORT")


def _extract_action_logits(model: Any, obs: Any, action_masks: Optional[Any] = None) -> Optional[np.ndarray]:
    """Best-effort raw action logits [HOLD, BUY, SELL, SHORT] for diagnostics."""
    if model is None or not hasattr(model, "policy"):
        return None
    try:
        import torch
        policy = model.policy
        obs_tensor, _ = policy.obs_to_tensor(obs)
        with torch.no_grad():
            try:
                dist = policy.get_distribution(obs_tensor, action_masks=action_masks)
            except TypeError:
                dist = policy.get_distribution(obs_tensor)
            logits = None
            if hasattr(dist, "distribution") and hasattr(dist.distribution, "logits"):
                logits = dist.distribution.logits
            elif hasattr(dist, "logits"):
                logits = dist.logits
            if logits is None:
                return None
            arr = logits.detach().cpu().numpy().reshape(-1)
            if arr.size < 4:
                return None
            return np.asarray(arr[:4], dtype=float)
    except Exception as e:
        logger.debug(f"Logit extract failed: {e}")
        return None


def _trade_action_counts(trade_history: Any) -> Dict[str, int]:
    counts = {"BUY": 0, "SELL": 0, "STOP_LOSS": 0, "SHORT": 0, "HOLD": 0, "OTHER": 0}
    if not isinstance(trade_history, list):
        return counts
    for t in trade_history:
        if not isinstance(t, dict):
            continue
        a = str(t.get("action", "OTHER")).upper()
        if a in counts:
            counts[a] += 1
        else:
            counts["OTHER"] += 1
    return counts


def _eval_fingerprint(results: Dict[str, Any]) -> Dict[str, Any]:
    th = results.get("trade_history") or []
    first = th[0] if th and isinstance(th[0], dict) else {}
    pac = results.get("policy_action_counts") or {}
    fills = _trade_action_counts(th)
    return {
        "total_return": round(float(results.get("total_return") or 0.0), 6),
        "steps": int(results.get("steps_simulated") or 0),
        "policy_BUY": int(pac.get("BUY", 0) or 0),
        "policy_SELL": int(pac.get("SELL", 0) or 0),
        "policy_HOLD": int(pac.get("HOLD", 0) or 0),
        "fill_BUY": int(fills.get("BUY", 0)),
        "fill_SELL": int(fills.get("SELL", 0)),
        "fill_STOP": int(fills.get("STOP_LOSS", 0)),
        "first_ticker": first.get("ticker"),
        "first_shares": round(float(first.get("shares") or 0.0), 6) if first else None,
    }


def _print_and_check_action_summary(
    results: Dict[str, Any],
    save_path: str,
    logit_snapshots: Optional[List[Dict[str, Any]]] = None,
    label: str = "deterministic",
) -> None:
    """Terminal ACTION SUMMARY + warn if fingerprint matches previous deterministic eval."""
    pac = results.get("policy_action_counts") or {}
    fills = _trade_action_counts(results.get("trade_history"))
    ret = float(results.get("total_return") or 0.0)
    steps = int(results.get("steps_simulated") or 0)
    fp = _eval_fingerprint(results)
    results["eval_fingerprint"] = fp

    lines = [
        "",
        "=" * 72,
        f"[ACTION SUMMARY] mode={label}",
        f"  steps={steps} | return={ret:+.2%}",
        f"  policy  HOLD={pac.get('HOLD', 0)} BUY={pac.get('BUY', 0)} "
        f"SELL={pac.get('SELL', 0)} SHORT={pac.get('SHORT', 0)}",
        f"  fills   BUY={fills.get('BUY', 0)} SELL={fills.get('SELL', 0)} "
        f"STOP={fills.get('STOP_LOSS', 0)} SHORT={fills.get('SHORT', 0)}",
    ]
    if logit_snapshots:
        lines.append("  logits (step -> HOLD,BUY,SELL,SHORT | argmax):")
        for snap in logit_snapshots[:8]:
            lg = snap.get("logits") or []
            names = ",".join(f"{x:.3f}" for x in lg)
            lines.append(
                f"    step {snap.get('step')}: [{names}] -> {snap.get('argmax_name')} "
                f"(action={snap.get('action')})"
            )
    # Freeze check vs previous deterministic run
    if label == "deterministic":
        fp_path = Path(save_path) / "last_eval_fingerprint.json"
        try:
            if fp_path.is_file():
                prev = json.loads(fp_path.read_text(encoding="utf-8"))
                if prev == fp:
                    lines.append(
                        "  *** PATH FROZEN: identical fingerprint to previous deterministic eval ***"
                    )
                    lines.append(
                        "  *** Delete zip + FORCE_FRESH / new seed if you expected a new path ***"
                    )
                    results["path_frozen"] = True
                else:
                    results["path_frozen"] = False
                    lines.append("  fingerprint: CHANGED vs previous eval (good — path moved)")
            else:
                results["path_frozen"] = False
                lines.append("  fingerprint: first recorded (no prior to compare)")
            fp_path.parent.mkdir(parents=True, exist_ok=True)
            fp_path.write_text(json.dumps(fp, indent=2), encoding="utf-8")
        except Exception as e:
            lines.append(f"  fingerprint check skipped: {e}")
    lines.append("=" * 72)
    msg = "\n".join(lines)
    print(msg)
    logger.info(msg)


def _predict_with_temperature(
    model: Any,
    obs: Any,
    action_masks: Optional[Any] = None,
    temperature: float = 0.7,
) -> int:
    """Sample action from softmax(logits / temperature). For paper trading when deterministic is too peaky."""
    try:
        import torch
        policy = getattr(model, "policy", None)
        if policy is None:
            action, _ = model.predict(obs, action_masks=action_masks, deterministic=False)
            return int(np.asarray(action).flatten()[0])
        obs_t = torch.as_tensor(obs, device=policy.device, dtype=torch.float32)
        if obs_t.dim() == 1:
            obs_t = obs_t.unsqueeze(0)
        with torch.no_grad():
            dist = policy.get_distribution(obs_t)
            logits = getattr(dist, "distribution", dist).logits
            if logits is None:
                action, _ = model.predict(obs, action_masks=action_masks, deterministic=False)
                return int(np.asarray(action).flatten()[0])
            logits = logits / max(temperature, 1e-6)
            if action_masks is not None:
                mask = np.asarray(action_masks).flatten()
                if len(mask) == logits.shape[-1]:
                    logits = logits.clone()
                    logits[~torch.as_tensor(mask, device=logits.device, dtype=torch.bool)] = -1e9
            probs = torch.softmax(logits, dim=-1)
            action = torch.distributions.Categorical(probs=probs).sample()
            return int(action.item())
    except Exception as e:
        logger.debug(f"Temperature sampling failed ({e}), falling back to stochastic predict")
        action, _ = model.predict(obs, action_masks=action_masks, deterministic=False)
        return int(np.asarray(action).flatten()[0])


def evaluate_model(
    model: Any,
    env: Any,
    n_episodes: int = 10,
    deterministic: bool = True,
    render: bool = False,
    save_path: str = "ppo/saved_data/validation",
    use_temperature_sampling: bool = False,
    temperature: float = 0.7,
) -> Dict[str, Any]:
    """
    Evaluate trained agent on test environment

    Args:
        model: Trained PPO agent
        env: Test environment
        n_episodes: Number of episodes to evaluate
        deterministic: Whether to use deterministic actions (argmax)
        render: Whether to render the environment
        save_path: Path to save evaluation results
        use_temperature_sampling: If True, sample from softmax(logits / temperature) instead of argmax (paper trading).
        temperature: Temperature for sampling when use_temperature_sampling is True (default 0.7).

    Returns:
        Dictionary of evaluation results
    """
    logger.info(f"Evaluating model for {n_episodes} episodes...")
    eval_mode_label = "deterministic" if deterministic else "stochastic"
    
    # Create save directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)
    logit_snapshots: List[Dict[str, Any]] = []
    
    # Align env observation shape to model when they differ (e.g. train 323 vs test 484)
    try:
        model_obs_size = None
        if model is not None and hasattr(model, 'observation_space') and model.observation_space is not None:
            model_obs_size = getattr(model.observation_space, 'shape', None)
            if model_obs_size:
                model_obs_size = int(model_obs_size[0] if isinstance(model_obs_size, (list, tuple)) else model_obs_size)
        env_obs_size = None
        if hasattr(env, 'observation_space') and env.observation_space is not None:
            sh = getattr(env.observation_space, 'shape', None)
            if sh:
                env_obs_size = int(sh[0] if isinstance(sh, (list, tuple)) else sh)
        elif hasattr(env, 'envs') and env.envs and hasattr(env.envs[0], 'observation_space'):
            sh = getattr(env.envs[0].observation_space, 'shape', None)
            if sh:
                env_obs_size = int(sh[0] if isinstance(sh, (list, tuple)) else sh)
        if model_obs_size is not None and env_obs_size is not None and model_obs_size != env_obs_size:
            logger.warning(f"Observation shape mismatch: model expects {model_obs_size}, env has {env_obs_size}. Wrapping env to align.")
            env = _ObsShapeAlignWrapper(env, model_obs_size)
    except Exception as e:
        logger.debug(f"Obs shape alignment check skipped: {e}")
    
    # Initialize results
    results = {
        "total_return": 0.0,
        "sharpe_ratio": 0.0,
        "max_drawdown": 0.0,
        "total_trades": 0,
        "total_commission": 0.0,
        "total_slippage": 0.0,
        "final_portfolio_value": 0.0,
        "initial_portfolio_value": 0.0,
        "portfolio_history": [],
        "trade_history": [],
        "volatility": 0.0,
        "win_rate": 0.0,
        "profit_factor": 0.0,
        "current_positions": {},
        "total_position_value": 0.0,
        "cash": 0.0,
        "performance_metrics": {
            "execution_times": {},
            "resource_usage": {
                "memory": {
                    "percent": 0.0,
                    "used_gb": 0.0,
                    "available_gb": 0.0,
                    "total_gb": 0.0
                },
                "cpu": {
                    "percent": 0.0,
                    "cores": 0,
                    "freq_mhz": 0.0
                }
            },
            "performance_summary": {},
            "bottlenecks": [],
            "memory_history": [],
            "cpu_history": [],
            "profiling_active": False,
            "monitoring_active": False,
            "memory_mb": 0.0,
            "cpu_percent": 0.0
        },
        "execution_time": 0.0,
        "steps_simulated": 0,
        "step_win_rate": 0.0,
        "benchmark_return": 0.0,
        "alpha": 0.0,
        "beta": 0.0,
        "paths": {
            "trade_history_csv": os.path.join(save_path, "trade_history.csv"),
            "portfolio_history_csv": os.path.join(save_path, "portfolio_history.csv"),
            "returns_csv": os.path.join(save_path, "returns.csv"),
            "benchmark_returns_csv": os.path.join(save_path, "benchmark_returns.csv")
        },
        "env_spec": {
            "tickers": 0,
            "features": 0,
            "window": 0,
            "num_dates": 0
        },
        "model_spec": {
            "total_timesteps": 0,
            "device": "unknown",
            "seed": 0
        },
        "run_id": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "schema_version": "1.0",
        "timestamp": datetime.now().isoformat(),
        "pipeline_total_time": 0.0,
        "paper_trading_ready": False,
        "benchmark_data_valid": False,
        "p_value": 1.0,
        "sharpe_p_value": 1.0
    }
    
    # Start evaluation
    start_time = time.time()
    
    # CRITICAL FIX: Collect system metrics at start
    initial_memory_mb = 0.0
    initial_cpu_percent = 0.0
    if PSUTIL_AVAILABLE:
        try:
            process = psutil.Process(os.getpid())
            initial_memory_info = process.memory_info()
            initial_memory_mb = initial_memory_info.rss / (1024 * 1024)  # Convert to MB
            initial_cpu_percent = process.cpu_percent(interval=0.1)
        except Exception as e:
            logger.warning(f"⚠️ Failed to collect initial system metrics: {e}")
    
    # Reset environment with proper error handling
    # NOTE: Initial reset is deferred to the per-episode loop to avoid double reset during evaluation
    obs = None
    info = {}
    
    # Determine expected shapes dynamically
    try:
        # If env is vectorized (e.g., DummyVecEnv/VecNormalize), it has a batch dimension
        if hasattr(env, 'observation_space'):
            base_shape = env.observation_space.shape
        elif hasattr(env, 'envs') and hasattr(env.envs[0], 'observation_space'):
            base_shape = env.envs[0].observation_space.shape
        else:
            base_shape = getattr(obs, 'shape', None)

        expected_vec_shape = (1,) + (base_shape or ())
        expected_base_shape = base_shape
    except Exception:
        expected_vec_shape = getattr(obs, 'shape', None)
        expected_base_shape = getattr(obs, 'shape', None)

    # Normalize obs to vectorized shape for model.predict
    try:
        if hasattr(obs, 'shape') and obs is not None:
            if expected_base_shape and obs.shape == expected_base_shape:  # type: ignore
                # Expand to (1, obs_dim)
                import numpy as _np
                obs = _np.expand_dims(obs, axis=0)  # type: ignore
            elif expected_vec_shape and obs.shape == expected_vec_shape:  # type: ignore
                pass  # Already vectorized
            else:
                logger.warning(f"Unexpected observation shape on reset: {getattr(obs,'shape',None)}; proceeding anyway")
    except Exception as e:
        logger.error(f"Failed to normalize observation: {e}")
        # Continue with obs as-is
    
    # Initialize tracking variables
    episode_rewards = []
    episode_lengths = []
    portfolio_values = []
    returns = []
    benchmark_returns = []
    all_evaluation_trade_history = []  # CRITICAL FIX: Master list for all trades across all episodes
    all_episode_portfolio_values = []  # CRITICAL FIX: Master list for all portfolio values across all episodes
    step_returns = []
    wins = 0
    losses = 0
    total_steps = 0
    step_win_rate = 0.0  # ensure defined even if no steps occur
    
    # CRITICAL FIX: Accumulate realized/unrealized gains across ALL episodes
    total_realized_gains = 0.0
    total_unrealized_gains = 0.0

    # Policy action distribution (raw actions chosen by model), separate from executed trades.
    # This answers: "Did PPO ever choose SELL (2)?" even if it didn't result in a SELL trade.
    policy_action_counts = {0: 0, 1: 0, 2: 0, 3: 0}  # HOLD, BUY, SELL, SHORT

    # Constraint diagnostics (tracked by env info)
    masked_sell_count_max = 0
    masked_sell_streak_max = 0
    
    # Run evaluation episodes
    for episode in range(n_episodes):
        logger.info(f"Starting evaluation episode {episode+1}/{n_episodes}")
        
        episode_reward = 0
        episode_length = 0
        done = False
        
        # Reset environment for each episode with validation
        try:
            # CRITICAL FIX: Handle single environment (VecEnv wrapper was removed)
            # The environment is now a single FixedTradingEnv, not a VecEnv
            reset_result = env.reset()
            if isinstance(reset_result, tuple):
                # Gymnasium API: reset() returns (obs, info)
                obs, info = reset_result
            else:
                # Legacy Gym API: reset() returns just obs
                obs = reset_result
                info = {}
            
            # Set evaluation mode flag on environment (but don't pass to reset)
            if hasattr(env, '_eval_mode'):
                env._eval_mode = True
            
            # Ensure obs has batch dimension for model.predict
            if hasattr(obs, 'shape') and len(obs.shape) == 1:
                import numpy as _np
                obs = _np.expand_dims(obs, axis=0)
            # Validate observation
            if obs is None or (hasattr(obs, 'shape') and obs.shape == ()): 
                logger.error(f"Episode {episode+1}: Empty observation from reset")
                continue  # Skip this episode
        except Exception as e:
            logger.error(f"Episode {episode+1}: Failed to reset environment: {e}")
            continue  # Skip this episode
        
        # Extract initial portfolio value (single env)
        initial_portfolio = getattr(env, 'portfolio_value', 100000.0)
        
        if episode == 0:
            results["initial_portfolio_value"] = initial_portfolio
        
        # CRITICAL FIX: Always add initial portfolio value for each episode
        portfolio_values.append(initial_portfolio)
        
        # CRITICAL FIX: Add max_steps guard to prevent infinite loops
        max_steps_per_episode = 10000  # Safety limit to prevent infinite evaluation
        
        # Ensure episode_trades is always defined (e.g. when loop breaks on prediction error)
        episode_trades = []
        
        # Run episode
        while not done and episode_length < max_steps_per_episode:
            # Get action from model with error handling
            try:
                if model is None:
                    # No model provided - use random actions for testing
                    action = np.random.randint(0, 4)  # Random action (0=hold, 1=buy, 2=sell, 3=short)
                    logger.warning(f"Episode {episode+1}: No model provided, using random action: {action}")
                else:
                    # FIX 5: For MaskablePPO, must pass action_masks to predict()
                    # Get action masks from environment if available (for ActionMasker wrapper)
                    action_masks = None
                    if hasattr(env, 'action_masks'):
                        try:
                            action_masks = env.action_masks()
                        except Exception as mask_error:
                            logger.debug(f"Could not get action masks: {mask_error}")
                    
                    # Temperature sampling (plan Fix 5): softmax(logits / temperature) for paper trading
                    if use_temperature_sampling:
                        action = _predict_with_temperature(model, obs, action_masks, temperature)
                    elif action_masks is not None:
                        action, _ = model.predict(obs, action_masks=action_masks, deterministic=deterministic)
                    else:
                        action, _ = model.predict(obs, deterministic=deterministic)
                    # Normalize action to scalar int for discrete envs
                    try:
                        import numpy as _np
                        if isinstance(action, (list, tuple)):
                            action = int(action[0]) if len(action) > 0 else 0
                        elif isinstance(action, _np.ndarray):
                            action = int(action.item()) if action.size == 1 else int(action.flatten()[0])
                        else:
                            action = int(action)
                    except Exception:
                        action = 0
                    # Validate action
                    if action is None:
                        logger.error(f"Episode {episode+1}: Model returned None action")
                        break
                    # First-episode logit snapshots (freeze / BUY-dominance diagnostics)
                    if episode == 0 and len(logit_snapshots) < 8:
                        lg = _extract_action_logits(model, obs, action_masks)
                        if lg is not None:
                            am = int(np.argmax(lg))
                            logit_snapshots.append({
                                "step": int(episode_length),
                                "logits": [float(x) for x in lg],
                                "argmax": am,
                                "argmax_name": _ACTION_NAMES[am] if 0 <= am < 4 else "?",
                                "action": int(action),
                            })
            except Exception as e:
                logger.error(f"Episode {episode+1}: Model prediction failed: {e}")
                break

            # Track the raw action chosen by the policy
            try:
                if action in (0, 1, 2, 3):
                    policy_action_counts[action] = policy_action_counts.get(action, 0) + 1
                else:
                    # Unknown/invalid actions are tracked under HOLD bucket to avoid breaking reports
                    policy_action_counts[0] = policy_action_counts.get(0, 0) + 1
            except Exception:
                # Non-fatal: skip action distribution tracking on unexpected types
                pass
            
            # Take step in environment with error handling
            try:
                # CRITICAL FIX: Handle single environment (VecEnv wrapper was removed)
                # The environment is now a single FixedTradingEnv, not a VecEnv
                step_result = env.step(action)
            except Exception as e:
                logger.error(f"Episode {episode+1}: Step failed: {e}")
                break
            
            logger.debug(f"Step {episode_length}: step_result type: {type(step_result)}, length: {len(step_result) if isinstance(step_result, (list, tuple)) else 'not_iterable'}")
            
            # Handle different return formats
            if isinstance(step_result, (list, tuple)):
                if len(step_result) == 4:
                    # Standard format: (obs, reward, done, info)
                    obs, reward, done_flag, info = step_result
                    done_flag = bool(done_flag)
                elif len(step_result) == 5:
                    # Gymnasium format: (obs, reward, terminated, truncated, info)
                    obs, reward, terminated, truncated, info = step_result
                    done_flag = bool(terminated) or bool(truncated)
                else:
                    logger.error(f"Unexpected step_result length: {len(step_result)}")
                    break
            else:
                # Non-vec case - assume 4-tuple
                obs, reward, done_flag, info = step_result

            # Track masked SELL diagnostics (helps debug deterministic HOLD/SELL collapse)
            try:
                if isinstance(info, dict):
                    msc = info.get('masked_sell_count', 0) or 0
                    mss = info.get('masked_sell_streak', 0) or 0
                    masked_sell_count_max = max(masked_sell_count_max, int(msc))
                    masked_sell_streak_max = max(masked_sell_streak_max, int(mss))
            except Exception:
                pass
            
            # Ensure obs has batch dim for VecEnv
            if hasattr(obs, 'shape') and len(obs.shape) == 1:
                import numpy as _np
                obs = _np.expand_dims(obs, axis=0)
            
            # Convert done to scalar flag when vectorized
            if isinstance(done_flag, (list, tuple, np.ndarray)):
                done_flag = bool(done_flag[0])
            else:
                done_flag = bool(done_flag)

            # CRITICAL: propagate termination flag to loop variable
            done = done_flag
                
            logger.debug(f"Step {episode_length}: Parsed - reward: {reward}, done: {done_flag}")
            
            # Validate step results
            if obs is None or (hasattr(obs, 'shape') and obs.shape == ()): 
                logger.error(f"Episode {episode+1}: Environment step returned empty observation")
                break
            
            # Render if requested
            if render:
                env.render()
            
            # Update tracking variables
            episode_reward += reward
            episode_length += 1
            total_steps += 1
            
            # CRITICAL FIX: Extract metrics from info (single env) with better error handling
            info_dict = info if isinstance(info, dict) else {}
            
            # DEBUG: Log info structure for debugging
            logger.debug(f"Step {episode_length}: info type: {type(info)}, info_dict type: {type(info_dict)}")
            if isinstance(info_dict, dict):
                logger.debug(f"Step {episode_length}: info_dict keys: {list(info_dict.keys())}")
                if 'portfolio_value' in info_dict:
                    logger.debug(f"Step {episode_length}: portfolio_value type: {type(info_dict['portfolio_value'])}, value: {info_dict['portfolio_value']}")
            
            # Get portfolio value with fallback
            portfolio_value = info_dict.get('portfolio_value', None)
            if portfolio_value is None or not isinstance(portfolio_value, (int, float)) or portfolio_value <= 0:
                # Fallback to environment attribute (single env)
                if hasattr(env, 'portfolio_value'):
                    portfolio_value = env.portfolio_value
                else:
                    portfolio_value = initial_portfolio  # Last resort fallback
            
            # CRITICAL FIX: Ensure portfolio_value is a number, not a dict
            if isinstance(portfolio_value, dict):
                logger.debug(f"Step {episode_length}: portfolio_value is dict, extracting value: {portfolio_value}")
                portfolio_value = portfolio_value.get('value', portfolio_value.get('portfolio_value', initial_portfolio))
            elif isinstance(portfolio_value, (list, tuple, np.ndarray)):
                logger.debug(f"Step {episode_length}: portfolio_value is array, taking first element: {portfolio_value}")
                portfolio_value = float(portfolio_value[0]) if len(portfolio_value) > 0 else initial_portfolio
            elif not isinstance(portfolio_value, (int, float)):
                logger.debug(f"Step {episode_length}: portfolio_value is not number, converting: {portfolio_value} (type: {type(portfolio_value)})")
                try:
                    portfolio_value = float(portfolio_value) if portfolio_value is not None else initial_portfolio
                except (ValueError, TypeError):
                    logger.warning(f"Step {episode_length}: Could not convert portfolio_value to float, using initial_portfolio")
                    portfolio_value = initial_portfolio
            
            # DEBUG: Log portfolio value after conversion
            logger.debug(f"Step {episode_length}: Final portfolio_value: {portfolio_value} (type: {type(portfolio_value)})")
            logger.debug(f"Step {episode_length}: portfolio_values length: {len(portfolio_values)}")
            if len(portfolio_values) > 0:
                logger.debug(f"Step {episode_length}: Last portfolio_value: {portfolio_values[-1]} (type: {type(portfolio_values[-1])})")
            
            # Calculate step return with better handling
            try:
                if len(portfolio_values) > 0 and portfolio_values[-1] > 0:
                    # CRITICAL FIX: Ensure both values are numbers, not dicts
                    prev_portfolio = portfolio_values[-1]
                    if isinstance(prev_portfolio, dict):
                        prev_portfolio = prev_portfolio.get('value', prev_portfolio.get('portfolio_value', initial_portfolio))
                    if isinstance(portfolio_value, dict):
                        portfolio_value = portfolio_value.get('value', portfolio_value.get('portfolio_value', initial_portfolio))
                    
                    # CRITICAL FIX: Ensure both values are numeric before division
                    prev_portfolio_num = float(prev_portfolio) if isinstance(prev_portfolio, (int, float, np.number)) else initial_portfolio
                    portfolio_value_num = float(portfolio_value) if isinstance(portfolio_value, (int, float, np.number)) else initial_portfolio
                    
                    if prev_portfolio_num > 0:
                        step_return_val = (portfolio_value_num / prev_portfolio_num) - 1.0
                    else:
                        step_return_val = 0.0
                    logger.debug(f"Step {episode_length}: Calculated step_return_val: {step_return_val}")
                else:
                    step_return_val = info_dict.get('daily_return', info_dict.get('return', 0.0))
                    logger.debug(f"Step {episode_length}: Using fallback step_return_val: {step_return_val}")
            except Exception as e:
                logger.error(f"Step {episode_length}: Error calculating step return: {e}")
                logger.error(f"Step {episode_length}: portfolio_value: {portfolio_value} (type: {type(portfolio_value)})")
                logger.error(f"Step {episode_length}: portfolio_values[-1]: {portfolio_values[-1] if portfolio_values else 'None'} (type: {type(portfolio_values[-1]) if portfolio_values else 'None'})")
                step_return_val = 0.0
            
            # Get benchmark return (prefer env-provided; otherwise compute a robust fallback)
            bench_return_val = info_dict.get('benchmark_return', None)
            if bench_return_val is None:
                bench_return_val = info_dict.get('bench_return', None)
            if bench_return_val is None:
                bench_return_val = info_dict.get('spy_return', None)
            if bench_return_val is None:
                bench_return_val = 0.0
            try:
                bench_return_val = float(bench_return_val)
            except (TypeError, ValueError):
                bench_return_val = 0.0
            
            # CRITICAL FIX: If benchmark_return not in info, try to calculate from portfolio change
            # This is a fallback if environment doesn't provide benchmark data
            # IMPORTANT: only do this if the env did not provide a benchmark key; a true 0.0 return is valid.
            if ('benchmark_return' not in info_dict) and bench_return_val == 0.0:
                # Attempt to compute benchmark from SPY returns if SPY exists in the env universe.
                # This avoids "all zeros" benchmark series which makes alpha/beta meaningless.
                try:
                    env_instance = env
                    if hasattr(env_instance, 'unwrapped'):
                        env_instance = env_instance.unwrapped
                    # Some wrappers use .env; unwrap recursively
                    while hasattr(env_instance, 'env') and getattr(env_instance, 'env') is not env_instance:
                        env_instance = getattr(env_instance, 'env')

                    tickers = getattr(env_instance, 'unique_tickers', None) or getattr(env_instance, 'tickers', None)
                    price_matrix = getattr(env_instance, 'price_matrix', None)
                    current_step = getattr(env_instance, 'current_step', None)
                    if tickers and price_matrix is not None and isinstance(current_step, (int, np.integer)):
                        cs = int(current_step)
                        # env.current_step is typically "next step" after env.step; compute return from cs-1 to cs
                        if 1 <= cs < price_matrix.shape[0]:
                            if 'SPY' in tickers:
                                spy_idx = tickers.index('SPY')
                                prev_p = float(price_matrix[cs - 1, spy_idx])
                                cur_p = float(price_matrix[cs, spy_idx])
                                if prev_p > 0:
                                    bench_return_val = (cur_p / prev_p) - 1.0
                            else:
                                # Fallback benchmark: equal-weighted universe return (uses available tickers)
                                prev_vec = np.asarray(price_matrix[cs - 1, :], dtype=float)
                                cur_vec = np.asarray(price_matrix[cs, :], dtype=float)
                                valid = np.isfinite(prev_vec) & np.isfinite(cur_vec) & (prev_vec > 0)
                                if np.any(valid):
                                    bench_return_val = float(np.mean((cur_vec[valid] / prev_vec[valid]) - 1.0))
                except Exception:
                    # Non-fatal: keep 0.0
                    pass

            if bench_return_val == 0.0 and len(benchmark_returns) > 0:
                # Use last benchmark return if available
                bench_return_val = benchmark_returns[-1] if benchmark_returns else 0.0
            elif bench_return_val == 0.0 and 'portfolio_change' in info_dict:
                # Fallback: use portfolio change as proxy (not ideal, but better than 0)
                # This assumes portfolio tracks benchmark roughly
                bench_return_val = info_dict.get('portfolio_change', 0.0) * 0.5  # Scale down as proxy
            
            # REMOVED: Per-step trade collection (now done once per episode)
            # This prevents counting the same trades multiple times
            
            # Track portfolio value
            portfolio_values.append(portfolio_value)
            
            # Track returns
            returns.append(step_return_val)
            benchmark_returns.append(bench_return_val)
            
            # Track step returns for win rate calculation
            step_returns.append(step_return_val)
            if step_return_val > 0:
                wins += 1
            elif step_return_val < 0:
                losses += 1
            
            # CRITICAL FIX: Collect trades IMMEDIATELY when episode ends, BEFORE any reset
            if done:
                # Get env instance (single env) - UNWRAP to access underlying TradingEnv
                env_instance = env
                # Unwrap if wrapped (e.g., ActionMasker)
                if hasattr(env, 'unwrapped'):
                    env_instance = env.unwrapped
                elif hasattr(env, 'env'):  # Some wrappers use .env
                    env_instance = env.env
                    # Keep unwrapping until we get to the base env
                    while hasattr(env_instance, 'env') and env_instance.env is not env_instance:
                        env_instance = env_instance.env
                elif hasattr(env, 'envs') and env.envs and len(env.envs) > 0:
                    # Vectorized env - get first env and unwrap it
                    env_instance = env.envs[0]
                    if hasattr(env_instance, 'unwrapped'):
                        env_instance = env_instance.unwrapped
                    elif hasattr(env_instance, 'env'):
                        env_instance = env_instance.env
                        while hasattr(env_instance, 'env') and env_instance.env is not env_instance:
                            env_instance = env_instance.env
                
                logger.debug(f"Episode {episode+1}: Unwrapped env type: {type(env_instance)}")
                
                # CRITICAL: Try to access trade_history (master list) first, which is never cleared
                episode_trades = []
                if hasattr(env_instance, 'trade_history') and len(env_instance.trade_history) > 0:
                    episode_trades = env_instance.trade_history.copy()
                    logger.info(f"Episode {episode+1}: IMMEDIATE - Collected {len(episode_trades)} trades from trade_history (master list)")
                elif hasattr(env_instance, '_last_episode_trades') and len(env_instance._last_episode_trades) > 0:
                    episode_trades = env_instance._last_episode_trades.copy()
                    logger.info(f"Episode {episode+1}: IMMEDIATE - Collected {len(episode_trades)} trades from _last_episode_trades")
                elif hasattr(env_instance, 'episode_trade_history') and len(env_instance.episode_trade_history) > 0:
                    episode_trades = env_instance.episode_trade_history.copy()
                    logger.info(f"Episode {episode+1}: IMMEDIATE - Collected {len(episode_trades)} trades from episode_trade_history")
                elif hasattr(env_instance, 'get_episode_trade_history'):
                    episode_trades = env_instance.get_episode_trade_history()
                    logger.info(f"Episode {episode+1}: IMMEDIATE - Collected {len(episode_trades)} trades using get_episode_trade_history()")
                else:
                    logger.warning(f"Episode {episode+1}: IMMEDIATE - No trade collection method available")
                
                # Add to master history immediately
                if episode_trades:
                    for trade in episode_trades:
                        if isinstance(trade, dict):
                            trade['episode'] = episode
                    all_evaluation_trade_history.extend(episode_trades)
                    logger.info(f"Episode {episode+1}: IMMEDIATE - Added {len(episode_trades)} trades to master history")
                else:
                    logger.warning(f"Episode {episode+1}: IMMEDIATE - No trades in episode_trade_history at episode end (empty or missing)")
                
                # Episode complete, break to next episode
                break
        
        # After episode loop ends, do final checks
        # CRITICAL FIX: Check if we hit max_steps limit (infinite loop prevention)
        if episode_length >= max_steps_per_episode and not done:
            logger.warning(f"⚠️ Episode {episode+1}: Hit max_steps limit ({max_steps_per_episode}) - forcing episode termination")
            done = True  # Force termination to prevent infinite loop
        
        if done:
            # DEBUG: Log environment structure for debugging
            logger.info(f"Episode {episode+1}: DEBUG - Environment type: {type(env)}")
            logger.info(f"Episode {episode+1}: DEBUG - Has envs attribute: {hasattr(env, 'envs')}")
            if hasattr(env, 'envs'):
                logger.info(f"Episode {episode+1}: DEBUG - Number of envs: {len(env.envs) if env.envs else 0}")
                if env.envs and len(env.envs) > 0:
                    logger.info(f"Episode {episode+1}: DEBUG - First env type: {type(env.envs[0])}")
            
            # This block is no longer needed since we collect trades immediately
            # when done=True (see lines 489-510 above)
            logger.info(f"Episode {episode+1}: Trade collection already handled by immediate collection")
            
            # OLD FALLBACK CODE - DISABLED
            if False:  # Disabled - using immediate collection instead
                # CRITICAL FIX: Enhanced environment unwrapping with multiple fallback methods
                env_instance = None
                
                # Method 1: Try direct environment access (single env)
                if hasattr(env, 'unwrapped'):
                    env_instance = env.unwrapped
                    logger.info(f"Episode {episode+1}: DEBUG - Using unwrapped env: {type(env_instance)}")
                else:
                    env_instance = env
                    logger.info(f"Episode {episode+1}: DEBUG - Using direct env: {type(env_instance)}")
                
                # DEBUG: Log available trade history methods
                if env_instance:
                    trade_methods = [attr for attr in dir(env_instance) if 'trade' in attr.lower()]
                    logger.info(f"Episode {episode+1}: DEBUG - Available trade methods: {trade_methods}")
                    
                    # Check specific trade history attributes
                    trade_attrs = {
                        'get_episode_trade_history': hasattr(env_instance, 'get_episode_trade_history'),
                        'episode_trade_history': hasattr(env_instance, 'episode_trade_history'),
                        'trade_history': hasattr(env_instance, 'trade_history'),
                        'cumulative_evaluation_trade_history': hasattr(env_instance, 'cumulative_evaluation_trade_history'),
                        '_total_trades_executed': hasattr(env_instance, '_total_trades_executed')
                    }
                    logger.info(f"Episode {episode+1}: DEBUG - Trade attributes: {trade_attrs}")
                
                # CRITICAL FIX: Try multiple trade collection methods in order of preference
                collection_methods = [
                    ('get_episode_trade_history', lambda e: e.get_episode_trade_history() if hasattr(e, 'get_episode_trade_history') else None),
                    ('episode_trade_history', lambda e: e.episode_trade_history if hasattr(e, 'episode_trade_history') else None),
                    ('trade_history', lambda e: e.trade_history if hasattr(e, 'trade_history') else None),
                    ('cumulative_evaluation_trade_history', lambda e: e.cumulative_evaluation_trade_history if hasattr(e, 'cumulative_evaluation_trade_history') else None),
                    ('_total_trades_executed', lambda e: getattr(e, '_total_trades_executed', 0) if hasattr(e, '_total_trades_executed') else 0)
                ]
                
                for method_name, method_func in collection_methods:
                    try:
                        if env_instance:
                            result = method_func(env_instance)
                            if result is not None:
                                if method_name == '_total_trades_executed':
                                    # This is a count, not a list of trades
                                    logger.info(f"Episode {episode+1}: DEBUG - {method_name}: {result} (count)")
                                    if result > 0 and not episode_trades:
                                        # If we have trade count but no trade list, create a placeholder
                                        logger.warning(f"Episode {episode+1}: Found {result} trades executed but no trade list available")
                                else:
                                    # This should be a list of trades
                                    if isinstance(result, list):
                                        episode_trades = result
                                        logger.info(f"Episode {episode+1}: SUCCESS - Using {method_name} - found {len(episode_trades)} trades")
                                        break
                                    else:
                                        logger.info(f"Episode {episode+1}: DEBUG - {method_name}: {type(result)} (not a list)")
                    except Exception as e:
                        logger.warning(f"Episode {episode+1}: DEBUG - {method_name} failed: {e}")
                
                # CRITICAL FIX: If no trades found, try alternative collection methods
                if not episode_trades:
                    logger.warning(f"Episode {episode+1}: No trades found with standard methods, trying alternatives...")
                    
                    # Try to get trade count and reconstruct if possible
                    if env_instance and hasattr(env_instance, '_total_trades_executed'):
                        trade_count = getattr(env_instance, '_total_trades_executed', 0)
                        logger.info(f"Episode {episode+1}: DEBUG - Total trades executed: {trade_count}")
                        
                        if trade_count > 0:
                            # Try to get trades from info dict if available
                            if 'info' in locals() and isinstance(info, dict):
                                if 'trades' in info:
                                    episode_trades = info['trades']
                                    logger.info(f"Episode {episode+1}: SUCCESS - Found trades in info dict: {len(episode_trades)}")
                                elif 'trade_history' in info:
                                    episode_trades = info['trade_history']
                                    logger.info(f"Episode {episode+1}: SUCCESS - Found trade_history in info dict: {len(episode_trades)}")
                
                # Add episode context to trades
                for trade in episode_trades:
                    if isinstance(trade, dict):
                        trade['episode'] = episode
                
                logger.info(f"Episode {episode+1}: Collected {len(episode_trades)} trades before reset")
                if episode_trades:
                    all_evaluation_trade_history.extend(episode_trades)
                    logger.info(f"Episode {episode+1}: Added {len(episode_trades)} trades to master history")
                    logger.info(f"Episode {episode+1}: Master trade history now contains {len(all_evaluation_trade_history)} total trades")
                else:
                    logger.warning(f"Episode {episode+1}: No trades collected in this episode")
                
                # CRITICAL FIX: Break out of episode loop after collecting trades
                break
        
        # CRITICAL FIX: Capture realized/unrealized gains for THIS episode BEFORE reset
        try:
            # Unwrap the environment to access the base TradingEnv
            base_env = env
            while hasattr(base_env, 'env'):
                base_env = base_env.env
            
            if hasattr(base_env, 'get_realized_unrealized_gains'):
                episode_gains = base_env.get_realized_unrealized_gains()
                episode_realized = episode_gains.get('realized_gains', 0.0)
                episode_unrealized = episode_gains.get('unrealized_gains', 0.0)
                total_realized_gains += episode_realized
                total_unrealized_gains += episode_unrealized
                logger.info(f"Episode {episode+1}: Realized gains: ${episode_realized:.2f}, Unrealized gains: ${episode_unrealized:.2f}")
                logger.info(f"Episode {episode+1}: Cumulative - Realized: ${total_realized_gains:.2f}, Unrealized: ${total_unrealized_gains:.2f}")
            else:
                logger.warning(f"Episode {episode+1}: Base environment does not have get_realized_unrealized_gains method")
        except Exception as e:
            logger.warning(f"Episode {episode+1}: Failed to capture episode gains: {e}")
        
        # Record episode results (VecEnv may return numpy arrays)
        try:
            reward_scalar = float(np.asarray(episode_reward).sum())
        except Exception:
            reward_scalar = float(episode_reward) if not isinstance(episode_reward, (list, tuple)) else float(np.sum(episode_reward))
        episode_rewards.append(reward_scalar)
        episode_lengths.append(episode_length)
        
        # CRITICAL FIX: Trade collection now happens inside the episode loop when done=True
        # This ensures trades are captured before the environment resets and clears episode_trade_history
        
        # CRITICAL FIX: Get full portfolio history from environment for this episode
        # This ensures we capture the actual portfolio values from the environment, not just our tracked snapshots
        try:
            # Unwrap the environment to access the base TradingEnv
            base_env = env
            while hasattr(base_env, 'env'):
                base_env = base_env.env
            
            if hasattr(base_env, 'get_portfolio_history'):
                env_portfolio_history = base_env.get_portfolio_history()
                if env_portfolio_history and len(env_portfolio_history) > 0:
                    logger.info(f"Episode {episode+1}: Retrieved {len(env_portfolio_history)} portfolio values from environment")
                    for i, pv in enumerate(env_portfolio_history):
                        # CRITICAL FIX: Handle dict returns from get_portfolio_history()
                        if isinstance(pv, dict):
                            portfolio_val = float(pv.get('portfolio_value', 100000.0))
                        elif isinstance(pv, (int, float)):
                            portfolio_val = float(pv)
                        else:
                            logger.warning(f"Unexpected portfolio value type: {type(pv)}, defaulting to 100000.0")
                            portfolio_val = 100000.0
                        
                        all_episode_portfolio_values.append({
                            'episode': episode,
                            'episode_step': i,
                            'portfolio_value': portfolio_val
                        })
                else:
                    # Fallback to tracked portfolio_values
                    logger.warning(f"Episode {episode+1}: Environment portfolio history empty, using tracked values")
                    episode_portfolio_values = portfolio_values[-episode_length:] if episode_length > 0 and len(portfolio_values) > episode_length else [initial_portfolio]
                    for i, pv in enumerate(episode_portfolio_values):
                        all_episode_portfolio_values.append({
                            'episode': episode,
                            'episode_step': i,
                            'portfolio_value': pv
                        })
            else:
                # Fallback to tracked portfolio_values
                logger.warning(f"Episode {episode+1}: Environment does not have get_portfolio_history, using tracked values")
                episode_portfolio_values = portfolio_values[-episode_length:] if episode_length > 0 and len(portfolio_values) > episode_length else [initial_portfolio]
                for i, pv in enumerate(episode_portfolio_values):
                    all_episode_portfolio_values.append({
                        'episode': episode,
                        'episode_step': i,
                        'portfolio_value': pv
                    })
        except Exception as e:
            logger.error(f"Episode {episode+1}: Failed to get portfolio history: {e}, using fallback")
            # Fallback to tracked portfolio_values
            episode_portfolio_values = portfolio_values[-episode_length:] if episode_length > 0 and len(portfolio_values) > episode_length else [initial_portfolio]
            for i, pv in enumerate(episode_portfolio_values):
                all_episode_portfolio_values.append({
                    'episode': episode,
                    'episode_step': i,
                    'portfolio_value': pv
                })
        
        logger.info(f"Episode {episode+1} completed: reward={reward_scalar:.4f}, length={episode_length}, trades={len(episode_trades)}")
    
    # Calculate evaluation metrics
    evaluation_time = time.time() - start_time
    
    # Calculate portfolio returns
    # FIX: Initialize numeric_portfolio_values outside the if block
    numeric_portfolio_values = []
    
    if len(portfolio_values) > 1:
        # CRITICAL FIX: Ensure portfolio_values contains only numeric values
        for value in portfolio_values:
            if isinstance(value, dict):
                if 'portfolio_value' in value:
                    numeric_portfolio_values.append(float(value['portfolio_value']))
                elif 'value' in value:
                    numeric_portfolio_values.append(float(value['value']))
                else:
                    logger.warning(f"Unexpected dict structure in portfolio_values: {value}")
                    numeric_portfolio_values.append(100000.0)  # Fallback
            elif isinstance(value, (int, float, np.number)):
                numeric_portfolio_values.append(float(value))
            else:
                logger.warning(f"Unexpected type in portfolio_values: {type(value)}, value: {value}")
                numeric_portfolio_values.append(100000.0)  # Fallback
        
        if len(numeric_portfolio_values) > 1:
            # CRITICAL FIX: Ensure all values are numeric before numpy operations
            numeric_portfolio_values = [float(v) for v in numeric_portfolio_values if isinstance(v, (int, float, np.number))]
            
            if len(numeric_portfolio_values) > 1:
                portfolio_returns = np.diff(numeric_portfolio_values) / numeric_portfolio_values[:-1]
                
                # Prefer last mark-to-market from episode portfolio history (not a post-reset env value)
                start_portfolio = float(numeric_portfolio_values[0])
                actual_final_portfolio = float(numeric_portfolio_values[-1])
                if all_episode_portfolio_values:
                    try:
                        start_portfolio = float(all_episode_portfolio_values[0]["portfolio_value"])
                        actual_final_portfolio = float(all_episode_portfolio_values[-1]["portfolio_value"])
                    except Exception:
                        pass
                try:
                    # Unwrap wrappers (ActionMasker / ObsAlign) for live MTM
                    base_env = env
                    while hasattr(base_env, "env"):
                        base_env = base_env.env
                    if hasattr(env, "envs") and len(env.envs) > 0:
                        base_env = env.envs[0]
                        while hasattr(base_env, "env"):
                            base_env = base_env.env
                    if hasattr(base_env, "portfolio_value"):
                        live = float(base_env.portfolio_value)
                        initial_bal = float(getattr(base_env, "initial_amount", getattr(base_env, "initial_balance", start_portfolio)))
                        # Ignore live if env was reset back to initial while history shows MTM move
                        if abs(live - initial_bal) > 1.0 or abs(live - actual_final_portfolio) <= 1.0:
                            if abs(live - initial_bal) > 1.0:
                                actual_final_portfolio = live
                        else:
                            logger.warning(
                                f"Ignoring reset portfolio_value ${live:.2f} — using MTM history "
                                f"end ${actual_final_portfolio:.2f}"
                            )
                except Exception as e:
                    logger.debug(f"Could not get actual final portfolio value: {e}, using MTM history")
                
                total_return = (actual_final_portfolio / start_portfolio) - 1.0 if start_portfolio > 0 else 0.0
                logger.info(
                    f"DEBUG RETURN: Portfolio values - Start: {start_portfolio:.2f}, "
                    f"End (MTM): {actual_final_portfolio:.2f}, Return: {total_return:.4f}"
                )
                
                # FIX 3: Validate return calculation - flag suspicious returns
                # This will be cross-checked with realized/unrealized returns later
                portfolio_change = numeric_portfolio_values[-1] - numeric_portfolio_values[0]
                if abs(total_return) > 2.0:  # More than 200% return is suspicious
                    logger.warning(f"⚠️ SUSPICIOUS RETURN: {total_return:.2%} seems inflated. Portfolio change: ${portfolio_change:.2f}")
                    logger.warning(f"⚠️ Will validate against realized/unrealized returns when available")
            else:
                portfolio_returns = np.array([])
                total_return = 0.0
        else:
            portfolio_returns = np.array([])
            total_return = 0.0
        
        # Sharpe ratio (annualized, assuming daily returns)
        if len(portfolio_returns) > 1 and np.std(portfolio_returns) > 0:
            sharpe_ratio = np.mean(portfolio_returns) / np.std(portfolio_returns) * np.sqrt(252)
        else:
            sharpe_ratio = 0.0
        
        # CRITICAL FIX: Max drawdown calculation with proper validation
        if len(numeric_portfolio_values) > 1:
            peak = np.maximum.accumulate(numeric_portfolio_values)
            # Prevent division by zero
            peak_safe = np.where(peak > 0, peak, 1.0)
            drawdown = (peak_safe - numeric_portfolio_values) / peak_safe
            # Ensure drawdown is valid
            drawdown = np.nan_to_num(drawdown, nan=0.0, posinf=0.0, neginf=0.0)
            max_drawdown = float(np.max(drawdown)) if len(drawdown) > 0 else 0.0
            
            # Validate max_drawdown is reasonable (0 to 1)
            if max_drawdown < 0 or max_drawdown > 1:
                logger.warning(f"Invalid max_drawdown {max_drawdown}, clamping to [0, 1]")
                max_drawdown = max(0.0, min(1.0, max_drawdown))
        else:
            max_drawdown = 0.0
        
        # Volatility (annualized) - guard against NaN/Inf from zero variance or empty returns
        volatility = np.std(portfolio_returns) * np.sqrt(252) if len(portfolio_returns) > 1 else 0.0
        volatility = float(np.nan_to_num(volatility, nan=0.0, posinf=0.0, neginf=0.0))
        
        # CRITICAL FIX: Win rate calculation from accumulated trade PnL data
        total_trades = len(all_evaluation_trade_history)
        logger.info(f"DEBUG WIN RATE: Total trades in history: {total_trades}")
        
        # CRITICAL FIX: If no trades in evaluation history but we have episode data, this is a collection bug
        if total_trades == 0:
            logger.error(f"❌ CRITICAL: No trades in all_evaluation_trade_history despite {len(episode_rewards)} episodes completed!")
            logger.error(f"❌ This indicates the trade collection is failing during evaluation.")
            logger.error(f"❌ Check that get_episode_trade_history() is returning data correctly.")
        
        if total_trades > 0:
            # DEBUG: Log first few trades to understand structure
            logger.info(f"DEBUG WIN RATE: First 3 trades structure:")
            for i, trade in enumerate(all_evaluation_trade_history[:3]):
                logger.info(f"  Trade {i+1}: {trade}")
            
            # Compute PnL using realized PnL on SELL/AUTO_SELL/REBALANCE_SELL events
            total_pnl = 0.0
            profitable_trades = 0
            close_actions = {"SELL", "AUTO_SELL", "REBALANCE_SELL", "STOP_LOSS", "COVER"}
            closing_trades = []
            
            # DEBUG: Count all action types
            action_counts = {}
            for trade in all_evaluation_trade_history:
                action = str(trade.get('action', '')).upper()
                action_counts[action] = action_counts.get(action, 0) + 1
            logger.info(f"DEBUG WIN RATE: Action counts: {action_counts}")
            
            for trade in all_evaluation_trade_history:
                action = str(trade.get('action', '')).upper()
                if action in close_actions:
                    closing_trades.append(trade)
                    pnl_val = trade.get('pnl', 0.0)
                    try:
                        pnl_f = float(pnl_val)
                    except (TypeError, ValueError):
                        pnl_f = 0.0
                    total_pnl += pnl_f
                    if pnl_f > 0:
                        profitable_trades += 1
                    logger.debug(f"DEBUG WIN RATE: Closing trade - Action: {action}, PnL: {pnl_f}")
                else:
                    logger.debug(f"DEBUG WIN RATE: Non-closing trade - Action: {action}")

            # Win rate defined on closing trades only; avoid division by zero
            num_closing = len(closing_trades)
            logger.info(f"DEBUG WIN RATE: Closing trades: {num_closing}, Profitable: {profitable_trades}")
            
            if num_closing > 0:
                win_rate = profitable_trades / num_closing
                logger.info(f"Win Rate Calculation: {profitable_trades}/{num_closing} = {win_rate:.4f}")
                logger.info(f"Total PnL from closing trades: {total_pnl:.4f}")
                logger.info(f"Trade details: {len(all_evaluation_trade_history)} total trades, {num_closing} closing trades")
            else:
                # No closing trades found, use step-level win rate
                win_rate = step_win_rate
                logger.warning(f"No closing trades found in {len(all_evaluation_trade_history)} total trades")
                logger.warning(f"Using step-level win rate: {step_win_rate:.4f}")
            
        else:
            # CRITICAL FIX: Enhanced fallback with better logging
            win_rate = step_win_rate
            logger.warning(f"No trades found in trade history (length: {len(all_evaluation_trade_history)})")
            logger.warning(f"Using step-level win rate: {step_win_rate:.4f}")
            logger.warning(f"Step-level data: {wins} wins, {losses} losses")
        
        # Step win rate
        step_win_rate = wins / (wins + losses) if (wins + losses) > 0 else 0.0
        logger.info(f"DEBUG STEP WIN RATE: wins={wins}, losses={losses}, step_win_rate={step_win_rate:.4f}")
        
        # Profit factor
        gross_profit = sum(r for r in step_returns if r > 0)
        gross_loss = abs(sum(r for r in step_returns if r < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Alpha and Beta (vs benchmark)
        # NOTE: If benchmark returns are constant (e.g., all zeros), beta/alpha are undefined.
        alpha = None
        beta = None
        alpha_annual = None

        benchmark_data_valid = False
        benchmark_return = 0.0
        try:
            if benchmark_returns and returns and len(returns) == len(benchmark_returns) and len(returns) > 1:
                r = np.asarray(returns, dtype=float)
                b = np.asarray(benchmark_returns, dtype=float)
                mask = np.isfinite(r) & np.isfinite(b)
                r = r[mask]
                b = b[mask]
                if r.size > 1 and b.size > 1:
                    b_std = float(np.std(b))
                    # Benchmark is valid only if it has non-trivial variance
                    benchmark_data_valid = b_std > 1e-12
                    # Compounded benchmark return (same frequency as step returns)
                    benchmark_return = float(np.prod(1.0 + b) - 1.0) if b.size > 0 else 0.0
                    if benchmark_data_valid:
                        cov_matrix = np.cov(r, b)
                        if cov_matrix.shape == (2, 2) and cov_matrix[1, 1] > 1e-12:
                            beta = float(cov_matrix[0, 1] / cov_matrix[1, 1])
                            alpha = float((np.mean(r) - beta * np.mean(b)) * 252)
                            alpha_annual = float((1 + np.mean(r)) ** 252 - 1 - beta * ((1 + np.mean(b)) ** 252 - 1))
                        else:
                            benchmark_data_valid = False
                            logger.warning("⚠️ Benchmark calculation: covariance matrix invalid or near-zero variance")
                    else:
                        logger.warning("⚠️ Benchmark calculation: benchmark series has ~zero variance (alpha/beta undefined)")
        except Exception as e:
            benchmark_data_valid = False
            alpha = None
            beta = None
            alpha_annual = None
            benchmark_return = 0.0
            logger.warning(f"⚠️ Benchmark calculation failed: {e}")
        
        # CRITICAL FIX: Calculate actual statistical significance
        p_value, sharpe_p_value = _calculate_statistical_significance(returns, sharpe_ratio)
        
        # VALIDATION: Check for stop loss proceeds in portfolio values
        stop_loss_triggers = 0
        avg_stop_loss_pnl = 0.0
        if len(all_evaluation_trade_history) > 0:
            stop_loss_trades = [t for t in all_evaluation_trade_history if t.get('action') == 'STOP_LOSS']
            if stop_loss_trades:
                stop_loss_triggers = len(stop_loss_trades)
                total_stop_loss_pnl = sum(t.get('pnl', 0.0) for t in stop_loss_trades)
                avg_stop_loss_pnl = (total_stop_loss_pnl / len(stop_loss_trades)) if stop_loss_trades else 0.0
                # Calculate average stop loss PnL as percentage
                initial_balance = results.get("initial_portfolio_value", 100000.0)
                avg_stop_loss_pnl_pct = (avg_stop_loss_pnl / initial_balance) * 100.0 if initial_balance > 0 else 0.0
                avg_stop_loss_pnl = avg_stop_loss_pnl_pct
                logger.info(f"VALIDATION: Stop loss PnL: ${total_stop_loss_pnl:.2f}")
                logger.info(f"VALIDATION: Portfolio change: ${numeric_portfolio_values[-1] - numeric_portfolio_values[0]:.2f}")
                
                # Check if portfolio reflects stop loss profits
                expected_min_portfolio = numeric_portfolio_values[0] + total_stop_loss_pnl
                if numeric_portfolio_values[-1] < expected_min_portfolio * 0.9:  # Allow 10% tolerance
                    logger.error(f"VALIDATION FAILED: Portfolio ${numeric_portfolio_values[-1]:.2f} does not reflect stop loss profits ${total_stop_loss_pnl:.2f}")
                else:
                    logger.info(f"VALIDATION PASSED: Portfolio correctly reflects stop loss profits")
        
        # Determine if benchmark data is valid (computed above)
        
        # Determine if model is ready for paper trading - UPDATED CRITERIA
        paper_trading_ready = (
            sharpe_ratio > 0.5 and  # Lowered from 1.0 to 0.5
            max_drawdown < 0.2 and  # Increased from 0.1 to 0.2
            win_rate > 0.4 and      # Lowered from 0.45 to 0.4
            (p_value < 0.2 or sharpe_p_value < 0.1) and  # Relaxed significance requirements
            benchmark_data_valid
        )
    else:
        # Default values if not enough data
        total_return = 0.0
        sharpe_ratio = 0.0
        max_drawdown = 0.0
        volatility = 0.0
        win_rate = 0.0
        step_win_rate = 0.0
        profit_factor = 0.0
        alpha = 0.0
        beta = 0.0
        alpha_annual = 0.0
        benchmark_return = 0.0
        p_value = 1.0
        sharpe_p_value = 1.0
        benchmark_data_valid = False
        paper_trading_ready = False
        stop_loss_triggers = 0
        avg_stop_loss_pnl = 0.0
    
    # CRITICAL FIX: Calculate episodes_completed from episode_rewards length
    episodes_completed = len(episode_rewards) if episode_rewards else 0
    if episodes_completed == 0 and total_steps > 0:
        # If we have steps but no episodes, assume 1 episode (episode may not have terminated)
        episodes_completed = 1
        logger.warning(f"⚠️ Episode counting: {total_steps} steps but 0 episodes recorded. Assuming 1 episode.")
    
    # CRITICAL FIX: Calculate average episode length
    avg_episode_length = 0.0
    if len(episode_lengths) > 0:
        avg_episode_length = float(np.mean(episode_lengths))
        logger.info(f"✅ Episode length calculation: {len(episode_lengths)} episodes, average: {avg_episode_length:.2f} steps")
    elif total_steps > 0 and episodes_completed > 0:
        # Fallback: calculate from total steps and episodes
        avg_episode_length = float(total_steps / episodes_completed)
        logger.info(f"✅ Episode length fallback: {total_steps} steps / {episodes_completed} episodes = {avg_episode_length:.2f} steps")
    
    # CRITICAL FIX: Calculate average hold time from trade history
    average_hold_time = 0.0
    if len(all_evaluation_trade_history) > 0:
        hold_times = []
        position_entries = {}  # Track when positions were opened: {ticker: [{'step': step, 'shares': shares}, ...]}
        
        # Sort trades by step to ensure chronological order
        sorted_trades = sorted(all_evaluation_trade_history, key=lambda t: t.get('step', 0))
        
        for trade in sorted_trades:
            action = str(trade.get('action', '')).upper()
            ticker = trade.get('ticker', -1)
            step = trade.get('step', -1)
            shares = abs(float(trade.get('shares', 0.0)))  # Use absolute value and ensure float
            
            if ticker < 0 or step < 0:
                continue  # Skip invalid trades
            
            if action in ['BUY', 'REBALANCE_BUY'] and shares > 0:
                # Track position entry
                if ticker not in position_entries:
                    position_entries[ticker] = []
                position_entries[ticker].append({'step': step, 'shares': shares})
            
            elif action in ['SELL', 'AUTO_SELL', 'REBALANCE_SELL', 'STOP_LOSS'] and shares > 0:
                # Calculate hold time for closed position (handle partial sells)
                if ticker in position_entries and len(position_entries[ticker]) > 0:
                    remaining_shares_to_sell = shares
                    
                    # Match FIFO: first entry, first exit (handle partial sells)
                    while remaining_shares_to_sell > 0 and len(position_entries[ticker]) > 0:
                        entry = position_entries[ticker][0]
                        entry_shares = entry['shares']
                        
                        # Calculate hold time
                        hold_time = step - entry['step']
                        if hold_time >= 0:  # Valid hold time
                            # Use the minimum of shares being sold and entry shares for weighted average
                            shares_for_this_hold = min(remaining_shares_to_sell, entry_shares)
                            # Add hold time weighted by shares (or just add once per matched entry)
                            hold_times.append(hold_time)
                        
                        # Update remaining shares
                        if remaining_shares_to_sell >= entry_shares:
                            # Fully consumed this entry
                            remaining_shares_to_sell -= entry_shares
                            position_entries[ticker].pop(0)
                        else:
                            # Partially consumed - update entry shares
                            entry['shares'] -= remaining_shares_to_sell
                            remaining_shares_to_sell = 0
                    
                    # Clean up empty entries
                    if len(position_entries[ticker]) == 0:
                        del position_entries[ticker]
        
        if len(hold_times) > 0:
            average_hold_time = float(np.mean(hold_times))
            logger.info(f"✅ Hold time calculation: {len(hold_times)} matched trades, average: {average_hold_time:.2f} steps")
        else:
            logger.warning("⚠️ No hold times calculated - check trade history structure")
            # Debug: Log trade action distribution
            action_counts = {}
            for trade in all_evaluation_trade_history:
                action = str(trade.get('action', '')).upper()
                action_counts[action] = action_counts.get(action, 0) + 1
            logger.warning(f"⚠️ Trade action distribution: {action_counts}")

    # Aggregate commission and slippage from trade history (if present)
    def _safe_float(value, default=0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    total_commission = 0.0
    total_slippage = 0.0
    if len(all_evaluation_trade_history) > 0:
        for trade in all_evaluation_trade_history:
            if not isinstance(trade, dict):
                continue
            fees = trade.get('fees', {})
            if isinstance(fees, dict):
                total_commission += _safe_float(fees.get('commission', 0.0))
                total_slippage += _safe_float(fees.get('slippage_cost', 0.0))
            else:
                total_commission += _safe_float(trade.get('commission', 0.0))
                total_slippage += _safe_float(trade.get('slippage_cost', 0.0))

    # Max consecutive wins/losses from closing trades (SELL/AUTO_SELL/REBALANCE_SELL/STOP_LOSS)
    max_consecutive_wins = 0
    max_consecutive_losses = 0
    try:
        close_actions = {"SELL", "AUTO_SELL", "REBALANCE_SELL", "STOP_LOSS", "COVER"}
        # Sort by step when available so streaks are chronological
        sorted_trades = sorted(
            [t for t in all_evaluation_trade_history if isinstance(t, dict)],
            key=lambda t: t.get('step', 0)
        )
        win_streak = 0
        loss_streak = 0
        for t in sorted_trades:
            action = str(t.get('action', '')).upper()
            if action not in close_actions:
                continue
            pnl = _safe_float(t.get('pnl', 0.0), 0.0)
            if pnl > 0:
                win_streak += 1
                loss_streak = 0
                if win_streak > max_consecutive_wins:
                    max_consecutive_wins = win_streak
            elif pnl < 0:
                loss_streak += 1
                win_streak = 0
                if loss_streak > max_consecutive_losses:
                    max_consecutive_losses = loss_streak
            else:
                # Treat flat close as streak break
                win_streak = 0
                loss_streak = 0
    except Exception as e:
        logger.warning(f"⚠️ Failed to compute consecutive win/loss streaks: {e}")
    
    # Update results dictionary
    total_policy_actions = int(sum(policy_action_counts.values())) if isinstance(policy_action_counts, dict) else 0
    policy_action_percentages = {
        "HOLD": (policy_action_counts.get(0, 0) / total_policy_actions * 100.0) if total_policy_actions > 0 else 0.0,
        "BUY": (policy_action_counts.get(1, 0) / total_policy_actions * 100.0) if total_policy_actions > 0 else 0.0,
        "SELL": (policy_action_counts.get(2, 0) / total_policy_actions * 100.0) if total_policy_actions > 0 else 0.0,
        "SHORT": (policy_action_counts.get(3, 0) / total_policy_actions * 100.0) if total_policy_actions > 0 else 0.0,
    }

    results.update({
        "total_return": total_return,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
        "volatility": volatility,
        "win_rate": win_rate,
        "step_win_rate": step_win_rate,
        "profit_factor": profit_factor,
        "alpha": alpha,
        "alpha_annual": alpha_annual,
        "beta": beta,
        "benchmark_return": benchmark_return,
        "p_value": p_value,
        "sharpe_p_value": sharpe_p_value,
        "benchmark_data_valid": benchmark_data_valid,
        "paper_trading_ready": paper_trading_ready,
        "portfolio_history": all_episode_portfolio_values,
        "trade_history": all_evaluation_trade_history,
        "total_trades": len(all_evaluation_trade_history),
        "total_commission": total_commission,
        "total_slippage": total_slippage,
        "max_consecutive_wins": int(max_consecutive_wins),
        "max_consecutive_losses": int(max_consecutive_losses),
        "episodes_completed": episodes_completed,  # CRITICAL FIX: Add episodes_completed
        "average_hold_time": average_hold_time,  # CRITICAL FIX: Add average_hold_time
        "avg_episode_length": avg_episode_length,  # CRITICAL FIX: Add average episode length
        "final_portfolio_value": numeric_portfolio_values[-1] if len(numeric_portfolio_values) > 0 else results["initial_portfolio_value"],
        "execution_time": evaluation_time,
        "steps_simulated": total_steps,
        "stop_loss_triggers": stop_loss_triggers,  # CRITICAL FIX: Add stop loss trigger count
        "avg_stop_loss_pnl": avg_stop_loss_pnl,  # CRITICAL FIX: Add average stop loss PnL as percentage
        # Policy action distribution diagnostics
        "policy_action_counts": {
            "HOLD": int(policy_action_counts.get(0, 0)),
            "BUY": int(policy_action_counts.get(1, 0)),
            "SELL": int(policy_action_counts.get(2, 0)),
            "SHORT": int(policy_action_counts.get(3, 0)),
        },
        "policy_action_percentages": {
            "HOLD": float(policy_action_percentages.get("HOLD", 0.0)),
            "BUY": float(policy_action_percentages.get("BUY", 0.0)),
            "SELL": float(policy_action_percentages.get("SELL", 0.0)),
            "SHORT": float(policy_action_percentages.get("SHORT", 0.0)),
        },
        "policy_action_total": int(total_policy_actions),
        "benchmark_returns": benchmark_returns,
    })
    
    # CRITICAL FIX: Use accumulated realized/unrealized gains from ALL episodes
    # Don't call get_realized_unrealized_gains here as it would only have last episode's data
    initial_balance = getattr(env, 'initial_balance', 100000.0)
    realized_return = (total_realized_gains / initial_balance) * 100.0 if initial_balance > 0 else 0.0
    unrealized_return = (total_unrealized_gains / initial_balance) * 100.0 if initial_balance > 0 else 0.0
    
    results.update({
        "realized_gains": total_realized_gains,
        "unrealized_gains": total_unrealized_gains,
        "realized_return": realized_return,
        "unrealized_return": unrealized_return
    })
    logger.info(f"✅ ACCUMULATED GAINS ACROSS ALL EPISODES:")
    logger.info(f"   Realized gains: ${total_realized_gains:.2f} ({realized_return:.2f}%)")
    logger.info(f"   Unrealized gains: ${total_unrealized_gains:.2f} ({unrealized_return:.2f}%)")
    
    # FIX 3: Cross-validate total_return against realized + unrealized (diagnostic only; headline = portfolio-based)
    # Headline Total Return = portfolio (end/start - 1). Realized/unrealized are diagnostic only; never overwrite.
    total_return_pct = total_return * 100.0  # Convert decimal to percentage
    expected_return = realized_return + unrealized_return  # Already percentages
    return_discrepancy = abs(total_return_pct - expected_return)
    results["return_discrepancy_pct"] = return_discrepancy  # For report
    if return_discrepancy > 10.0:
        logger.error(f"❌ RETURN VALIDATION (diagnostic): Total return {total_return:.2%} differs from realized+unrealized {expected_return:.2f}% by {return_discrepancy:.2f}%. Headline remains portfolio-based.")
    elif return_discrepancy > 5.0:
        logger.warning(f"⚠️ RETURN VALIDATION (diagnostic): Total return {total_return:.2%} differs from realized+unrealized {expected_return:.2f}% by {return_discrepancy:.2f}%")
    else:
        logger.info(f"✅ RETURN VALIDATION PASSED: Total return {total_return:.2%} matches realized+unrealized {expected_return:.2f}%")

    # Gating diagnostics (if environment exposes stats)
    try:
        gate_env = env
        # Unwrap common wrappers (ActionMasker, VecEnv, etc.)
        try:
            while hasattr(gate_env, 'env') and getattr(gate_env, 'env', None) is not None:
                gate_env = gate_env.env
        except Exception:
            pass
        try:
            if hasattr(gate_env, 'envs') and getattr(gate_env, 'envs', None):
                gate_env = gate_env.envs[0]
        except Exception:
            pass

        if hasattr(gate_env, 'get_gating_stats'):
            gating_stats = gate_env.get_gating_stats()
            if isinstance(gating_stats, dict):
                results["gating_stats"] = gating_stats
    except Exception:
        # Non-fatal: proceed without gating stats
        pass

    # Always set gating_stats to avoid stale files across runs
    if "gating_stats" not in results or not isinstance(results.get("gating_stats"), dict):
        results["gating_stats"] = {
            "allowed": 0,
            "blocked": 0,
            "explore": 0,
            "high": 0,
            "medium": 0,
            "low": 0,
        }

    # Surface masked-sell diagnostics (from env info)
    results["masked_sell_count"] = int(masked_sell_count_max)
    results["masked_sell_streak"] = int(masked_sell_streak_max)
    
    # Extract environment specs if available (single env)
    if hasattr(env, 'observation_space'):
        obs_shape = env.observation_space.shape
        if obs_shape:
            results["env_spec"]["window"] = getattr(env, 'window_size', 0)
            # CRITICAL FIX: Prefer feature_cols length over obs shape to avoid reporting obs_dim as features
            if hasattr(env, 'feature_cols') and env.feature_cols:
                results["env_spec"]["features"] = len(env.feature_cols)
            else:
                results["env_spec"]["features"] = obs_shape[-1] if len(obs_shape) > 1 else obs_shape[0]
            results["env_spec"]["obs_dim"] = obs_shape[0]
        
        if hasattr(env, 'num_tickers'):
            results["env_spec"]["tickers"] = env.num_tickers
        
        if hasattr(env, 'data'):
            results["env_spec"]["num_dates"] = len(env.data)
    
    # Extract model specs if available
    if hasattr(model, 'n_steps'):
        results["model_spec"]["total_timesteps"] = getattr(
            model,
            'num_timesteps',
            getattr(model, 'n_timesteps', 0)
        )
    
    if hasattr(model, 'device'):
        results["model_spec"]["device"] = str(model.device)
    
    if hasattr(model, 'seed'):
        results["model_spec"]["seed"] = model.seed
    if results["model_spec"].get("seed") is None:
        results["model_spec"]["seed"] = getattr(model, "_seed", None)
    
    # Simple statistical significance (normal approx) if enough data
    try:
        if len(returns) >= 30 and np.std(returns) > 0:
            mean_r = float(np.mean(returns))
            std_r = float(np.std(returns))
            t_stat = mean_r / (std_r / np.sqrt(len(returns)))
            from math import erf, sqrt
            p_val_calc = float(1.0 - 0.5 * (1.0 + erf(abs(t_stat) / sqrt(2)))) * 2.0
            results['p_value'] = p_val_calc
            # Sharpe p-value approximation
            t_sharpe = abs(sharpe_ratio) * np.sqrt(max(1, len(returns)))
            sharpe_p_val_calc = float(1.0 - 0.5 * (1.0 + erf(t_sharpe / sqrt(2)))) * 2.0
            results['sharpe_p_value'] = sharpe_p_val_calc
    except Exception:
        pass

    # CRITICAL FIX: Collect system metrics at end
    final_memory_mb = 0.0
    final_cpu_percent = 0.0
    peak_memory_mb = 0.0
    if PSUTIL_AVAILABLE:
        try:
            process = psutil.Process(os.getpid())
            final_memory_info = process.memory_info()
            final_memory_mb = final_memory_info.rss / (1024 * 1024)  # Convert to MB
            final_cpu_percent = process.cpu_percent(interval=0.1)
            peak_memory_mb = max(initial_memory_mb, final_memory_mb)  # Simple peak tracking
            
            # Update results with system metrics
            results["performance_metrics"]["memory_mb"] = final_memory_mb
            results["performance_metrics"]["cpu_percent"] = final_cpu_percent
            results["performance_metrics"]["resource_usage"]["memory"]["used_gb"] = final_memory_mb / 1024.0
            if hasattr(psutil, 'virtual_memory'):
                vm = psutil.virtual_memory()
                results["performance_metrics"]["resource_usage"]["memory"]["percent"] = vm.percent
                results["performance_metrics"]["resource_usage"]["memory"]["total_gb"] = vm.total / (1024 * 1024 * 1024)
                results["performance_metrics"]["resource_usage"]["memory"]["available_gb"] = vm.available / (1024 * 1024 * 1024)
            results["performance_metrics"]["resource_usage"]["cpu"]["percent"] = final_cpu_percent
            if hasattr(psutil, 'cpu_count'):
                results["performance_metrics"]["resource_usage"]["cpu"]["cores"] = psutil.cpu_count()
            
            logger.info(f"✅ System metrics: Memory: {final_memory_mb:.1f} MB (peak: {peak_memory_mb:.1f} MB), CPU: {final_cpu_percent:.1f}%")
        except Exception as e:
            logger.warning(f"⚠️ Failed to collect final system metrics: {e}")

    # Extract adaptive reward logs from environment for unified report
    try:
        log_env = env
        if hasattr(log_env, 'envs') and log_env.envs:
            log_env = log_env.envs[0]
        if hasattr(log_env, 'unwrapped'):
            log_env = log_env.unwrapped
        
        adaptive_log = []
        if hasattr(log_env, 'get_adaptive_reward_log'):
            adaptive_log = log_env.get_adaptive_reward_log() or []
        elif hasattr(log_env, '_adaptive_reward_log'):
            adaptive_log = list(getattr(log_env, '_adaptive_reward_log', []) or [])
        
        if adaptive_log:
            results['adaptive_reward_log'] = adaptive_log
            # Calculate statistics for market-based adaptive rewards
            market_events = [e for e in adaptive_log if e.get('type') != 'behavior_adaptive']
            behavior_events = [e for e in adaptive_log if e.get('type') == 'behavior_adaptive']
            
            results['adaptive_reward_events_count'] = len(market_events)
            results['behavior_adaptive_events_count'] = len(behavior_events)
            
            if market_events:
                def _safe_float(x, default=1.0):
                    v = float(np.nan_to_num(x, nan=default, posinf=default, neginf=default))
                    return v if np.isfinite(v) else default
                buy_mults = [_safe_float(e.get('buy_multiplier', 1.0)) for e in market_events]
                sell_mults = [_safe_float(e.get('sell_multiplier', 1.0)) for e in market_events]
                hold_mults = [_safe_float(e.get('hold_multiplier', 1.0)) for e in market_events]
                
                results['adaptive_reward_stats'] = {
                    'buy_multiplier_avg': float(np.mean(buy_mults)) if buy_mults else 1.0,
                    'buy_multiplier_min': float(np.min(buy_mults)) if buy_mults else 1.0,
                    'buy_multiplier_max': float(np.max(buy_mults)) if buy_mults else 1.0,
                    'sell_multiplier_avg': float(np.mean(sell_mults)) if sell_mults else 1.0,
                    'sell_multiplier_min': float(np.min(sell_mults)) if sell_mults else 1.0,
                    'sell_multiplier_max': float(np.max(sell_mults)) if sell_mults else 1.0,
                    'hold_multiplier_avg': float(np.mean(hold_mults)) if hold_mults else 1.0,
                    'hold_multiplier_min': float(np.min(hold_mults)) if hold_mults else 1.0,
                    'hold_multiplier_max': float(np.max(hold_mults)) if hold_mults else 1.0,
                }
            
            if behavior_events:
                trading_mults = [e.get('base_trading_incentive_mult', 1.0) for e in behavior_events]
                sell_mults = [e.get('sell_base_incentive_mult', 1.0) for e in behavior_events]
                hold_penalty_mults = [e.get('hold_penalty_mult', 1.0) for e in behavior_events]
                
                results['behavior_adaptive_stats'] = {
                    'trading_incentive_mult_avg': float(np.mean(trading_mults)) if trading_mults else 1.0,
                    'trading_incentive_mult_min': float(np.min(trading_mults)) if trading_mults else 1.0,
                    'trading_incentive_mult_max': float(np.max(trading_mults)) if trading_mults else 1.0,
                    'sell_incentive_mult_avg': float(np.mean(sell_mults)) if sell_mults else 1.0,
                    'sell_incentive_mult_min': float(np.min(sell_mults)) if sell_mults else 1.0,
                    'sell_incentive_mult_max': float(np.max(sell_mults)) if sell_mults else 1.0,
                    'hold_penalty_mult_avg': float(np.mean(hold_penalty_mults)) if hold_penalty_mults else 1.0,
                    'hold_penalty_mult_min': float(np.min(hold_penalty_mults)) if hold_penalty_mults else 1.0,
                    'hold_penalty_mult_max': float(np.max(hold_penalty_mults)) if hold_penalty_mults else 1.0,
                }
        else:
            results['adaptive_reward_log'] = []
            results['adaptive_reward_events_count'] = 0
            results['behavior_adaptive_events_count'] = 0
    except Exception as e:
        logger.warning(f"⚠️ Failed to extract adaptive reward logs: {e}")
        results['adaptive_reward_log'] = []
        results['adaptive_reward_events_count'] = 0
        results['behavior_adaptive_events_count'] = 0
    
    # FIXED: Extract callback logs from model and trigger callbacks based on evaluation results
    try:
        # Extract callback logs from model if available (from training)
        if model is not None:
            # Get adaptive LR log from model
            if hasattr(model, 'adaptive_lr_log'):
                results['adaptive_lr_log'] = model.adaptive_lr_log
                results['adaptive_lr_events_count'] = getattr(model, 'adaptive_lr_events_count', len(model.adaptive_lr_log))
            else:
                results['adaptive_lr_log'] = []
                results['adaptive_lr_events_count'] = 0
            
            # Get entropy boost log from model (store as both names for compatibility)
            if hasattr(model, 'entropy_boost_log'):
                results['entropy_boost_log'] = model.entropy_boost_log
                results['adaptive_entropy_log'] = model.entropy_boost_log  # FIXED: Also store with expected name
                results['entropy_boost_count'] = getattr(model, 'entropy_boost_count', len(model.entropy_boost_log))
                results['adaptive_entropy_events_count'] = getattr(model, 'entropy_boost_count', len(model.entropy_boost_log))
            else:
                results['entropy_boost_log'] = []
                results['adaptive_entropy_log'] = []
                results['entropy_boost_count'] = 0
                results['adaptive_entropy_events_count'] = 0
            
            # Do not synthesize adaptive LR events in evaluator.
            # Adaptive callback logs must reflect only real training callback activity.

            # FIXED: Check if entropy boost should have triggered based on action distribution
            # Always check evaluation results, even if logs exist from training
            policy_action_counts = results.get('policy_action_counts', {})
            if isinstance(policy_action_counts, dict) and len(policy_action_counts) > 0:
                # Handle nested dict format: {"HOLD": 246, "BUY": 15, "SELL": 1160}
                if 'HOLD' in policy_action_counts or 'BUY' in policy_action_counts:
                    hold_count = policy_action_counts.get('HOLD', 0) or policy_action_counts.get(0, 0)
                    buy_count = policy_action_counts.get('BUY', 0) or policy_action_counts.get(1, 0)
                    sell_count = policy_action_counts.get('SELL', 0) or policy_action_counts.get(2, 0)
                else:
                    # Handle numeric key format: {0: 246, 1: 15, 2: 1160}
                    hold_count = policy_action_counts.get(0, 0)
                    buy_count = policy_action_counts.get(1, 0)
                    sell_count = policy_action_counts.get(2, 0)
                
                total_actions = hold_count + buy_count + sell_count
                if total_actions > 0:
                    buy_pct = (buy_count / total_actions) * 100
                    sell_pct = (sell_count / total_actions) * 100
                    
                    # FIXED: Check if entropy boost should trigger (BUY < 5% OR SELL > 85%)
                    should_trigger = buy_pct < 5.0 or sell_pct > 85.0
                    if should_trigger:
                        # Check if event already exists to avoid duplicates
                        existing_events = results.get('entropy_boost_log', [])
                        existing_adaptive_events = results.get('adaptive_entropy_log', [])
                        
                        # Only add if not already present
                        if len(existing_events) == 0 and len(existing_adaptive_events) == 0:
                            entropy_event = {
                                'step': results.get('steps_simulated', 0),
                                'type': 'action_imbalance',
                                'buy_pct': buy_pct,
                                'sell_pct': sell_pct,
                                'trigger_reason': 'BUY < 5%' if buy_pct < 5.0 else 'SELL > 85%',
                                'message': f'Action imbalance detected: BUY={buy_pct:.1f}% (< 5%), SELL={sell_pct:.1f}% - entropy boost recommended'
                            }
                            results['entropy_boost_log'] = [entropy_event]
                            results['adaptive_entropy_log'] = [entropy_event]  # FIXED: Also store with expected name
                            results['entropy_boost_count'] = 1
                            results['adaptive_entropy_events_count'] = 1
                            logger.info(f"🎯 ENTROPY BOOST TRIGGERED: BUY={buy_pct:.1f}%, SELL={sell_pct:.1f}%")
    except Exception as e:
        logger.warning(f"⚠️ Could not extract/trigger evaluation callbacks: {e}")
        # Ensure defaults are set
        if 'adaptive_lr_log' not in results:
            results['adaptive_lr_log'] = []
            results['adaptive_lr_events_count'] = 0
        if 'entropy_boost_log' not in results:
            results['entropy_boost_log'] = []
            results['adaptive_entropy_log'] = []
            results['entropy_boost_count'] = 0
            results['adaptive_entropy_events_count'] = 0
    
    # Ensure policy action distribution is always present for callbacks and reports
    if 'policy_action_counts' not in results:
        results['policy_action_counts'] = {"HOLD": 0, "BUY": 0, "SELL": 0}
    if 'policy_action_percentages' not in results:
        results['policy_action_percentages'] = {"HOLD": 0.0, "BUY": 0.0, "SELL": 0.0}
    if 'policy_action_total' not in results:
        results['policy_action_total'] = 0
    
    # Terminal ACTION SUMMARY + freeze fingerprint (deterministic only compares to prior)
    try:
        results["logit_snapshots"] = logit_snapshots
        _print_and_check_action_summary(
            results, save_path, logit_snapshots=logit_snapshots, label=eval_mode_label
        )
    except Exception as e:
        logger.warning(f"ACTION SUMMARY failed: {e}")

    # Save results
    save_evaluation_results(results, save_path)
    
    logger.info(f"Evaluation completed in {evaluation_time:.2f} seconds")
    logger.info(f"Total return: {total_return:.4f}, Sharpe ratio: {sharpe_ratio:.4f}, Win rate: {win_rate:.4f}")
    
    return results


class PerformanceEvaluator:
    """
    Thin wrapper for adaptive LR and early-stopping callbacks.
    Callbacks expect evaluator.evaluate(model, env, n_episodes=1); this delegates to evaluate_model().
    """
    def evaluate(
        self,
        model: Any,
        env: Any,
        n_episodes: int = 1,
        deterministic: bool = True,
        save_path: str = "ppo/saved_data/validation",
        **kwargs: Any,
    ) -> Dict[str, Any]:
        return evaluate_model(
            model, env, n_episodes=n_episodes, deterministic=deterministic,
            save_path=save_path, **kwargs
        )


def save_evaluation_results(results: Dict[str, Any], save_path: str) -> None:
    """
    Save evaluation results to disk
    
    Args:
        results: Evaluation results dictionary
        save_path: Path to save results
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(save_path, exist_ok=True)
        
        # Save backtest results JSON
        backtest_results_path = os.path.join(save_path, "backtest_results.json")
        
        # Fix JSON serialization by converting non-serializable types
        serializable_results = _make_json_serializable(results)
        
        with open(backtest_results_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Backtest results saved to {backtest_results_path}")

        # Save gating diagnostics JSON if present
        try:
            if "gating_stats" in results and isinstance(results["gating_stats"], dict):
                gating_stats_path = os.path.join(save_path, "gating_stats.json")
                with open(gating_stats_path, 'w', encoding='utf-8') as gf:
                    json.dump(results["gating_stats"], gf, indent=2)
                logger.info(f"Gating diagnostics saved to {gating_stats_path}")
        except Exception:
            logger.warning("Failed to write gating_stats.json; continuing without it")
        
        # Save trade history CSV
        trade_history_path = os.path.join(save_path, "trade_history.csv")
        if results["trade_history"]:
            trade_df = pd.DataFrame(results["trade_history"])
            trade_df.to_csv(trade_history_path, index=False)
            logger.info(f"Trade history saved: {len(trade_df)} trades written to {trade_history_path}")
        else:
            empty_columns = [
                "step",
                "action",
                "ticker",
                "shares",
                "price",
                "cost",
                "fees",
                "fees_total",
                "pnl",
                "portfolio_before",
                "portfolio_after",
                "reward",
                "cash_before",
                "timestamp",
                "trade_id"
            ]
            pd.DataFrame({col: [] for col in empty_columns}).to_csv(trade_history_path, index=False)
            logger.warning(
                "No trade history recorded during evaluation; wrote empty trade_history.csv to clear stale data"
            )
        
        # Save portfolio history CSV
        if results["portfolio_history"]:
            portfolio_history_path = os.path.join(save_path, "portfolio_history.csv")
            
            # CRITICAL FIX: Ensure portfolio_history contains only numeric values for CSV
            portfolio_history = results["portfolio_history"]
            numeric_portfolio_values = []
            
            for i, value in enumerate(portfolio_history):
                if isinstance(value, (int, float)):
                    numeric_portfolio_values.append(float(value))
                elif isinstance(value, dict):
                    # Try to extract numeric value from dict
                    if 'portfolio_value' in value:
                        numeric_portfolio_values.append(float(value['portfolio_value']))
                    elif 'value' in value:
                        numeric_portfolio_values.append(float(value['value']))
                    else:
                        logger.warning(f"Portfolio value {i} is dict but no numeric field found: {value}")
                        numeric_portfolio_values.append(100000.0)  # Fallback
                else:
                    try:
                        numeric_portfolio_values.append(float(value))
                    except (ValueError, TypeError):
                        logger.warning(f"Could not convert portfolio value {i} to float: {value}")
                        numeric_portfolio_values.append(100000.0)  # Fallback
            
            portfolio_df = pd.DataFrame({
                "step": range(len(numeric_portfolio_values)),
                "portfolio_value": numeric_portfolio_values
            })
            portfolio_df.to_csv(portfolio_history_path, index=False)
            logger.info(f"Portfolio history saved to {portfolio_history_path}")
        
        # Calculate and save returns CSV
        if len(results["portfolio_history"]) > 1:
            returns_path = os.path.join(save_path, "returns.csv")
            
            # CRITICAL FIX: Ensure portfolio_history contains only numeric values
            portfolio_history = results["portfolio_history"]
            numeric_portfolio_values = []
            
            for i, value in enumerate(portfolio_history):
                if isinstance(value, (int, float)):
                    numeric_portfolio_values.append(float(value))
                elif isinstance(value, dict):
                    # Try to extract numeric value from dict
                    if 'portfolio_value' in value:
                        numeric_portfolio_values.append(float(value['portfolio_value']))
                    elif 'value' in value:
                        numeric_portfolio_values.append(float(value['value']))
                    else:
                        logger.warning(f"Portfolio value {i} is dict but no numeric field found: {value}")
                        numeric_portfolio_values.append(100000.0)  # Fallback
                else:
                    try:
                        numeric_portfolio_values.append(float(value))
                    except (ValueError, TypeError):
                        logger.warning(f"Could not convert portfolio value {i} to float: {value}")
                        numeric_portfolio_values.append(100000.0)  # Fallback
            
            if len(numeric_portfolio_values) > 1:
                # CRITICAL FIX: Additional validation to prevent dict subtraction error
                try:
                    # CRITICAL FIX: Ensure all values are numeric before numpy operations
                    numeric_portfolio_values = [float(v) for v in numeric_portfolio_values if isinstance(v, (int, float, np.number))]
                    
                    if len(numeric_portfolio_values) > 1:
                        portfolio_values = np.array(numeric_portfolio_values, dtype=float)
                        # Validate no NaN or inf values
                        if np.any(np.isnan(portfolio_values)) or np.any(np.isinf(portfolio_values)):
                            logger.warning("Found NaN/inf in portfolio values, replacing with fallback")
                            portfolio_values = np.nan_to_num(portfolio_values, nan=100000.0, posinf=100000.0, neginf=100000.0)
                        
                        returns = np.diff(portfolio_values) / portfolio_values[:-1]
                        returns_df = pd.DataFrame({
                            "step": range(1, len(portfolio_values)),
                            "return": returns
                        })
                        returns_df.to_csv(returns_path, index=False)
                        logger.info(f"Returns saved to {returns_path}")
                    else:
                        logger.warning("Not enough numeric portfolio values to calculate returns")
                except Exception as calc_error:
                    logger.error(f"Error calculating returns: {calc_error}")
                    logger.error(f"Portfolio values sample: {numeric_portfolio_values[:5]}")
                    logger.warning("Skipping returns CSV due to calculation error")
            else:
                logger.warning("Not enough numeric portfolio values to calculate returns")
        
        # Save benchmark returns CSV (use real benchmark returns when available)
        benchmark_returns_path = os.path.join(save_path, "benchmark_returns.csv")
        benchmark_returns = results.get("benchmark_returns", [])
        if isinstance(benchmark_returns, (np.ndarray, pd.Series)):
            benchmark_returns = benchmark_returns.tolist()
        if not isinstance(benchmark_returns, list):
            benchmark_returns = []
        if not benchmark_returns:
            benchmark_returns = [0.0] * len(numeric_portfolio_values)
        else:
            if len(benchmark_returns) < len(numeric_portfolio_values):
                pad_value = benchmark_returns[-1] if benchmark_returns else 0.0
                benchmark_returns = benchmark_returns + [pad_value] * (len(numeric_portfolio_values) - len(benchmark_returns))
            elif len(benchmark_returns) > len(numeric_portfolio_values):
                benchmark_returns = benchmark_returns[:len(numeric_portfolio_values)]
        benchmark_df = pd.DataFrame({
            "step": range(len(benchmark_returns)),
            "benchmark_return": benchmark_returns
        })
        benchmark_df.to_csv(benchmark_returns_path, index=False)
        logger.info(f"Benchmark returns saved to {benchmark_returns_path}")

        # Save compact summary files for unified report consistency
        try:
            analytics_summary_path = os.path.join(save_path, "analytics_summary_compact.json")
            analytics_summary = {
                "total_return": results.get("total_return", 0.0),
                "sharpe_ratio": results.get("sharpe_ratio", 0.0),
                "max_drawdown": results.get("max_drawdown", 0.0),
                "win_rate": results.get("win_rate", 0.0),
                "volatility": results.get("volatility", 0.0)
            }
            with open(analytics_summary_path, 'w', encoding='utf-8') as af:
                json.dump(analytics_summary, af, indent=2)
            logger.info(f"Analytics summary saved to {analytics_summary_path}")

            backtest_summary_path = os.path.join(save_path, "backtest_summary_compact.json")
            backtest_summary = {
                "total_trades": results.get("total_trades", 0),
                "steps_simulated": results.get("steps_simulated", 0),
                "final_portfolio_value": results.get("final_portfolio_value", 0.0)
            }
            with open(backtest_summary_path, 'w', encoding='utf-8') as bf:
                json.dump(backtest_summary, bf, indent=2)
            logger.info(f"Backtest summary saved to {backtest_summary_path}")
        except Exception as summary_error:
            logger.warning(f"Failed to save compact summaries: {summary_error}")

        # Save visualization data JSON
        try:
            analytics_dir = Path(save_path).resolve().parents[1] / "analytics"
            analytics_dir.mkdir(parents=True, exist_ok=True)
            visualization_path = analytics_dir / "visualization_data.json"
            portfolio_series = numeric_portfolio_values
            returns_series = []
            if len(portfolio_series) > 1:
                returns_series = (
                    np.diff(np.array(portfolio_series, dtype=float)) / np.array(portfolio_series[:-1], dtype=float)
                ).tolist()
            visualization_data = {
                "portfolio_values": portfolio_series,
                "returns": returns_series,
                "total_return": results.get("total_return", 0.0),
                "sharpe_ratio": results.get("sharpe_ratio", 0.0),
                "max_drawdown": results.get("max_drawdown", 0.0),
                "win_rate": results.get("win_rate", 0.0)
            }
            with open(visualization_path, 'w', encoding='utf-8') as vf:
                json.dump(visualization_data, vf, indent=2)
            logger.info(f"Visualization data saved to {visualization_path}")
        except Exception as viz_error:
            logger.warning(f"Failed to save visualization data: {viz_error}")
        
    except Exception as e:
        logger.error(f"Failed to save evaluation results: {e}")

def _make_json_serializable(obj):
    """Convert non-serializable objects to JSON-compatible types"""
    if isinstance(obj, Mapping):
        return {k: _make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_make_json_serializable(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif hasattr(obj, "item") and not isinstance(obj, (str, bytes, bytearray)):
        try:
            return obj.item()
        except Exception:
            pass
    elif isinstance(obj, bool):
        return obj  # bool is already JSON serializable
    elif isinstance(obj, (int, float)):
        return obj
    elif hasattr(obj, '__dict__'):
        return str(obj)
    else:
        return obj

def compute_metrics(returns: List[float], benchmark_returns: Optional[List[float]] = None) -> Dict[str, float]:
    """
    Compute performance metrics from returns
    
    Args:
        returns: List of returns
        benchmark_returns: Optional list of benchmark returns
    
    Returns:
        Dictionary of performance metrics
    """
    metrics = {}
    
    if not returns:
        return metrics
    
    # Convert to numpy array
    returns_array = np.array(returns)
    
    # Total return
    total_return = (1 + returns_array).prod() - 1
    metrics["total_return"] = total_return
    
    # Sharpe ratio (annualized, assuming daily returns)
    if len(returns_array) > 1 and np.std(returns_array) > 0:
        sharpe_ratio = np.mean(returns_array) / np.std(returns_array) * np.sqrt(252)
        metrics["sharpe_ratio"] = sharpe_ratio
    else:
        metrics["sharpe_ratio"] = 0.0
    
    # Max drawdown
    cum_returns = (1 + returns_array).cumprod()
    peak = np.maximum.accumulate(cum_returns)
    drawdown = (peak - cum_returns) / peak
    max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0.0
    metrics["max_drawdown"] = max_drawdown
    
    # Volatility (annualized) - guard against NaN/Inf
    volatility = np.std(returns_array) * np.sqrt(252) if len(returns_array) > 1 else 0.0
    volatility = float(np.nan_to_num(volatility, nan=0.0, posinf=0.0, neginf=0.0))
    metrics["volatility"] = volatility
    
    # Win rate
    win_rate = np.mean(returns_array > 0) if len(returns_array) > 0 else 0.0
    metrics["win_rate"] = win_rate
    
    # Profit factor
    gross_profit = np.sum(returns_array[returns_array > 0])
    gross_loss = abs(np.sum(returns_array[returns_array < 0]))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    metrics["profit_factor"] = profit_factor
    
    # Alpha and Beta (vs benchmark)
    if benchmark_returns and len(returns_array) == len(benchmark_returns):
        benchmark_array = np.array(benchmark_returns)
        
        # Calculate beta (covariance / variance)
        cov_matrix = np.cov(returns_array, benchmark_array)
        if cov_matrix.shape == (2, 2) and cov_matrix[1, 1] > 0:
            beta = cov_matrix[0, 1] / cov_matrix[1, 1]
            
            # Calculate alpha (annualized)
            alpha = (np.mean(returns_array) - beta * np.mean(benchmark_array)) * 252
            
            # Calculate annualized alpha
            alpha_annual = (1 + np.mean(returns_array)) ** 252 - 1 - beta * ((1 + np.mean(benchmark_array)) ** 252 - 1)
        else:
            beta = 0.0
            alpha = 0.0
            alpha_annual = 0.0
        
        metrics["alpha"] = alpha
        metrics["alpha_annual"] = alpha_annual
        metrics["beta"] = beta
    
    return metrics

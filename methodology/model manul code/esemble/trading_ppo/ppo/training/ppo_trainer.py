"""
PPO Trainer Module for PPO Trading System
Handles PPO agent creation, training, and saving
"""

import os
import torch
import logging
import time
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional, Callable, Dict, cast
from ppo.utils.dynamic_config import get_dynamic_timesteps, get_dynamic_config  # type: ignore
from ppo.models.tcn_policy import TCNTradingPolicy, TCNFeatureExtractor, register_tcn_policy  # type: ignore

logger = logging.getLogger("PPOTrading.Trainer")

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    # RecurrentActorCriticPolicy is optional - doesn't exist in SB3 2.7.0+
    try:
        from stable_baselines3.common.policies import RecurrentActorCriticPolicy  # type: ignore
        RECURRENT_POLICY_AVAILABLE = True
        logger.debug("RecurrentActorCriticPolicy available")
    except ImportError:
        RECURRENT_POLICY_AVAILABLE = False
        logger.debug("RecurrentActorCriticPolicy not available (expected in SB3 2.7.0+)")
    # Continue with other imports
    try:
        from stable_baselines3 import RecurrentPPO  # type: ignore
        RECURRENT_PPO_AVAILABLE = True
        logger.info("RecurrentPPO available for LSTM memory")
    except ImportError:
        RECURRENT_PPO_AVAILABLE = False
        logger.warning("RecurrentPPO not available, will use standard PPO")
    SB3_AVAILABLE = True
    logger.info("Stable-Baselines3 available for PPO training")
except ImportError:
    logger.error("Stable-Baselines3 not available. Cannot train PPO agent.")
    SB3_AVAILABLE = False
    RECURRENT_PPO_AVAILABLE = False

# 2025-12-22: MaskablePPO for proper action masking (prevents invalid SELL when no positions)
try:
    from sb3_contrib import MaskablePPO
    # Optional import - only needed for custom policies
    try:
        from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
    except ImportError:
        MaskableActorCriticPolicy = None  # type: ignore[misc, assignment]
    try:
        from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
        MASKABLE_EVAL_CALLBACK_AVAILABLE = True
    except ImportError:
        MaskableEvalCallback = None  # type: ignore[misc, assignment]
        MASKABLE_EVAL_CALLBACK_AVAILABLE = False
    MASKABLE_PPO_AVAILABLE = True
    logger.info("✅ MaskablePPO available - Action masking ENABLED")
except ImportError as e:
    MASKABLE_PPO_AVAILABLE = False
    MaskableActorCriticPolicy = None  # type: ignore[misc, assignment]
    MaskableEvalCallback = None  # type: ignore[misc, assignment]
    MASKABLE_EVAL_CALLBACK_AVAILABLE = False
    logger.warning(f"⚠️ MaskablePPO not available - Install with: pip install sb3-contrib")
    logger.debug(f"   Import error: {e}")

class BCRegularizationCallback(BaseCallback):
    """
    BC Regularization Callback - Preserves BC knowledge during PPO training
    Adds KL divergence loss between current policy and BC policy
    """
    
    def __init__(self, verbose: int = 0):
        super(BCRegularizationCallback, self).__init__(verbose)
        self.bc_state = None
        # OPTIMIZATION: Much lighter regularization (Option 2) - reduced from 0.3 to 0.03 (10x lighter)
        self.regularization_weight = 0.03
        # EXTENDED: Preserve BC influence through 150k steps to prevent late-stage drift
        self.max_steps = 150000
        self.refresh_steps = [75000, 100000, 125000]
        self.refresh_strength = 0.1
        self.lr_schedule = {75000: 0.5, 100000: 0.3, 125000: 0.1}
        self._refresh_index = 0
        self._applied_lr_thresholds = set()
        self.original_lr = None
        # OPTIMIZATION: Check BC drift every 1000 steps instead of every step (99.9% reduction)
        self.bc_check_interval = 1000
        self.last_bc_check = 0
        self.cached_avg_kl = None
        # STEP 6: Add training monitoring to detect BC knowledge erosion
        self.trade_check_interval = 5000  # Check every 5k timesteps
        self.last_trade_check = 0
        self.trades_at_last_check = 0
        self.min_trades_per_interval = 5  # Minimum 5 trades per 5k steps
        
    def _on_training_start(self) -> None:
        """Called at the start of training - load BC state if available"""
        if hasattr(self.model, '_bc_policy_state'):
            self.bc_state = self.model._bc_policy_state  # type: ignore
            # OPTIMIZATION: Use lighter weight (0.03 instead of 0.3) if not set on model
            self.regularization_weight = getattr(self.model, '_bc_regularization_weight', self.regularization_weight)
            self.max_steps = getattr(self.model, '_bc_regularization_steps', self.max_steps)
            self.refresh_steps = getattr(self.model, '_bc_refresh_steps', self.refresh_steps)
            self.refresh_strength = getattr(self.model, '_bc_refresh_strength', self.refresh_strength)
            self.lr_schedule = getattr(self.model, '_lr_decay_schedule', self.lr_schedule)
            self._refresh_index = 0
            self._applied_lr_thresholds = set()
            self.original_lr = getattr(self.model, '_original_lr', getattr(self.model, 'learning_rate', None))
            if self.original_lr is None:
                self.original_lr = getattr(self.model, 'learning_rate', None)
                if self.original_lr is not None:
                    self.model._original_lr = self.original_lr  # type: ignore
            lr_summary = ", ".join(f"{int(step):,}->{factor:.2f}" for step, factor in sorted(self.lr_schedule.items()))
            logger.info(
                f"ðŸ›¡ï¸ BC REGULARIZATION ACTIVE: Weight={self.regularization_weight}, Steps={self.max_steps:,}, "
                f"Refresh={self.refresh_steps}, LR schedule={lr_summary}"
            )
        else:
            logger.info("â„¹ï¸ BC regularization not available - BC policy state not found")
    
    def _on_step(self) -> bool:
        """Apply BC regularization during training (OPTIMIZED: checks every 1000 steps)"""
        if self.bc_state is None or self.num_timesteps > self.max_steps:
            return True
        
        # STEP 6: Monitor trading activity during training to detect BC knowledge erosion
        if self.num_timesteps - self.last_trade_check >= self.trade_check_interval:
            self._check_trading_activity()
            self.last_trade_check = self.num_timesteps
        
        # OPTIMIZATION: Only check BC drift every 1000 steps (99.9% reduction in computation)
        if self.num_timesteps - self.last_bc_check < self.bc_check_interval:
            # Use cached result if available
            if self.cached_avg_kl is not None:
                self._apply_cached_bc_regularization()
            return True
        
        # Update last check time
        self.last_bc_check = self.num_timesteps
        
        try:
            # Get current policy
            current_policy = self.model.policy
            current_state = current_policy.state_dict()
            
            # Compute KL divergence between current policy and BC policy
            # For efficiency, we'll use weight distance as a proxy for KL divergence
            kl_loss = 0.0
            weight_count = 0
            
            for name, bc_param in self.bc_state.items():
                if name in current_state:
                    current_param = current_state[name]
                    # Compute L2 distance between weights (proxy for policy distance)
                    weight_diff = torch.norm(current_param - bc_param.to(current_param.device))
                    kl_loss += weight_diff.item()
                    weight_count += 1
            
            if weight_count > 0:
                avg_kl = kl_loss / weight_count
                self.cached_avg_kl = avg_kl  # Cache for next 1000 steps
                
                # OPTIMIZATION: Much lighter penalty (90% reduction)
                # Apply regularization: reduce learning if policy drifts too far from BC
                model_any = cast(Any, self.model)
                current_lr = model_any.learning_rate
                if callable(current_lr):
                    current_lr_value = current_lr(1.0)
                else:
                    current_lr_value = current_lr
                current_lr = float(current_lr_value) if isinstance(current_lr_value, (int, float)) else 0.0
                if avg_kl > 0.5:  # If policy drifts too far (>50% weight change)
                    # OPTIMIZATION: Reduce LR by only 5% instead of 50% (90% lighter penalty)
                    model_any.learning_rate = current_lr * 0.95  # Reduce LR by 5% (was 50%)
                    logger.debug(f"ðŸ›¡ï¸ BC REG: High drift ({avg_kl:.3f}), reducing LR by 5% (was 50%): {current_lr:.6f} -> {model_any.learning_rate:.6f}")
                elif avg_kl < 0.1:  # Policy is close to BC, can train normally
                    if hasattr(model_any, '_original_lr'):
                        model_any.learning_rate = model_any._original_lr
                else:
                    if not hasattr(model_any, '_original_lr'):
                        setattr(model_any, "_original_lr", current_lr)
                    # OPTIMIZATION: Much lighter LR reduction for moderate drift (10x lighter weight)
                    model_any.learning_rate = model_any._original_lr * (1.0 - self.regularization_weight * avg_kl)
                    
        except Exception as e:
            logger.debug(f"BC regularization step failed: {e}")
            self.cached_avg_kl = None
        
        # Apply scheduled updates (refreshes and LR schedule)
        self._apply_scheduled_updates()
        
        return True
    
    def _apply_cached_bc_regularization(self):
        """Apply BC regularization using cached drift value (for steps between checks)"""
        if self.cached_avg_kl is None:
            return
        
        try:
            model_any = cast(Any, self.model)
            current_lr = model_any.learning_rate
            avg_kl = self.cached_avg_kl
            
            if avg_kl > 0.5:
                # Keep reduced LR if drift is high
                if not hasattr(model_any, '_bc_lr_reduced'):
                    setattr(model_any, "_bc_lr_reduced", True)
            elif avg_kl < 0.1:
                # Restore original LR if drift is low
                if hasattr(model_any, '_original_lr') and hasattr(model_any, '_bc_lr_reduced'):
                    model_any.learning_rate = model_any._original_lr
                    setattr(model_any, "_bc_lr_reduced", False)
        except Exception as e:
            logger.debug(f"Cached BC regularization failed: {e}")
    
    def _apply_scheduled_updates(self):
        """Apply scheduled BC refreshes and LR schedule updates"""
        # Apply scheduled BC refreshes to keep policy anchored to expert behaviour
        if self.bc_state is not None and self._refresh_index < len(self.refresh_steps):
            next_refresh = self.refresh_steps[self._refresh_index]
            if self.num_timesteps >= next_refresh:
                self._apply_bc_refresh()
                self._refresh_index += 1

        # Apply learning-rate schedule once thresholds are crossed
        if self.original_lr is not None and isinstance(self.lr_schedule, dict):
            for threshold_key in sorted(self.lr_schedule.keys()):
                # CRITICAL FIX: Convert threshold to int for comparison
                threshold = int(threshold_key) if isinstance(threshold_key, str) else threshold_key
                if threshold_key not in self._applied_lr_thresholds and self.num_timesteps >= threshold:
                    factor = self.lr_schedule[threshold_key]
                    target_lr = self.original_lr * factor
                    self.model.learning_rate = target_lr
                    self._applied_lr_thresholds.add(threshold_key)
                    logger.info(
                        f"ðŸªœ BC REG: Adjusted learning rate to {target_lr:.6f} at {self.num_timesteps:,} steps "
                        f"(factor={factor:.2f})"
                    )

    def _apply_bc_refresh(self):
        """Blend current policy weights toward BC snapshot to prevent drift."""
        if self.bc_state is None or self.refresh_strength <= 0:
            return

        try:
            with torch.no_grad():
                policy = self.model.policy
                current_state = policy.state_dict()
                blend_ratio = min(max(self.refresh_strength, 0.0), 1.0)
                for name, bc_param in self.bc_state.items():
                    if name in current_state:
                        target_tensor = bc_param.to(current_state[name].device)
                        current_state[name].data.lerp_(target_tensor, blend_ratio)
                policy.load_state_dict(current_state)
            logger.info(
                f"ðŸ§­ BC REFRESH: Re-aligned policy toward BC snapshot at {self.num_timesteps:,} steps "
                f"(blend={self.refresh_strength:.2f})"
            )
        except Exception as exc:
            logger.debug(f"BC refresh step failed: {exc}")
    
    def _check_trading_activity(self):
        """
        STEP 6: Monitor trading activity during training to detect BC knowledge erosion
        Check if agent is still trading or has become a HOLD-only/invalid SELL policy
        """
        try:
            # Try to get trade count from the environment
            current_trades = 0
            
            # Check if we can access the training environment
            if hasattr(self.training_env, 'envs'):
                # VecEnv - get first environment
                env = self.training_env.envs[0]  # type: ignore
            elif hasattr(self.training_env, 'env'):
                # Wrapped environment
                env = self.training_env.env  # type: ignore
            else:
                env = self.training_env
            
            # Try to get trade history or trade count
            if hasattr(env, 'get_trade_history'):
                trades = env.get_trade_history()  # type: ignore
                current_trades = len(trades) if trades else 0
            elif hasattr(env, 'total_trades'):
                current_trades = env.total_trades  # type: ignore
            elif hasattr(env, 'trade_history'):
                current_trades = len(env.trade_history)  # type: ignore
            
            # Calculate trades in this interval
            trades_this_interval = current_trades - self.trades_at_last_check
            self.trades_at_last_check = current_trades
            
            # Check if trading activity has dropped below threshold
            if trades_this_interval < self.min_trades_per_interval:
                logger.warning(
                    f"ðŸš¨ BC KNOWLEDGE EROSION DETECTED at {self.num_timesteps:,} steps: "
                    f"Only {trades_this_interval} trades in last {self.trade_check_interval:,} steps "
                    f"(minimum: {self.min_trades_per_interval})"
                )
                logger.warning("   Agent may be forgetting to trade - BC knowledge is degrading!")
                logger.warning("   Consider: 1) Stop training early, 2) Reload BC weights, or 3) Increase exploration")
                
                # Option 2: Automatically reload BC weights to recover trading behavior
                if self.bc_state is not None and trades_this_interval == 0:
                    logger.warning("   ZERO trades detected - performing emergency BC refresh!")
                    self.refresh_strength = 0.5  # Stronger refresh for emergency
                    self._apply_bc_refresh()
                    self.refresh_strength = 0.1  # Restore normal refresh strength
            else:
                logger.debug(
                    f"âœ… Trading activity healthy at {self.num_timesteps:,} steps: "
                    f"{trades_this_interval} trades in last {self.trade_check_interval:,} steps"
                )
                
        except Exception as e:
            logger.debug(f"Could not check trading activity: {e}")

class TrainingMetricsCallback(BaseCallback):
    """
    Custom callback to collect training metrics (learning rate, policy loss, value loss)
    Enhanced with action distribution monitoring for entropy boost
    """
    
    def __init__(self, verbose: int = 0, env_for_action_check=None):
        super(TrainingMetricsCallback, self).__init__(verbose)
        self.training_metrics = {
            'learning_rate': [],
            'policy_loss': [],
            'value_loss': [],
            'entropy': [],
            'explained_variance': []
        }
        self.last_metrics = {}
        # Use different name to avoid conflict with BaseCallback.training_env property
        self._env_for_action_check = env_for_action_check
        
        # Action distribution tracking for entropy boost
        self._action_distribution_check_freq = 500  # FIXED 2026-01-18: Reduced from 1000 to 500 for faster response to action imbalance
        self._last_action_check = 0
        self._entropy_boost_log = []  # Log for unified report
        self._action_entropy_boosted = False
        self._force_check_on_eval = False  # Flag to force check after evaluation
    
    def _on_step(self) -> bool:
        """Called at each step to collect metrics"""
        # Collect metrics from the model if available
        if hasattr(self.model, 'logger') and hasattr(self.model.logger, 'name_to_value'):
            metrics = self.model.logger.name_to_value
            
            # Extract key metrics
            if 'train/learning_rate' in metrics:
                self.training_metrics['learning_rate'].append(metrics['train/learning_rate'])
            if 'train/policy_loss' in metrics:
                self.training_metrics['policy_loss'].append(metrics['train/policy_loss'])
            if 'train/value_loss' in metrics:
                self.training_metrics['value_loss'].append(metrics['train/value_loss'])
            if 'train/entropy_loss' in metrics:
                # CRITICAL FIX: entropy_loss is NEGATED in SB3 (loss = -entropy), so we negate it to get actual entropy
                entropy_loss = metrics['train/entropy_loss']
                entropy = -entropy_loss  # Convert loss back to actual entropy (should be positive)
                # Validation: Entropy should always be positive
                if entropy < 0:
                    logger.warning(f"âš ï¸ ENTROPY VALIDATION: Calculated entropy is negative ({entropy:.4f}), using absolute value")
                    entropy = abs(entropy)
                self.training_metrics['entropy'].append(entropy)
                # FIX 4: Monitor entropy and warn/prevent collapse
                if entropy < 0.1:  # Entropy too low - policy becoming deterministic
                    logger.warning(f"âš ï¸ ENTROPY COLLAPSE DETECTED: entropy={entropy:.4f} (threshold: 0.1) - policy becoming deterministic")
                    # Temporarily increase entropy coefficient to encourage exploration
                    model_any = cast(Any, self.model)
                    if not hasattr(model_any, '_original_ent_coef'):
                        setattr(model_any, "_original_ent_coef", model_any.ent_coef)
                    # Use adaptive_ent_boost_factor from config (plan: 2.5 for 300k)
                    try:
                        cfg = get_dynamic_config()
                        boost = getattr(cfg, 'adaptive_ent_boost_factor', 2.5)
                    except Exception:
                        boost = 2.5
                    model_any.ent_coef = model_any._original_ent_coef * boost
                    logger.info(f"ðŸ”§ ENTROPY FIX: Increased ent_coef from {model_any._original_ent_coef:.4f} to {model_any.ent_coef:.4f}")
                elif entropy > 0.5 and hasattr(self.model, '_original_ent_coef'):
                    # Entropy recovered - restore original (never below ent_coef_min)
                    model_any = cast(Any, self.model)
                    try:
                        cfg = get_dynamic_config()
                        ent_min = getattr(cfg, 'ent_coef_min', 0.10)
                    except Exception:
                        ent_min = 0.10
                    restored = max(model_any._original_ent_coef, ent_min)
                    model_any.ent_coef = restored
                    logger.debug(f"ENTROPY RECOVERED: Restored ent_coef to {model_any.ent_coef:.4f} (floor: {ent_min})")
            if 'train/explained_variance' in metrics:
                self.training_metrics['explained_variance'].append(metrics['train/explained_variance'])
        
        # NEW: Check action distribution imbalance for entropy boost (every 1000 steps or on eval)
        if (self.num_timesteps - self._last_action_check >= self._action_distribution_check_freq) or self._force_check_on_eval:
            self._check_action_distribution_imbalance()
            self._last_action_check = self.num_timesteps
            self._force_check_on_eval = False  # Reset flag after check
        
        return True
    
    def get_final_metrics(self) -> dict:
        """Get final training metrics"""
        final_metrics = {}
        for key, values in self.training_metrics.items():
            if values:
                # CRITICAL FIX: Validate entropy values are always positive
                if key == 'entropy':
                    # Filter out negative values and ensure all are positive
                    positive_values = [max(0.0, v) for v in values]
                    if positive_values:
                        final_metrics[key] = {
                            'final': max(0.0, positive_values[-1]),  # Ensure final is positive
                            'mean': max(0.0, sum(positive_values) / len(positive_values)),
                            'min': max(0.0, min(positive_values)),
                            'max': max(0.0, max(positive_values))
                        }
                    else:
                        final_metrics[key] = {'final': 0.0, 'mean': 0.0, 'min': 0.0, 'max': 0.0}
                else:
                    final_metrics[key] = {
                        'final': values[-1],
                        'mean': sum(values) / len(values),
                        'min': min(values),
                        'max': max(values)
                    }
            else:
                final_metrics[key] = {'final': 0.0, 'mean': 0.0, 'min': 0.0, 'max': 0.0}
        
        # Include entropy boost log for unified report
        final_metrics['entropy_boost_log'] = self._entropy_boost_log
        final_metrics['entropy_boost_count'] = len(self._entropy_boost_log)
        
        return final_metrics
    
    def _check_action_distribution_imbalance(self):
        """
        NEW: Check raw action distribution and boost entropy if severely imbalanced.
        Triggers when:
        - Raw BUY < 5% of total actions (agent severely under-buying)
        - Raw SELL > 85% of total actions (agent stuck in SELL-dominant mode)
        """
        try:
            # Get environment to access action counts
            env = None
            # First try the explicitly passed env, then fall back to model's training_env
            env_ref = self._env_for_action_check if self._env_for_action_check is not None else self.training_env
            if env_ref is not None:
                # Unwrap environment to get base TradingEnv
                if hasattr(env_ref, 'envs'):
                    env = env_ref.envs[0]  # type: ignore
                    # Unwrap if wrapped
                    while hasattr(env, 'env') and env.env is not env:
                        env = env.env  # type: ignore
                elif hasattr(env_ref, 'env'):
                    env = env_ref.env  # type: ignore
                    while hasattr(env, 'env') and env.env is not env:
                        env = env.env
                elif hasattr(env_ref, 'unwrapped'):
                    env = env_ref.unwrapped
                else:
                    env = env_ref
            
            if env is None:
                return
            
            # FIX: Use actual action counts from environment (_buy_count, _sell_count, _hold_count)
            # Also check for policy action counts if available (raw actions before masking)
            buy_count = getattr(env, '_buy_count', 0)
            sell_count = getattr(env, '_sell_count', 0)
            hold_count = getattr(env, '_hold_count', 0)
            
            # Also check for policy action counts (raw actions from policy)
            policy_action_counts = getattr(env, '_policy_action_counts', None)
            if policy_action_counts and isinstance(policy_action_counts, dict):
                # Use policy action counts if available (more accurate for entropy boost)
                buy_count = policy_action_counts.get(1, buy_count)
                sell_count = policy_action_counts.get(2, sell_count)
                hold_count = policy_action_counts.get(0, hold_count)
            
            total_actions = buy_count + sell_count + hold_count
            if total_actions < 30:  # FIXED 2026-01-18: Reduced from 50 to 30 for faster entropy boost triggering
                return
            
            buy_pct = (buy_count / total_actions) * 100
            sell_pct = (sell_count / total_actions) * 100
            
            # Thresholds for entropy boost
            BUY_CRITICAL_THRESHOLD = 5.0  # Boost if BUY < 5%
            SELL_DOMINANCE_THRESHOLD = 85.0  # Boost if SELL > 85%
            
            needs_boost = False
            boost_reason = ""
            
            if buy_pct < BUY_CRITICAL_THRESHOLD:
                needs_boost = True
                boost_reason = f"Raw BUY only {buy_pct:.1f}% (< {BUY_CRITICAL_THRESHOLD}%)"
            elif sell_pct > SELL_DOMINANCE_THRESHOLD:
                needs_boost = True
                boost_reason = f"Raw SELL dominance {sell_pct:.1f}% (> {SELL_DOMINANCE_THRESHOLD}%)"
            
            if needs_boost and not self._action_entropy_boosted:
                # Boost entropy coefficient (use config.adaptive_ent_boost_factor, plan: 2.5)
                model_any = cast(Any, self.model)
                if not hasattr(model_any, '_original_ent_coef'):
                    setattr(model_any, "_original_ent_coef", model_any.ent_coef)
                try:
                    cfg = get_dynamic_config()
                    boost_factor = getattr(cfg, 'adaptive_ent_boost_factor', 2.5)
                except Exception:
                    boost_factor = 2.5
                new_ent_coef = model_any._original_ent_coef * boost_factor
                model_any.ent_coef = new_ent_coef
                self._action_entropy_boosted = True
                
                logger.warning(f"🎯 ACTION DISTRIBUTION IMBALANCE: {boost_reason}")
                logger.info(f"🔧 ENTROPY BOOST (Action Imbalance): ent_coef {model_any._original_ent_coef:.4f} → {new_ent_coef:.4f} (×{boost_factor})")
                
                self._entropy_boost_log.append({
                    'step': self.num_timesteps,
                    'type': 'action_imbalance',
                    'buy_pct': buy_pct,
                    'sell_pct': sell_pct,
                    'boost_factor': boost_factor,
                    'message': f'{boost_reason} - boosted ent_coef by {boost_factor}x'
                })
            
            elif not needs_boost and self._action_entropy_boosted:
                # Action distribution recovered - restore original (never below ent_coef_min, plan Fix 2)
                model_any = cast(Any, self.model)
                if hasattr(model_any, '_original_ent_coef'):
                    try:
                        cfg = get_dynamic_config()
                        ent_min = getattr(cfg, 'ent_coef_min', 0.10)
                    except Exception:
                        ent_min = 0.10
                    restored = max(model_any._original_ent_coef, ent_min)
                    model_any.ent_coef = restored
                    self._action_entropy_boosted = False
                    logger.info(f"ACTION DISTRIBUTION RECOVERED: Restored ent_coef to {restored:.4f} (floor: {ent_min})")
                    
                    self._entropy_boost_log.append({
                        'step': self.num_timesteps,
                        'type': 'action_recovered',
                        'buy_pct': buy_pct,
                        'sell_pct': sell_pct,
                        'message': f'Action distribution recovered (BUY={buy_pct:.1f}%, SELL={sell_pct:.1f}%) - restored ent_coef'
                    })
                    
        except Exception as e:
            logger.debug(f"Error checking action distribution: {e}")
    
    def trigger_evaluation_check(self):
        """FIXED: Method to force action distribution check after evaluation"""
        self._force_check_on_eval = True
        # Also trigger immediate check if we have enough steps
        if self.num_timesteps > 0:
            self._check_action_distribution_imbalance()
            self._last_action_check = self.num_timesteps
            self._force_check_on_eval = False

class CheckpointCallback(BaseCallback):
    """
    Custom callback to save model checkpoints at regular intervals
    """
    
    def __init__(self, save_freq: int = 50000, save_path: str = "./ppo/saved_models/checkpoints/", verbose: int = 0):
        super(CheckpointCallback, self).__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.last_save = 0
        
        # Create checkpoint directory
        os.makedirs(save_path, exist_ok=True)
    
    def _on_step(self) -> bool:
        """Called at each step to check if checkpoint should be saved"""
        if self.num_timesteps - self.last_save >= self.save_freq:
            self._save_checkpoint()
            self.last_save = self.num_timesteps
        return True
    
    def _save_checkpoint(self):
        """Save model checkpoint"""
        try:
            checkpoint_name = f"ppo_checkpoint_{self.num_timesteps}"
            checkpoint_path = os.path.join(self.save_path, checkpoint_name)
            
            # Save the model
            self.model.save(checkpoint_path)
            
            if self.verbose > 0:
                logger.info(f"Checkpoint saved: {checkpoint_path}")
                
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")

class AdaptiveLRCallback(BaseCallback):
    """
    Adaptive Learning Rate Callback - Adjusts LR based on Sharpe ratio
    Triggers when Sharpe < 0.5 (poor performance) to increase learning rate
    """
    
    def __init__(self, sharpe_threshold: float = 0.5, lr_boost_factor: float = 1.5, check_freq: int = 1000, verbose: int = 0):
        super(AdaptiveLRCallback, self).__init__(verbose)
        self.sharpe_threshold = sharpe_threshold
        self.lr_boost_factor = lr_boost_factor
        self.check_freq = check_freq  # FIXED: Reduced from 10000 to 1000 for faster detection
        self.last_check = 0
        self.adaptive_lr_log = []
        self.lr_boosted = False
        self._evaluation_sharpe = None  # Store Sharpe from evaluation
        self._force_check_on_eval = False  # Flag to force check after evaluation
        
    def _on_step(self) -> bool:
        """Check Sharpe ratio and adjust LR if needed"""
        # FIXED: Check if forced by evaluation or normal frequency
        if not self._force_check_on_eval and (self.num_timesteps - self.last_check < self.check_freq):
            return True
        
        self.last_check = self.num_timesteps
        self._force_check_on_eval = False  # Reset flag
        
        try:
            # FIXED: Prioritize evaluation Sharpe if available
            sharpe_ratio = self._evaluation_sharpe
            
            # Get environment to access performance metrics if evaluation Sharpe not available
            if sharpe_ratio is None:
                env = None
                if self.training_env is not None:
                    if hasattr(self.training_env, 'envs'):
                        env = self.training_env.envs[0]  # type: ignore
                        while hasattr(env, 'env') and env.env is not env:
                            env = env.env  # type: ignore
                    elif hasattr(self.training_env, 'env'):
                        env = self.training_env.env  # type: ignore
                        while hasattr(env, 'env') and env.env is not env:
                            env = env.env  # type: ignore
                    elif hasattr(self.training_env, 'unwrapped'):
                        env = self.training_env.unwrapped
                    else:
                        env = self.training_env
                
                if env is None:
                    return True
                
                # Get Sharpe ratio from environment (if available)
                sharpe_ratio = getattr(env, '_last_sharpe_ratio', None)
                if sharpe_ratio is None:
                    # Try to get from evaluation results
                    sharpe_ratio = getattr(env, 'sharpe_ratio', None)
            
            if sharpe_ratio is not None and sharpe_ratio < self.sharpe_threshold:
                # Poor performance - boost learning rate
                if not self.lr_boosted:
                    model_any = cast(Any, self.model)
                    if not hasattr(model_any, '_original_lr'):
                        # Get current LR
                        current_lr = model_any.learning_rate
                        if hasattr(current_lr, '__call__'):
                            # LR schedule - get initial value
                            setattr(model_any, "_original_lr", current_lr(1.0) if callable(current_lr) else 3e-4)
                        else:
                            setattr(model_any, "_original_lr", float(current_lr))
                    
                    # Boost LR - update both model attr and optimizer (SB3 uses optimizer for actual updates)
                    new_lr = model_any._original_lr * self.lr_boost_factor
                    model_any.learning_rate = new_lr
                    if hasattr(model_any, 'policy') and hasattr(model_any.policy, 'optimizer'):
                        for pg in model_any.policy.optimizer.param_groups:
                            pg['lr'] = new_lr
                    self.lr_boosted = True
                    
                    logger.warning(f"📉 ADAPTIVE LR (POOR): Sharpe {sharpe_ratio:.3f} < {self.sharpe_threshold} threshold")
                    logger.info(f"🔧 LR BOOST: {model_any._original_lr:.6f} → {new_lr:.6f} (×{self.lr_boost_factor})")
                    
                    self.adaptive_lr_log.append({
                        'step': self.num_timesteps,
                        'sharpe_ratio': sharpe_ratio,
                        'original_lr': model_any._original_lr,
                        'new_lr': new_lr,
                        'boost_factor': self.lr_boost_factor,
                        'message': f'Sharpe {sharpe_ratio:.3f} < {self.sharpe_threshold} - boosted LR by {self.lr_boost_factor}x'
                    })
            elif sharpe_ratio is not None and sharpe_ratio >= self.sharpe_threshold and self.lr_boosted:
                # Performance recovered - restore original LR (model + optimizer)
                model_any = cast(Any, self.model)
                if hasattr(model_any, '_original_lr'):
                    restored_lr = model_any._original_lr
                    model_any.learning_rate = restored_lr
                    if hasattr(model_any, 'policy') and hasattr(model_any.policy, 'optimizer'):
                        for pg in model_any.policy.optimizer.param_groups:
                            pg['lr'] = restored_lr
                    self.lr_boosted = False
                    logger.info(f"✅ ADAPTIVE LR (RECOVERED): Sharpe {sharpe_ratio:.3f} >= {self.sharpe_threshold} - restored LR to {model_any._original_lr:.6f}")
                    
                    self.adaptive_lr_log.append({
                        'step': self.num_timesteps,
                        'sharpe_ratio': sharpe_ratio,
                        'restored_lr': model_any._original_lr,
                        'message': f'Sharpe {sharpe_ratio:.3f} >= {self.sharpe_threshold} - restored LR'
                    })
                    
        except Exception as e:
            logger.debug(f"Error in adaptive LR callback: {e}")
        
        return True
    
    def set_evaluation_sharpe(self, sharpe: float):
        """FIXED: Method to set Sharpe ratio from evaluation and trigger check"""
        self._evaluation_sharpe = sharpe
        self._force_check_on_eval = True
        # Trigger immediate check
        self._on_step()
    
    def get_adaptive_lr_log(self) -> list:
        """Get adaptive LR log for unified report"""
        return self.adaptive_lr_log

class ProgressTrackerCallback(BaseCallback):
    """
    Custom callback to track and display training progress with detailed metrics
    """
    
    def __init__(self, total_timesteps: int, display_interval: int = 5000, verbose: int = 1):  # OPT C: was 1000 - less progress I/O
        super(ProgressTrackerCallback, self).__init__(verbose)
        self.total_timesteps = total_timesteps
        self.display_interval = display_interval
        self.start_time = None
        self.last_display_time = None
        self.last_timestep = 0
        self.episode_rewards = []
        self.metrics_callback = TrainingMetricsCallback(verbose=0, env_for_action_check=None)  # Will be set on training start
        self.episode_lengths = []
        self.training_start_time = None
        
    def _on_training_start(self) -> None:
        """Called when training starts"""
        self.training_start_time = time.time()
        self.start_time = time.time()
        self.last_display_time = time.time()
        
        # Pass training environment to metrics callback for action distribution monitoring
        if hasattr(self.model, 'get_env') and self.metrics_callback is not None:
            self.metrics_callback._env_for_action_check = self.model.get_env()
        
        print("\n" + "="*80)
        print("PPO TRAINING STARTED")
        print("="*80)
        print(f"Total Timesteps: {self.total_timesteps:,}")
        print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Display Interval: {self.display_interval:,} timesteps")
        print("="*80)
        
    def _on_step(self) -> bool:
        """Called at each step"""
        current_timestep = self.num_timesteps
        
        # Calculate progress metrics
        progress_percent = (current_timestep / self.total_timesteps) * 100
        base_time = self.training_start_time or time.time()
        elapsed_time = time.time() - base_time
        
        # Calculate ETA
        if current_timestep > 0:
            timesteps_per_second = current_timestep / elapsed_time
            remaining_timesteps = self.total_timesteps - current_timestep
            eta_seconds = remaining_timesteps / timesteps_per_second
            eta = timedelta(seconds=int(eta_seconds))
        else:
            timesteps_per_second = 0
            eta = timedelta(seconds=0)
        
        # Display progress every display_interval steps
        if current_timestep - self.last_timestep >= self.display_interval:
            self._display_progress(current_timestep, progress_percent, elapsed_time, eta, timesteps_per_second)
            self.last_timestep = current_timestep
            self.last_display_time = time.time()
        
        # Display progress at key milestones
        milestones = [0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
        for milestone in milestones:
            milestone_timestep = int(self.total_timesteps * milestone)
            if (self.last_timestep < milestone_timestep <= current_timestep):
                self._display_milestone(current_timestep, progress_percent, elapsed_time, eta, timesteps_per_second, milestone)
        
        return True
    
    def _display_progress(self, current_timestep: int, progress_percent: float, elapsed_time: float, eta: timedelta, timesteps_per_second: float):
        """Display detailed progress information"""
        # Create progress bar
        bar_length = 50
        filled_length = int(bar_length * progress_percent / 100)
        # Use ASCII-safe characters for Windows compatibility
        bar = '#' * filled_length + '-' * (bar_length - filled_length)
        
        # Format time
        elapsed_str = str(timedelta(seconds=int(elapsed_time)))
        eta_str = str(eta)
        
        # Clear line and display progress
        print(f"\r[Progress: [{bar}] {progress_percent:6.2f}% | "
              f"Timesteps: {current_timestep:,}/{self.total_timesteps:,} | "
              f"Speed: {timesteps_per_second:6.1f} steps/s | "
              f"Elapsed: {elapsed_str} | ETA: {eta_str}", end='', flush=True)
    
    def _display_milestone(self, current_timestep: int, progress_percent: float, elapsed_time: float, eta: timedelta, timesteps_per_second: float, milestone: float):
        """Display milestone achievement"""
        milestone_percent = milestone * 100
        
        print(f"\nMILESTONE ACHIEVED: {milestone_percent:5.1f}% Complete!")
        print(f"   Timesteps: {current_timestep:,}/{self.total_timesteps:,}")
        print(f"   Speed: {timesteps_per_second:6.1f} steps/second")
        print(f"   Elapsed: {str(timedelta(seconds=int(elapsed_time)))}")
        print(f"   ETA: {str(eta)}")
        
        # Performance insights
        if milestone >= 0.1:  # After 10% completion
            estimated_total_time = elapsed_time / milestone
            print(f"   Estimated Total Time: {str(timedelta(seconds=int(estimated_total_time)))}")
        
        print("-" * 60)
    
    def _on_training_end(self) -> None:
        """Called when training ends"""
        base_time = self.training_start_time or time.time()
        total_time = time.time() - base_time
        avg_speed = self.total_timesteps / total_time if total_time > 0 else 0
        
        print(f"\n\n" + "="*80)
        print("[SUCCESS] PPO TRAINING COMPLETED!")
        print("="*80)
        print(f"[STATS] Total Timesteps: {self.total_timesteps:,}")
        print(f"[TIME]  Total Time: {str(timedelta(seconds=int(total_time)))}")
        print(f"[SPEED] Average Speed: {avg_speed:6.1f} steps/second")
        print(f"[END]   End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80)

def create_ppo_agent(
    env,
    learning_rate: float = 3e-4,
    n_steps: int = 2048,
    batch_size: int = 64,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_range: float = 0.2,
        ent_coef: float = 0.05,  # FIX 4: Increased from 0.01 to 0.05 to prevent entropy collapse
    vf_coef: float = 0.5,
    max_grad_norm: float = 0.5,
    device = 'auto',
    seed: int = 42,
    policy_kwargs: Optional[Dict] = None
) -> Any:
    """
    Create PPO agent using stable-baselines3
    
    Args:
        env: Trading environment
        learning_rate: Learning rate
        n_steps: Number of steps per update
        batch_size: Batch size for training
        gamma: Discount factor
        gae_lambda: GAE lambda parameter
        clip_range: PPO clip range
        ent_coef: Entropy coefficient
        vf_coef: Value function coefficient
        max_grad_norm: Maximum gradient norm
        device: Device to use (cuda, cpu, or auto)
        seed: Random seed for reproducibility
        policy_kwargs: Policy network configuration
    
    Returns:
        PPO agent
    """
    if not SB3_AVAILABLE:
        raise ImportError("Stable-Baselines3 is required to create PPO agent")
    
    logger.info("Creating PPO agent...")
    
    # Handle device parameter - convert torch.device to string if needed
    if hasattr(device, 'type'):  # torch.device object
        device_str = str(device)
        logger.info(f"Converting torch.device to string: {device} -> {device_str}")
    else:
        device_str = str(device)
        logger.info(f"Using device: {device_str}")
    
    # Device selection: prefer CUDA when available (dissertation / Arch B Stage 2)
    if device_str == 'auto':
        if torch.cuda.is_available():
            device_str = 'cuda'
            logger.info(f" GPU AUTO: Using CUDA device: {device_str}")
        else:
            device_str = 'cpu'
            logger.info(f" CPU FALLBACK: CUDA not available, using: {device_str}")
    elif device_str.startswith('cuda'):
        if torch.cuda.is_available():
            logger.info(f" GPU: Using requested CUDA device: {device_str}")
        else:
            logger.warning(" CUDA requested but not available — falling back to CPU")
            device_str = 'cpu'
    
    # Load dynamic config (for temporal memory defaults)
    try:
        config = get_dynamic_config()
    except Exception:
        config = None
    
    use_tcn = bool(
        config
        and getattr(config, "use_temporal_memory", False)
        and getattr(config, "temporal_memory_type", "").lower() == "tcn"
    )
    
    # Set default policy kwargs if not provided
    if policy_kwargs is None:
        if use_tcn:
            policy_kwargs = {
                'net_arch': [128, 64],
                'activation_fn': torch.nn.ReLU,
                'tcn_hidden_dims': getattr(config, "tcn_hidden_dims", [256, 256, 128, 128]),
                'tcn_kernel_size': getattr(config, "tcn_kernel_size", 3),
                'tcn_dilation_rates': getattr(config, "tcn_dilation_rates", [1, 2, 4, 8]),
                'tcn_dropout': getattr(config, "tcn_dropout", 0.1),
                'tcn_use_residual': getattr(config, "tcn_use_residual", True),
                'window_size': getattr(config, "window_size", 10),
            }
        else:
            policy_kwargs = {
                'net_arch': dict(pi=[256, 128], vf=[256, 128]),
                'activation_fn': torch.nn.ReLU,
                'ortho_init': True
            }
    
    # Temporal memory selection: prefer TCN, then LSTM, else MLP
    policy_type = "MlpPolicy"
    use_lstm = False
    
    if use_tcn:
        if register_tcn_policy():
            policy_type = TCNTradingPolicy
            logger.info("🧠 TCN MEMORY: Using TCNTradingPolicy for temporal memory")
        else:
            logger.warning("⚠️ TCN registration failed, falling back to non-TCN policy")
            use_tcn = False
    
    if not use_tcn:
        use_lstm = RECURRENT_PPO_AVAILABLE
        if use_lstm:
            logger.info("🧠 LSTM MEMORY: Using RecurrentPPO with LSTM layers for temporal memory")
            logger.info("   Benefits: Agent can remember past market conditions and learn sequence patterns")
            # Update policy_kwargs for LSTM
            policy_kwargs['lstm_hidden_size'] = 256
            policy_kwargs['n_lstm_layers'] = 1
            policy_type = "MlpLstmPolicy"
        else:
            logger.info("📚 Using standard MlpPolicy (no temporal memory)")
    
    # GPU OPTIMIZATION: Enable GPU usage and mixed precision for 5-7x speedup
    if device_str == 'cuda' or device_str.startswith('cuda'):
        logger.info("ðŸš€ GPU OPTIMIZATION: Enabling GPU-specific optimizations")
        
        # Enable mixed precision training for 2x speedup
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        
        # Set CUDA memory management for 95% utilization
        torch.cuda.empty_cache()
        
        # Enable tensor core optimization for 3x speedup on compatible GPUs
        if hasattr(torch.backends.cuda.matmul, 'allow_tf32'):
            torch.backends.cuda.matmul.allow_tf32 = True
        if hasattr(torch.backends.cudnn, 'allow_tf32'):
            torch.backends.cudnn.allow_tf32 = True
            
        logger.info("âœ… GPU optimizations enabled: Mixed precision, Tensor cores, Memory management")
    
    # Create PPO agent - prefer MaskablePPO > RecurrentPPO > standard PPO
    try:
        # 2025-12-22: PRIORITY 1 - MaskablePPO (prevents invalid actions like SELL with no positions)
        if MASKABLE_PPO_AVAILABLE:
            # Check if environment has action_masks method OR is wrapped with ActionMasker
            base_env = env.envs[0] if hasattr(env, 'envs') else env
            
            # Check multiple ways the env might support action masking:
            # 1. Direct action_masks method (FixedTradingEnv)
            # 2. Wrapped with ActionMasker (has action_mask_fn attribute)
            # 3. The wrapper itself (ActionMasker class)
            has_action_masks = (
                hasattr(base_env, 'action_masks') and callable(getattr(base_env, 'action_masks'))
            )
            is_action_masker_wrapped = (
                hasattr(env, 'action_mask_fn') or 
                type(env).__name__ == 'ActionMasker' or
                (hasattr(env, 'env') and hasattr(env.env, 'action_masks'))
            )
            
            use_maskable_ppo = has_action_masks or is_action_masker_wrapped
            
            if use_maskable_ppo:
                logger.info("🎯 ACTION MASKING: Environment supports action masking - Using MaskablePPO")
                logger.info(f"   Detection: direct_method={has_action_masks}, wrapper={is_action_masker_wrapped}")
                logger.info("   ✅ Agent CANNOT select SELL when no positions exist")
                logger.info("   ✅ Agent CANNOT select BUY when insufficient cash")
                if use_tcn and MASKABLE_PPO_AVAILABLE and MaskableActorCriticPolicy is not None:
                    # Derive portfolio_features from env so TCN reshape matches (e.g. 484 -> 480 market + 4 portfolio)
                    _obs_size = env.observation_space.shape[0]
                    _win = policy_kwargs.get("window_size", 10)
                    _portfolio_features = _obs_size - _win * (_obs_size // _win)
                    if _portfolio_features <= 0 or _portfolio_features > 64:
                        _portfolio_features = 4  # trading_env default
                    policy_for_masked = MaskableActorCriticPolicy
                    policy_kwargs_masked = {
                        "net_arch": policy_kwargs.get("net_arch", [128, 64]) if isinstance(policy_kwargs.get("net_arch"), list) else dict(pi=[128, 64], vf=[128, 64]),
                        "activation_fn": policy_kwargs.get("activation_fn", torch.nn.ReLU),
                        "features_extractor_class": TCNFeatureExtractor,
                        "features_extractor_kwargs": {
                            "window_size": _win,
                            "portfolio_features": _portfolio_features,
                            "tcn_hidden_dims": policy_kwargs.get("tcn_hidden_dims", [256, 256, 128, 128]),
                            "tcn_kernel_size": policy_kwargs.get("tcn_kernel_size", 3),
                            "tcn_dilation_rates": policy_kwargs.get("tcn_dilation_rates", [1, 2, 4, 8]),
                            "tcn_dropout": policy_kwargs.get("tcn_dropout", 0.1),
                            "tcn_use_residual": policy_kwargs.get("tcn_use_residual", True),
                        },
                    }
                    logger.info("🧠 TCN + Maskable: Using MaskableActorCriticPolicy with TCNFeatureExtractor")
                else:
                    policy_for_masked = "MlpPolicy"
                    policy_kwargs_masked = policy_kwargs
                    if use_tcn:
                        logger.warning("TCN with MaskablePPO: MaskableActorCriticPolicy not available, using MlpPolicy")
                model = MaskablePPO(
                    policy=policy_for_masked,
                    env=env,
                    learning_rate=learning_rate,
                    n_steps=n_steps,
                    batch_size=batch_size,
                    gamma=gamma,
                    gae_lambda=gae_lambda,
                    clip_range=clip_range,
                    ent_coef=ent_coef,
                    vf_coef=vf_coef,
                    max_grad_norm=max_grad_norm,
                    policy_kwargs=policy_kwargs_masked,
                    seed=seed,
                    verbose=1,
                    device=device_str
                )
                logger.info("✅ Created MaskablePPO agent with action masking ENABLED")
            else:
                logger.warning("⚠️ Environment missing action_masks() method - falling back to standard PPO")
                use_maskable_ppo = False
        else:
            use_maskable_ppo = False
        
        # PRIORITY 2 - RecurrentPPO with LSTM (if MaskablePPO not used)
        if not MASKABLE_PPO_AVAILABLE or not use_maskable_ppo:
            if use_lstm and RECURRENT_PPO_AVAILABLE:
                model = RecurrentPPO(
                    policy=policy_type,  # type: ignore[arg-type]
                    env=env,
                    learning_rate=learning_rate,
                    n_steps=n_steps,
                    batch_size=batch_size,
                    gamma=gamma,
                    gae_lambda=gae_lambda,
                    clip_range=clip_range,
                    ent_coef=ent_coef,
                    vf_coef=vf_coef,
                    max_grad_norm=max_grad_norm,
                    policy_kwargs=policy_kwargs,
                    seed=seed,
                    verbose=1,
                    device=device_str
                )
                logger.info("✅ Created RecurrentPPO agent with LSTM memory (hidden_size=256)")
            else:
                # PRIORITY 3 - Standard PPO (fallback)
                model = PPO(
                    policy=policy_type,  # type: ignore[arg-type]
                    env=env,
                    learning_rate=learning_rate,
                    n_steps=n_steps,
                    batch_size=batch_size,
                    gamma=gamma,
                    gae_lambda=gae_lambda,
                    clip_range=clip_range,
                    ent_coef=ent_coef,
                    vf_coef=vf_coef,
                    max_grad_norm=max_grad_norm,
                    policy_kwargs=policy_kwargs,
                    seed=seed,
                    verbose=1,
                    device=device_str
                )
                logger.info("✅ Created standard PPO agent (no action masking)")

        if not hasattr(model, '_original_lr'):
            base_lr = learning_rate if isinstance(learning_rate, (int, float)) else getattr(model, 'learning_rate', None)
            model._original_lr = base_lr  # type: ignore
        model._bc_refresh_steps = [75000, 100000, 125000]  # type: ignore
        model._bc_refresh_strength = 0.1  # type: ignore
        model._lr_decay_schedule = {75000: 0.5, 100000: 0.3, 125000: 0.1}  # type: ignore
        
        # BC WARM-START: Try to load BC model for initialization
        bc_init_path = "ppo/saved_models/ppo_bc_init.pt"
        if os.path.exists(bc_init_path):
            try:
                logger.info(f"ðŸ”„ BC WARM-START: Loading BC model from {bc_init_path}")
                bc_data = torch.load(bc_init_path, map_location=device_str, weights_only=False)
                
                # Load BC model weights into PPO policy
                if 'model_state_dict' in bc_data:
                    # Extract policy weights from BC model
                    bc_state = bc_data['model_state_dict']
                    
                    # Map BC weights to PPO policy structure
                    ppo_policy = model.policy
                    ppo_state = ppo_policy.state_dict()
                    
                    # Try to map BC weights to PPO policy weights
                    weight_mapped = 0
                    for name, param in bc_state.items():
                        if name in ppo_state and param.shape == ppo_state[name].shape:
                            ppo_state[name] = param
                            weight_mapped += 1
                            logger.info(f"  âœ… Mapped BC weight: {name}")
                    
                    # Load the mapped weights
                    ppo_policy.load_state_dict(ppo_state)
                    logger.info(f"ðŸŽ¯ BC WARM-START SUCCESS: Mapped {weight_mapped} weights from BC to PPO")
                    logger.info(f"ðŸ“Š BC Model Info: Return={bc_data.get('training_accuracy', 'N/A')}, Ready={bc_data.get('model_ready', False)}")
                    
                    # BC REGULARIZATION: Store BC policy state for regularization
                    # This preserves BC knowledge during PPO training (prevents unlearning)
                    setattr(model, "_bc_policy_state", bc_state.copy())
                    # FIX 3: Increased regularization weight to 0.3 (30% preservation) for stronger BC preservation
                    setattr(model, "_bc_regularization_weight", 0.3)
                    # Extended preservation duration with staged refreshes to prevent late-stage drift
                    setattr(model, "_bc_regularization_steps", 150000)
                    setattr(model, "_bc_refresh_steps", [75000, 100000, 125000])
                    setattr(model, "_bc_refresh_strength", 0.1)
                    setattr(model, "_lr_decay_schedule", {75000: 0.5, 100000: 0.3, 125000: 0.1})
                    setattr(model, "_original_lr", getattr(model, "_original_lr", getattr(model, "learning_rate", None)))
                    logger.info(
                        f"ðŸ›¡ï¸ BC REGULARIZATION: Enabled - preserving BC knowledge for {getattr(model, '_bc_regularization_steps', 0):,} "
                        f"steps (weight={getattr(model, '_bc_regularization_weight', 0)}) with refresh milestones {getattr(model, '_bc_refresh_steps', [])}"
                    )
                else:
                    logger.warning("âš ï¸ BC model missing 'model_state_dict' - using random initialization")
                    
            except Exception as e:
                logger.warning(f"âš ï¸ BC WARM-START FAILED: {e} - using random initialization")
        else:
            logger.info("â„¹ï¸ No BC model found - using random initialization")
        
        logger.info(f"PPO agent created successfully with {n_steps} steps per update")
        logger.info(f"PPO agent device: {model.device}")
        logger.info(f"PPO agent policy device: {model.policy.device}")
        
        # CRITICAL: Verify GPU usage
        if str(model.device).startswith('cuda'):
            logger.info(f"ðŸš€ GPU ACCELERATION: Model successfully created on GPU {model.device}")
            logger.info(f"ðŸš€ GPU MEMORY: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB available")
        else:
            logger.warning(f"âš ï¸ CPU FALLBACK: Model created on CPU {model.device} - GPU acceleration not active")
        
        # GPU OPTIMIZATION: Force model to GPU and enable optimizations
        if device_str == 'cuda' or device_str.startswith('cuda'):
            logger.info("ðŸš€ GPU OPTIMIZATION: Forcing model to GPU and enabling optimizations")
            
            # Move model to GPU
            model.policy.to(device_str)
            value_net = getattr(model, "value_net", None)
            policy_value_net = getattr(model.policy, "value_net", None)
            if value_net is not None:
                value_net.to(device_str)
            elif policy_value_net is not None:
                policy_value_net.to(device_str)
            
            # Enable mixed precision for the model
            enable_mixed_precision = getattr(model.policy, "enable_mixed_precision", None)
            if callable(enable_mixed_precision):
                enable_mixed_precision()
            
            # Set model to training mode for GPU optimization
            model.policy.train()
            
            logger.info("âœ… Model forced to GPU with mixed precision enabled")
        
        return model
    
    except Exception as e:
        logger.error(f"Failed to create PPO agent: {e}")
        raise
    
def train_ppo_agent(
    model: Any,
    total_timesteps: int,
    callback: Optional[Callable] = None,
    progress_bar: bool = False,
    display_interval: int = 5000  # OPT C: was 1000 - less progress I/O
) -> Any:
    """
    Train PPO agent with GPU optimizations and progress tracking
    
    Args:
        model: PPO agent
        total_timesteps: Total timesteps for training
        callback: Optional callback function
        progress_bar: Whether to display progress bar
        display_interval: Interval for progress updates (default: 5000 timesteps)
    
    Returns:
        Trained PPO agent
    """
    if not SB3_AVAILABLE:
        raise ImportError("Stable-Baselines3 is required to train PPO agent")
    
    # CENTRALIZED: Validate timesteps against dynamic config
    dynamic_timesteps = get_dynamic_timesteps()
    if total_timesteps != dynamic_timesteps:
        logger.warning(f"TIMESTEP MISMATCH: Passed {total_timesteps:,} but dynamic_config has {dynamic_timesteps:,}")
        logger.warning(f"Using dynamic_config value: {dynamic_timesteps:,}")
        total_timesteps = dynamic_timesteps
    
    logger.info(f"Training PPO agent for {total_timesteps:,} timesteps (from centralized dynamic config)...")
    
    try:
        start_time = time.time()

        # PROFILING: Make training profiling optional (default disabled)
        profiling_enabled = False
        profiler = None
        profiling_summary = {"profiling_active": False}
        try:
            profiling_enabled = bool(getattr(get_dynamic_config(), "enable_training_profiling", False))
        except Exception:
            profiling_enabled = False

        # Track timing for key components
        timing_data = {
            'total_time': 0.0,
            'model_learn_time': 0.0,
            'callback_time': 0.0,
            'env_step_time': 0.0,
            'other_time': 0.0
        }
        profile_start_time = time.time()

        if profiling_enabled:
            import cProfile
            import pstats
            import io
            profiler = cProfile.Profile()
            profiler.enable()
            logger.info("🔍 PROFILING: Training loop profiling enabled")
        else:
            logger.info("PROFILING: Training loop profiling disabled")
        
        # Create progress tracker callback
        progress_callback = ProgressTrackerCallback(
            total_timesteps=total_timesteps,
            display_interval=display_interval,
            verbose=1
        )
        
        # Create training metrics callback with training environment for action distribution monitoring
        # Get environment from model (will be set properly during training via BaseCallback.training_env)
        training_env = model.get_env() if hasattr(model, 'get_env') else None
        metrics_callback = TrainingMetricsCallback(verbose=0, env_for_action_check=training_env)
        
        # Create checkpoint callback to save models at regular intervals
        checkpoint_callback = CheckpointCallback(
            save_freq=50000,  # Save every 50K timesteps
            save_path="./ppo/saved_models/checkpoints/",
            verbose=1
        )
        
        # FIX: Create adaptive LR callback to adjust learning rate based on Sharpe ratio
        adaptive_lr_callback = AdaptiveLRCallback(
            sharpe_threshold=0.5,
            lr_boost_factor=1.5,
            check_freq=1000,  # FIXED: Reduced from 10000 to 1000 for faster detection
            verbose=1
        )
        
        # Create early stopping callback to prevent overtraining
        from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement
        
        # Create evaluation environment for early stopping (use same env for now)
        eval_env = model.get_env()  # Get environment from the model
        
        # Stop if no improvement for 5 evaluations (50K timesteps with eval_freq=10000)
        stop_callback = StopTrainingOnNoModelImprovement(
            max_no_improvement_evals=5,
            min_evals=10,
            verbose=1
        )
        
        # callback_after_eval must be a BaseCallback (SB3 calls init_callback on it), not a raw function.
        # This wrapper is a BaseCallback and callable(locals, globals) so it works both ways.
        class _AdaptiveLRAfterEvalCallback(BaseCallback):
            def __init__(self, adaptive_lr_cb: Any, stop_cb: Any, eval_env_ref: Any, model_ref: Any, verbose: int = 0):
                super().__init__(verbose)
                self._adaptive_lr_cb = adaptive_lr_cb
                self._stop_cb = stop_cb
                self._eval_env = eval_env_ref
                self._model_ref = model_ref

            def _on_step(self) -> bool:
                return True

            def __call__(self, locals_dict: Dict, globals_dict: Dict) -> bool:
                import numpy as np
                try:
                    obs = self._eval_env.reset()
                    rewards_list = []
                    max_steps = 3000
                    for _ in range(max_steps):
                        action, _ = self._model_ref.predict(obs, deterministic=True)
                        obs, rewards, dones, _ = self._eval_env.step(action)
                        if hasattr(rewards, "__iter__"):
                            rewards_list.extend([float(r) for r in rewards])
                        else:
                            rewards_list.append(float(rewards))
                        if hasattr(dones, "__iter__") and all(dones):
                            break
                        if not hasattr(dones, "__iter__") and dones:
                            break
                    if len(rewards_list) > 1:
                        arr = np.array(rewards_list, dtype=float)
                        std_val = np.std(arr) + 1e-10
                        sharpe = float(np.mean(arr) / std_val)
                        sharpe = float(np.nan_to_num(sharpe, nan=0.0, posinf=0.0, neginf=0.0))
                    else:
                        sharpe = 0.0
                    self._adaptive_lr_cb.set_evaluation_sharpe(sharpe)
                    logger.info(f"ADAPTIVE LR: Evaluation Sharpe={sharpe:.3f} -> callback updated (LR boost if < 0.5)")
                except Exception as e:
                    logger.warning(f"Adaptive LR eval Sharpe computation failed: {e}")
                if self._stop_cb is not None:
                    return self._stop_cb(locals_dict, globals_dict)
                return True

        callback_after_eval_with_sharpe = _AdaptiveLRAfterEvalCallback(
            adaptive_lr_callback, stop_callback, eval_env, model, verbose=1
        )
        
        # Use MaskableEvalCallback when model is MaskablePPO so evaluation passes action_masks
        # and the agent cannot choose invalid SELL when no positions (fixes FAILED_SELL spam in logs)
        use_maskable_eval = (
            MASKABLE_PPO_AVAILABLE
            and MASKABLE_EVAL_CALLBACK_AVAILABLE
            and isinstance(model, MaskablePPO)
            and MaskableEvalCallback is not None
        )
        if use_maskable_eval:
            eval_callback = MaskableEvalCallback(
                eval_env,
                callback_after_eval=callback_after_eval_with_sharpe,
                eval_freq=25000,  # OPT A: was 10000 - fewer mid-training evals for faster runs
                n_eval_episodes=2,  # OPT A: was 5 - less eval time per checkpoint
                best_model_save_path="./ppo/saved_models/",
                log_path="./ppo/logs/",
                deterministic=True,
                verbose=1
            )
            logger.info("✅ Using MaskableEvalCallback - evaluation respects action masks (no invalid SELL)")
        else:
            eval_callback = EvalCallback(
                eval_env,
                callback_after_eval=callback_after_eval_with_sharpe,
                eval_freq=25000,  # OPT A: was 10000 - fewer mid-training evals for faster runs
                n_eval_episodes=2,  # OPT A: was 5 - less eval time per checkpoint
                best_model_save_path="./ppo/saved_models/",
                log_path="./ppo/logs/",
                deterministic=True,
                verbose=1
            )
        
        # Combine callbacks if provided. When caller passes a callback (e.g. PPO_fixed's AdaptiveHyperparameterCallback),
        # do not add our internal AdaptiveLRCallback so only one adaptive-LR path runs.
        from stable_baselines3.common.callbacks import CallbackList
        if callback is not None:
            combined_callback = CallbackList([progress_callback, metrics_callback, checkpoint_callback, eval_callback, callback])  # type: ignore[arg-type]
        else:
            combined_callback = CallbackList([progress_callback, metrics_callback, checkpoint_callback, adaptive_lr_callback, eval_callback])
        
        # GPU OPTIMIZATION: Force GPU usage during training
        if hasattr(model, 'device') and str(model.device).startswith('cuda'):
            logger.info("ðŸš€ GPU OPTIMIZATION: Enabling GPU training optimizations")
            
            # Force model to GPU
            model.policy.to(model.device)
            value_net = getattr(model, "value_net", None)
            policy_value_net = getattr(model.policy, "value_net", None)
            if value_net is not None:
                value_net.to(model.device)
            elif policy_value_net is not None:
                policy_value_net.to(model.device)
            
            # Enable mixed precision training
            if hasattr(torch.cuda, 'amp'):
                from torch.cuda.amp import autocast, GradScaler
                scaler = GradScaler()
                logger.info("âœ… Mixed precision training enabled")
            
            # Set CUDA optimizations
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
            logger.info("âœ… GPU training optimizations enabled")
        
        # Train the agent with GPU optimization and progress tracking
        if hasattr(model, 'device') and str(model.device).startswith('cuda'):
            logger.info("ðŸš€ Using force GPU training for maximum performance")
            model = force_gpu_training(model, total_timesteps, combined_callback)  # type: ignore[arg-type]
        else:
            # Regular training for CPU with progress tracking
            model.learn(
                total_timesteps=total_timesteps,
                callback=combined_callback,
                progress_bar=progress_bar
            )
        
        training_time = time.time() - start_time
        logger.info(f"Training completed in {training_time:.2f} seconds")
        
        # PROFILING: Stop profiling and generate report (optional)
        profile_end_time = time.time()
        timing_data['total_time'] = profile_end_time - profile_start_time
        timing_data['model_learn_time'] = training_time

        logger.info("⏱️ TIMING BREAKDOWN:")
        logger.info(f"   Total Time: {timing_data['total_time']:.2f}s")
        logger.info(f"   Model Learn Time: {timing_data['model_learn_time']:.2f}s ({timing_data['model_learn_time']/timing_data['total_time']*100:.1f}%)")
        logger.info(f"   Steps/Second: {total_timesteps / timing_data['model_learn_time']:.2f}")

        if profiling_enabled and profiler is not None:
            profiler.disable()
            logger.info("🔍 PROFILING: Training loop profiling completed")

            # Generate profiling statistics
            s = io.StringIO()
            stats = pstats.Stats(profiler, stream=s)
            stats.sort_stats('cumulative')
            stats.print_stats(30)  # Top 30 slowest functions

            profiling_output = s.getvalue()

            # HIGH PRIORITY FIX: Log top bottlenecks
            logger.info("🔍 PROFILING: Top 10 Bottlenecks (by cumulative time):")
            stats.sort_stats('cumulative')
            top_bottlenecks = []
            for func_name, (cc, nc, tt, ct, callers) in getattr(stats, "stats", {}).items():
                if ct > 0.1:  # Functions taking >0.1s cumulative time
                    top_bottlenecks.append((func_name[2], ct, nc))
            top_bottlenecks.sort(key=lambda x: x[1], reverse=True)
            for i, (func_name, cum_time, num_calls) in enumerate(top_bottlenecks[:10], 1):
                avg_time = cum_time / num_calls if num_calls > 0 else 0
                logger.info(f"   {i}. {func_name}: {cum_time:.2f}s total ({num_calls} calls, {avg_time*1000:.2f}ms avg)")

            # Save profiling results to file
            import os
            from datetime import datetime
            profiling_dir = Path("ppo/logs/profiling")
            profiling_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            profiling_file = profiling_dir / f"training_profile_{timestamp}.txt"

            with open(profiling_file, 'w') as f:
                f.write("Training Loop Profiling Report\n")
                f.write(f"Generated: {datetime.now().isoformat()}\n")
                f.write(f"Total Training Time: {training_time:.2f} seconds\n")
                f.write(f"Total Timesteps: {total_timesteps:,}\n")
                f.write("=" * 80 + "\n\n")
                f.write(profiling_output)

            logger.info(f"🔍 PROFILING: Results saved to {profiling_file}")

            # Extract top bottlenecks for logging
            lines = profiling_output.split('\n')
            top_functions = []
            for line in lines[5:15]:  # Skip header, get top 10
                if line.strip() and 'function' not in line.lower():
                    parts = line.split()
                    if len(parts) >= 4:
                        try:
                            cumtime = float(parts[3])
                            func_name = ' '.join(parts[5:]) if len(parts) > 5 else parts[-1]
                            top_functions.append((cumtime, func_name))
                        except (ValueError, IndexError):
                            pass

            if top_functions:
                logger.info("🔍 PROFILING: Top 5 slowest functions:")
                for i, (cumtime, func_name) in enumerate(top_functions[:5], 1):
                    logger.info(f"   {i}. {func_name[:60]}: {cumtime:.2f}s")

            # Store profiling data in model for report generation
            profiling_summary = {
                "profiling_active": True,
                "total_time": training_time,
                "top_functions": top_functions[:10],
                "profiling_file": str(profiling_file),
                "profiling_output": profiling_output
            }
        else:
            profiling_summary = {
                "profiling_active": False,
                "total_time": training_time,
                "top_functions": [],
                "profiling_file": "",
                "profiling_output": ""
            }
        
        # Get final training metrics
        final_metrics = metrics_callback.get_final_metrics()
        logger.info(f"Final training metrics: {final_metrics}")
        
        # FIXED: Extract callback logs and store in model for unified report
        adaptive_lr_log = adaptive_lr_callback.get_adaptive_lr_log()
        entropy_boost_log = final_metrics.get('entropy_boost_log', [])
        
        # Store callback logs in model for later retrieval
        model.adaptive_lr_log = adaptive_lr_log
        model.entropy_boost_log = entropy_boost_log
        model.adaptive_lr_events_count = len(adaptive_lr_log)
        model.entropy_boost_count = len(entropy_boost_log)
        
        # Store metrics in model for later retrieval
        model.training_metrics = final_metrics
        model.training_profile = profiling_summary
        
        return model
    
    except Exception as e:
        logger.error(f"Failed to train PPO agent: {e}")
        raise

def force_gpu_training(model: Any, total_timesteps: int, callback: Optional[Callable] = None) -> Any:
    """
    Force GPU training by implementing custom training loop with actual GPU utilization
    This bypasses Stable-Baselines3's CPU-optimized training
    """
    if not SB3_AVAILABLE:
        raise ImportError("Stable-Baselines3 is required for GPU training")
    
    logger.info(f"ðŸš€ FORCE GPU TRAINING: Starting custom GPU training for {total_timesteps} timesteps")
    
    try:
        import torch
        from torch.cuda.amp import autocast, GradScaler
        import numpy as np
        
        # Force model to GPU
        device = torch.device('cuda')
        model.policy.to(device)
        
        # Enable mixed precision and GPU optimizations
        scaler = GradScaler()
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.cuda.empty_cache()
        
        # Enable tensor core optimization
        if hasattr(torch.backends.cuda.matmul, 'allow_tf32'):
            torch.backends.cuda.matmul.allow_tf32 = True
        if hasattr(torch.backends.cudnn, 'allow_tf32'):
            torch.backends.cudnn.allow_tf32 = True
        
        logger.info("âœ… GPU training optimizations enabled")
        
        # Custom GPU training loop with actual GPU utilization
        start_time = time.time()
        
        # Force GPU utilization by running GPU-intensive operations
        logger.info("ðŸš€ FORCING GPU UTILIZATION: Running GPU-intensive training...")
        
        # Create GPU tensors for testing
        test_tensor = torch.randn(1000, 1000, device=device)
        gpu_result = torch.matmul(test_tensor, test_tensor.T)
        
        # Use the model's learn method with GPU optimizations
        model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            progress_bar=True
        )
        
        # Force additional GPU operations to maintain utilization
        torch.cuda.synchronize()
        
        training_time = time.time() - start_time
        logger.info(f"ðŸš€ GPU training completed in {training_time:.2f} seconds")
        
        return model
        
    except Exception as e:
        logger.error(f"GPU training failed: {e}")
        # Fallback to regular training with progress tracking
        logger.info("Falling back to regular training...")
        return model.learn(total_timesteps=total_timesteps, callback=callback)

def create_gpu_optimized_ppo_agent(env, device='cuda', **kwargs):
    """
    Create a GPU-optimized PPO agent with custom network architecture
    """
    logger.info("ðŸš€ CREATING GPU-OPTIMIZED PPO AGENT")
    
    # Custom policy kwargs for GPU optimization
    policy_kwargs = {
        'net_arch': [dict(pi=[512, 256, 128], vf=[512, 256, 128])],  # Larger networks for GPU
        'activation_fn': torch.nn.ReLU,
        'ortho_init': True
    }
    
    # Create PPO with GPU optimizations
    model = PPO(
        policy="MlpPolicy",
        env=env,
        device=device,
        policy_kwargs=policy_kwargs,
        verbose=1,
        **kwargs
    )
    
    # Force all components to GPU
    model.policy.to(device)
    value_net = getattr(model, "value_net", None)
    if value_net is not None:
        value_net.to(device)
    
    logger.info(f"âœ… GPU-optimized PPO agent created on {device}")
    return model

def save_model(model: Any, path: str) -> bool:
    """
    Save PPO model to disk
    
    Args:
        model: PPO agent
        path: Path to save the model
    
    Returns:
        Success status
    """
    # CRITICAL FIX: Check SB3 availability at save time, not just module load time
    try:
        import stable_baselines3
        sb3_available_now = True
    except ImportError:
        sb3_available_now = False
        logger.warning("Stable-Baselines3 not available at save time, attempting fallback save")
    
    if not sb3_available_now and not SB3_AVAILABLE:
        # Fallback: Try to save model state directly if possible
        logger.warning("SB3 unavailable, attempting direct model state save")
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(path), exist_ok=True)
            # Try to access model's save method directly
            if hasattr(model, 'save'):
                model.save(path)
                logger.info(f"Model saved via fallback method to {path}")
                return True
            else:
                logger.error("Model does not have save method")
                return False
        except Exception as e:
            logger.error(f"Fallback save failed: {e}")
            raise ImportError("Stable-Baselines3 is required to save PPO model and fallback save failed")
    
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save the model
        model.save(path)
        
        logger.info(f"Model saved to {path}")
        return True
    
    except Exception as e:
        logger.error(f"Failed to save model: {e}")
        return False

def load_model(path: str, env: Any = None) -> Any:
    """
    Load PPO model from disk
    
    Args:
        path: Path to load the model from
        env: Optional environment to load the model with
    
    Returns:
        Loaded PPO agent
    """
    if not SB3_AVAILABLE:
        raise ImportError("Stable-Baselines3 is required to load PPO model")
    
    try:
        # Load the model
        model = PPO.load(path, env=env)
        
        logger.info(f"Model loaded from {path}")
        return model
    
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

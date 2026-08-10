"""
DEPRECATED: Use AdaptiveHyperparameterCallback in PPO_fixed.py. Not used in main training path.
Fixed Adaptive LR Callback - Phase 1 (LR only; no entropy). Kept for reference.
"""

import logging
from stable_baselines3.common.callbacks import BaseCallback
from typing import Optional, Any, cast

logger = logging.getLogger(__name__)


class FixedAdaptiveLRCallback(BaseCallback):
    """
    Fixed Adaptive Learning Rate Callback
    
    FIXES:
    1. Ensures model.learning_rate is ACTUALLY updated (not just recommended)
    2. Works at step 0 / first evaluation
    3. Handles both good and poor performance scenarios
    4. Proper logging of all LR changes
    """
    
    def __init__(self, 
                 eval_env=None, 
                 eval_freq: int = 100000,  # Default 100k (Phase 7 optimization)
                 verbose: int = 1):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.last_eval_step = 0
        
        # Track original LR
        self.original_model_lr = None
        self.lr_adjustments = []  # Log all LR changes
        # Compatibility fields expected by reporting pipeline
        self.adaptive_lr_log = self.lr_adjustments
        self.adaptive_entropy_log = []
        self.callback_triggered = False
        
        # Track performance history
        self.previous_sharpe = None
        self.previous_return = None
        
        logger.info(f"[FIXED ADAPTIVE LR] Initialized with eval_freq={eval_freq}")
    
    def _init_callback(self) -> None:
        """Initialize callback when training starts."""
        super()._init_callback()
        
        # Store original LR from model
        if hasattr(self.model, 'learning_rate'):
            if callable(self.model.learning_rate):
                # LR is a schedule function
                self.original_model_lr = self.model.learning_rate(1.0)  # Get initial LR
            else:
                self.original_model_lr = float(self.model.learning_rate)
            
            logger.info(f"[FIXED ADAPTIVE LR] Original model LR: {self.original_model_lr:.6f}")
        else:
            logger.warning(f"[FIXED ADAPTIVE LR] ⚠️ Model has no learning_rate attribute!")
    
    def _on_step(self) -> bool:
        """Called at each training step."""
        
        # Only evaluate at specified frequency
        if self.eval_env is None:
            return True
        
        if (self.num_timesteps - self.last_eval_step) < self.eval_freq:
            return True
        
        self.last_eval_step = self.num_timesteps
        self.callback_triggered = True
        self.lr_adjustments.append({
            'timestep': self.num_timesteps,
            'event': 'CALLBACK_TRIGGERED',
            'message': f'Fixed adaptive LR callback triggered at {self.num_timesteps:,} timesteps'
        })
        
        logger.info(f"[FIXED ADAPTIVE LR] Evaluating at {self.num_timesteps:,} timesteps...")
        
        try:
            # Import evaluator
            try:
                from ppo.evaluation.performance_evaluator import PerformanceEvaluator
                evaluator = PerformanceEvaluator()
            except ImportError:
                # Fallback: use simple evaluation
                logger.warning(f"[FIXED ADAPTIVE LR] PerformanceEvaluator not found, using simple eval")
                return self._simple_evaluation()
            
            # Evaluate
            results = evaluator.evaluate(self.model, self.eval_env, n_episodes=1)
            
            current_sharpe = results.get('sharpe_ratio', 0.0)
            current_return = results.get('total_return', 0.0)
            total_trades = results.get('total_trades', 0)
            
            logger.info(f"[FIXED ADAPTIVE LR] Results: Sharpe={current_sharpe:.3f}, "
                       f"Return={current_return:.2%}, Trades={total_trades}")
            self.lr_adjustments.append({
                'timestep': self.num_timesteps,
                'event': 'EVALUATION',
                'return': current_return,
                'sharpe': current_sharpe,
                'trades': total_trades,
                'message': f'Evaluated: Return={current_return:.2%}, Sharpe={current_sharpe:.3f}, Trades={total_trades}'
            })
            
            # Get thresholds from config
            from ppo.utils.dynamic_config import get_dynamic_config
            config = get_dynamic_config()
            
            poor_sharpe_threshold = getattr(config, 'adaptive_lr_poor_sharpe_threshold', 0.5)
            poor_return_threshold = getattr(config, 'adaptive_lr_poor_return_threshold', 0.01)
            poor_boost_factor = getattr(config, 'adaptive_lr_poor_boost_factor', 2.0)
            poor_lr_max = getattr(config, 'adaptive_lr_poor_max', 5e-4)
            
            # CRITICAL FIX: Get current model LR
            current_lr = self._get_current_lr()
            
            # Check if performance is POOR
            if current_sharpe < poor_sharpe_threshold or current_return < poor_return_threshold:
                logger.info(f"[FIXED ADAPTIVE LR] 🔴 POOR PERFORMANCE DETECTED!")
                logger.info(f"[FIXED ADAPTIVE LR]    Sharpe {current_sharpe:.3f} < {poor_sharpe_threshold}")
                logger.info(f"[FIXED ADAPTIVE LR]    Return {current_return:.2%} < {poor_return_threshold:.2%}")
                
                # Calculate new LR (ALWAYS INCREASE, NEVER DECREASE)
                boosted_lr = current_lr * poor_boost_factor
                new_lr = max(current_lr, min(poor_lr_max, boosted_lr))
                
                # CRITICAL FIX: ACTUALLY UPDATE THE MODEL
                self._set_model_lr(new_lr)
                
                adjustment = {
                    'timestep': self.num_timesteps,
                    'event': 'LR_INCREASE_POOR',
                    'previous_lr': current_lr,
                    'new_lr': new_lr,
                    'boost_factor': poor_boost_factor,
                    'sharpe': current_sharpe,
                    'return': current_return,
                    'reason': 'Poor performance'
                }
                self.lr_adjustments.append(adjustment)
                
                logger.info(f"[FIXED ADAPTIVE LR] ✅ LR INCREASED: {current_lr:.6f} → {new_lr:.6f}")
            
            # Good but not excellent: decrease LR toward lr_final to stabilize (bidirectional adaptive)
            lr_final = getattr(config, 'lr_final', 1.5e-5)
            good_sharpe_max = getattr(config, 'adaptive_lr_sharpe_threshold', 2.0)
            good_return_max = getattr(config, 'adaptive_lr_return_threshold', 0.03)
            if (0.2 <= current_sharpe <= good_sharpe_max and 0 <= current_return <= good_return_max and
                    current_lr > lr_final * 1.5):
                new_lr = max(lr_final, current_lr * 0.7)
                self._set_model_lr(new_lr)
                adjustment = {
                    'timestep': self.num_timesteps,
                    'event': 'LR_DECREASE_GOOD',
                    'previous_lr': current_lr,
                    'new_lr': new_lr,
                    'sharpe': current_sharpe,
                    'return': current_return,
                    'reason': 'Good-but-not-excellent: stabilize'
                }
                self.lr_adjustments.append(adjustment)
                logger.info(f"[FIXED ADAPTIVE LR] ✅ LR DECREASED: {current_lr:.6f} → {new_lr:.6f} (good performance)")
                current_lr = new_lr

            # Check if performance is EXCELLENT
            excellent_sharpe_threshold = getattr(config, 'adaptive_lr_sharpe_threshold', 2.0)
            excellent_return_threshold = getattr(config, 'adaptive_lr_return_threshold', 0.03)
            excellent_boost_factor = getattr(config, 'adaptive_lr_boost_factor', 1.5)
            
            if current_sharpe > excellent_sharpe_threshold and current_return > excellent_return_threshold:
                logger.info(f"[FIXED ADAPTIVE LR] 🟢 EXCELLENT PERFORMANCE!")
                logger.info(f"[FIXED ADAPTIVE LR]    Sharpe {current_sharpe:.3f} > {excellent_sharpe_threshold}")
                logger.info(f"[FIXED ADAPTIVE LR]    Return {current_return:.2%} > {excellent_return_threshold:.2%}")
                
                # Boost LR to maintain adaptability
                new_lr = min(poor_lr_max, current_lr * excellent_boost_factor)
                
                if new_lr > current_lr:
                    self._set_model_lr(new_lr)
                    
                    adjustment = {
                        'timestep': self.num_timesteps,
                        'event': 'LR_INCREASE_EXCELLENT',
                        'previous_lr': current_lr,
                        'new_lr': new_lr,
                        'boost_factor': excellent_boost_factor,
                        'sharpe': current_sharpe,
                        'return': current_return,
                        'reason': 'Excellent performance'
                    }
                    self.lr_adjustments.append(adjustment)
                    
                    logger.info(f"[FIXED ADAPTIVE LR] ✅ LR INCREASED: {current_lr:.6f} → {new_lr:.6f}")
            
            # Update history
            self.previous_sharpe = current_sharpe
            self.previous_return = current_return
            
        except Exception as e:
            logger.error(f"[FIXED ADAPTIVE LR] ❌ Error during evaluation: {e}")
            import traceback
            traceback.print_exc()
        
        return True
    
    def _simple_evaluation(self) -> bool:
        """Fallback simple evaluation when PerformanceEvaluator is not available."""
        logger.info(f"[FIXED ADAPTIVE LR] Running simple evaluation...")
        
        # Just do a basic rollout
        obs = self.eval_env.reset()
        done = False
        episode_return = 0
        
        while not done:
            action, _states = self.model.predict(obs, deterministic=True)
            obs, reward, done, info = self.eval_env.step(action)
            episode_return += reward
        
        logger.info(f"[FIXED ADAPTIVE LR] Simple eval return: {episode_return}")
        
        return True
    
    def _get_current_lr(self) -> float:
        """Get current learning rate from model."""
        if hasattr(self.model, 'learning_rate'):
            if callable(self.model.learning_rate):
                # LR is a schedule
                progress = self.num_timesteps / self.model.num_timesteps if hasattr(self.model, 'num_timesteps') else 1.0
                return float(self.model.learning_rate(progress))
            else:
                return float(self.model.learning_rate)
        else:
            logger.warning(f"[FIXED ADAPTIVE LR] ⚠️ Cannot get LR from model")
            return 0.0
    
    def _set_model_lr(self, new_lr: float) -> None:
        """ACTUALLY set the learning rate on the model."""
        if hasattr(self.model, 'learning_rate'):
            # CRITICAL: Set the attribute directly
            self.model.learning_rate = new_lr
            
            # Also update optimizer if accessible
            if hasattr(self.model, 'policy') and hasattr(self.model.policy, 'optimizer'):
                for param_group in self.model.policy.optimizer.param_groups:
                    param_group['lr'] = new_lr
            
            logger.info(f"[FIXED ADAPTIVE LR] ✅ Model LR set to: {new_lr:.6f}")
        else:
            logger.error(f"[FIXED ADAPTIVE LR] ❌ Cannot set LR - model has no learning_rate attribute")
    
    def get_lr_history(self) -> list:
        """Return history of all LR adjustments."""
        return self.lr_adjustments


if __name__ == "__main__":
    print("Fixed Adaptive LR Callback - Phase 1")
    print("=" * 80)
    print("\nKey features:")
    print("1. ✅ Actually updates model.learning_rate")
    print("2. ✅ Works at first evaluation")
    print("3. ✅ Handles poor and excellent performance")
    print("4. ✅ Full logging of all changes")
    print("\nIntegrate into PPO_fixed.py by replacing AdaptiveHyperparameterCallback")

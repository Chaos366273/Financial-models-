"""
Hyperparameter optimizer for PPO Trading Agent
Handles parameter tuning and optimization
"""

import optuna
import numpy as np
from typing import Dict, List, Any, Optional, Callable
import logging
import os
import json
from datetime import datetime

from ..core.exceptions import ValidationException

class HyperparameterOptimizer:
    """Hyperparameter optimizer for PPO agent"""

    def __init__(self, config: Dict):
        """
        Initialize hyperparameter optimizer

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Optimization parameters
        self.n_trials = config.get('n_trials', 50)
        self.timeout = config.get('timeout', 3600)  # 1 hour
        self.study_name = config.get('study_name', 'ppo_optimization')
        self.storage = config.get('storage', 'sqlite:///optuna_study.db')

        # Parameter search spaces
        self.param_spaces = {
            'learning_rate': (1e-5, 1e-3, 'log'),
            'hidden_dim': (128, 512, 'int'),
            'batch_size': (32, 128, 'int'),
            'clip_epsilon': (0.1, 0.3, 'float'),
            'value_loss_coef': (0.1, 1.0, 'float'),
            'entropy_coef': (0.001, 0.1, 'log'),
            'max_grad_norm': (0.1, 1.0, 'float'),
            'gamma': (0.9, 0.999, 'float'),
            'lambda_': (0.9, 0.999, 'float'),
            'update_frequency': (5, 20, 'int'),
            'save_frequency': (50, 200, 'int')
        }

        # Create study
        self.study = self._create_study()

    def _create_study(self) -> optuna.Study:
        """Create Optuna study"""
        try:
            study = optuna.create_study(
                study_name=self.study_name,
                storage=self.storage,
                load_if_exists=True,
                direction='maximize',
                sampler=optuna.samplers.TPESampler(seed=42)
            )
            return study
        except Exception as e:
            self.logger.error(f"Failed to create study: {e}")
            # Fallback to in-memory study
            study = optuna.create_study(
                study_name=self.study_name,
                direction='maximize',
                sampler=optuna.samplers.TPESampler(seed=42)
            )
            return study

    def suggest_parameters(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        Suggest hyperparameters for a trial

        Args:
            trial: Optuna trial object

        Returns:
            Dictionary of suggested parameters
        """
        try:
            params = {}

            # Learning rate
            params['learning_rate'] = trial.suggest_float(
                'learning_rate',
                self.param_spaces['learning_rate'][0],
                self.param_spaces['learning_rate'][1],
                log=True
            )

            # Hidden dimension
            params['hidden_dim'] = trial.suggest_int(
                'hidden_dim',
                self.param_spaces['hidden_dim'][0],
                self.param_spaces['hidden_dim'][1]
            )

            # Batch size
            params['batch_size'] = trial.suggest_int(
                'batch_size',
                self.param_spaces['batch_size'][0],
                self.param_spaces['batch_size'][1]
            )

            # PPO parameters
            params['clip_epsilon'] = trial.suggest_float(
                'clip_epsilon',
                self.param_spaces['clip_epsilon'][0],
                self.param_spaces['clip_epsilon'][1]
            )

            params['value_loss_coef'] = trial.suggest_float(
                'value_loss_coef',
                self.param_spaces['value_loss_coef'][0],
                self.param_spaces['value_loss_coef'][1]
            )

            params['entropy_coef'] = trial.suggest_float(
                'entropy_coef',
                self.param_spaces['entropy_coef'][0],
                self.param_spaces['entropy_coef'][1],
                log=True
            )

            params['max_grad_norm'] = trial.suggest_float(
                'max_grad_norm',
                self.param_spaces['max_grad_norm'][0],
                self.param_spaces['max_grad_norm'][1]
            )

            # Training parameters
            params['gamma'] = trial.suggest_float(
                'gamma',
                self.param_spaces['gamma'][0],
                self.param_spaces['gamma'][1]
            )

            params['lambda_'] = trial.suggest_float(
                'lambda_',
                self.param_spaces['lambda_'][0],
                self.param_spaces['lambda_'][1]
            )

            params['update_frequency'] = trial.suggest_int(
                'update_frequency',
                self.param_spaces['update_frequency'][0],
                self.param_spaces['update_frequency'][1]
            )

            params['save_frequency'] = trial.suggest_int(
                'save_frequency',
                self.param_spaces['save_frequency'][0],
                self.param_spaces['save_frequency'][1]
            )

            return params

        except Exception as e:
            self.logger.error(f"Parameter suggestion failed: {e}")
            raise ValidationException(f"Parameter suggestion failed: {e}")

    def objective(self, trial: optuna.Trial, training_function: Callable) -> float:
        """
        Objective function for optimization

        Args:
            trial: Optuna trial object
            training_function: Function that takes parameters and returns reward

        Returns:
            Objective value (reward)
        """
        try:
            # Get suggested parameters
            params = self.suggest_parameters(trial)

            # Log trial parameters
            self.logger.info(f"Trial {trial.number} parameters: {params}")

            # Run training with suggested parameters
            reward = training_function(params)

            # Log trial result
            self.logger.info(f"Trial {trial.number} result: {reward}")

            return reward

        except Exception as e:
            self.logger.error(f"Trial {trial.number} failed: {e}")
            return float('-inf')

    def optimize(self, training_function: Callable) -> Dict[str, Any]:
        """
        Run hyperparameter optimization

        Args:
            training_function: Function that takes parameters and returns reward

        Returns:
            Best parameters and study information
        """
        try:
            self.logger.info(f"Starting hyperparameter optimization with {self.n_trials} trials")

            # Run optimization
            self.study.optimize(
                lambda trial: self.objective(trial, training_function),
                n_trials=self.n_trials,
                timeout=self.timeout
            )

            # Get best parameters
            best_params = self.study.best_params
            best_value = self.study.best_value

            # Save optimization results
            self._save_optimization_results(best_params, best_value)

            self.logger.info(f"Optimization completed. Best reward: {best_value}")
            self.logger.info(f"Best parameters: {best_params}")

            return {
                'best_params': best_params,
                'best_value': best_value,
                'study': self.study
            }

        except Exception as e:
            self.logger.error(f"Optimization failed: {e}")
            raise ValidationException(f"Optimization failed: {e}")

    def _save_optimization_results(self, best_params: Dict, best_value: float):
        """Save optimization results to file"""
        try:
            results = {
                'best_params': best_params,
                'best_value': best_value,
                'timestamp': datetime.now().isoformat(),
                'n_trials': self.n_trials,
                'study_name': self.study_name
            }

            # Create results directory
            os.makedirs('optimization_results', exist_ok=True)
            
            # Save results
            results_file = f"optimization_results/optimization_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)

            self.logger.info(f"Optimization results saved to {results_file}")

        except Exception as e:
            self.logger.error(f"Failed to save optimization results: {e}")

    def get_study_summary(self) -> Dict[str, Any]:
        """Get study summary"""
        try:
            trials = self.study.trials
            completed_trials = [t for t in trials if t.state == optuna.trial.TrialState.COMPLETE]

            if not completed_trials:
                return {'error': 'No completed trials'}

            values = [t.value for t in completed_trials if t.value is not None]
            if not values:
                return {'error': 'No completed trials with values'}
            
            summary = {
                'n_trials': len(trials),
                'n_completed_trials': len(completed_trials),
                'best_value': self.study.best_value,
                'best_params': self.study.best_params,
                'mean_value': float(np.mean(values)),
                'std_value': float(np.std(values)),
                'min_value': float(np.min(values)),
                'max_value': float(np.max(values))
            }

            return summary

        except Exception as e:
            self.logger.error(f"Failed to get study summary: {e}")
            return {'error': str(e)}

    def plot_optimization_history(self, save_path: Optional[str] = None):
        """Plot optimization history"""
        try:
            import matplotlib.pyplot as plt

            # Plot optimization history
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

            # Plot optimization history
            optuna.visualization.matplotlib.plot_optimization_history(self.study, ax=ax1)
            ax1.set_title('Optimization History')

            # Plot parameter importance
            optuna.visualization.matplotlib.plot_param_importances(self.study, ax=ax2)
            ax2.set_title('Parameter Importance')

            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                self.logger.info(f"Optimization plots saved to {save_path}")
            else:
                plt.show()

        except ImportError:
            self.logger.warning("Matplotlib not available for plotting")
        except Exception as e:
            self.logger.error(f"Failed to plot optimization history: {e}")

    def get_parameter_importance(self) -> Dict[str, float]:
        """Get parameter importance scores"""
        try:
            importance = optuna.importance.get_param_importances(self.study)
            return importance
        except Exception as e:
            self.logger.error(f"Failed to get parameter importance: {e}")
            return {}

    def suggest_parameters_from_best(self, n_suggestions: int = 5) -> List[Dict[str, Any]]:
        """
        Suggest parameters based on best trials

        Args:
            n_suggestions: Number of parameter suggestions

        Returns:
            List of parameter dictionaries
        """
        try:
            suggestions = []
            
            # Get top trials
            trials = self.study.trials
            completed_trials = [t for t in trials if t.state == optuna.trial.TrialState.COMPLETE]
            sorted_trials = sorted(
                completed_trials,
                key=lambda t: t.value if t.value is not None else float("-inf"),
                reverse=True
            )

            # Suggest parameters from best trials
            for i in range(min(n_suggestions, len(sorted_trials))):
                trial = sorted_trials[i]
                suggestions.append({
                    'trial_number': trial.number,
                    'value': trial.value,
                    'params': trial.params
                })

            return suggestions

        except Exception as e:
            self.logger.error(f"Failed to suggest parameters: {e}")
            return []

    def validate_parameters(self, params: Dict[str, Any]) -> bool:
        """
        Validate parameter values

        Args:
            params: Parameter dictionary

        Returns:
            True if parameters are valid
        """
        try:
            for param_name, (min_val, max_val, param_type) in self.param_spaces.items():
                if param_name not in params:
                    self.logger.warning(f"Missing parameter: {param_name}")
                    return False

                value = params[param_name]

                # Check range
                if value < min_val or value > max_val:
                    self.logger.warning(f"Parameter {param_name} out of range: {value}")
                    return False

                # Check type
                if param_type == 'int' and not isinstance(value, int):
                    self.logger.warning(f"Parameter {param_name} should be int: {value}")
                    return False

            return True

        except Exception as e:
            self.logger.error(f"Parameter validation failed: {e}")
            return False

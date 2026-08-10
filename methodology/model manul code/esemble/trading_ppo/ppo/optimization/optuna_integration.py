"""
Optuna integration for PPO Trading Agent
Provides additional utilities for hyperparameter optimization
"""

import optuna
import numpy as np
from typing import Dict, List, Any, Optional, Callable
import logging
import os
import json
from datetime import datetime
import pickle

from ..core.exceptions import ValidationException

class OptunaIntegration:
    """Optuna integration utilities"""

    def __init__(self, study_name: str = "ppo_study", storage: Optional[str] = None):
        """
        Initialize Optuna integration

        Args:
            study_name: Name of the study
            storage: Storage URL for the study
        """
        self.study_name = study_name
        self.storage = storage
        self.logger = logging.getLogger(__name__)

        # Create or load study
        self.study = self._create_or_load_study()

    def _create_or_load_study(self) -> optuna.Study:
        """Create or load Optuna study"""
        try:
            if self.storage:
                study = optuna.create_study(
                    study_name=self.study_name,
                    storage=self.storage,
                    load_if_exists=True,
                    direction='maximize'
                )
            else:
                study = optuna.create_study(
                    study_name=self.study_name,
                    direction='maximize'
                )
            return study
        except Exception as e:
            self.logger.error(f"Failed to create/load study: {e}")
            raise ValidationException(f"Failed to create/load study: {e}")

    def create_pruner(self, pruner_type: str = "median") -> optuna.pruners.BasePruner:
        """
        Create a pruner for early stopping

        Args:
            pruner_type: Type of pruner ('median', 'percentile', 'threshold')

        Returns:
            Optuna pruner
        """
        try:
            if pruner_type == "median":
                return optuna.pruners.MedianPruner(
                    n_startup_trials=5,
                    n_warmup_steps=10,
                    interval_steps=1
                )
            elif pruner_type == "percentile":
                return optuna.pruners.PercentilePruner(
                    percentile=25.0,
                    n_startup_trials=5,
                    n_warmup_steps=10,
                    interval_steps=1
                )
            elif pruner_type == "threshold":
                return optuna.pruners.ThresholdPruner(
                    upper=0.0,
                    n_warmup_steps=10,
                    interval_steps=1
                )
            else:
                self.logger.warning(f"Unknown pruner type: {pruner_type}, using MedianPruner")
                return optuna.pruners.MedianPruner()

        except Exception as e:
            self.logger.error(f"Failed to create pruner: {e}")
            return optuna.pruners.MedianPruner()

    def create_sampler(self, sampler_type: str = "tpe") -> optuna.samplers.BaseSampler:
        """
        Create a sampler for parameter suggestion

        Args:
            sampler_type: Type of sampler ('tpe', 'random', 'cmaes')

        Returns:
            Optuna sampler
        """
        try:
            if sampler_type == "tpe":
                return optuna.samplers.TPESampler(seed=42)
            elif sampler_type == "random":
                return optuna.samplers.RandomSampler(seed=42)
            elif sampler_type == "cmaes":
                return optuna.samplers.CmaEsSampler(seed=42)
            else:
                self.logger.warning(f"Unknown sampler type: {sampler_type}, using TPESampler")
                return optuna.samplers.TPESampler(seed=42)

        except Exception as e:
            self.logger.error(f"Failed to create sampler: {e}")
            return optuna.samplers.TPESampler(seed=42)

    def suggest_hyperparameters(self, trial: optuna.Trial, param_config: Dict) -> Dict[str, Any]:
        """
        Suggest hyperparameters based on configuration

        Args:
            trial: Optuna trial
            param_config: Parameter configuration

        Returns:
            Dictionary of suggested parameters
        """
        try:
            params = {}

            for param_name, config in param_config.items():
                param_type = config.get('type', 'float')
                min_val = config['min']
                max_val = config['max']
                log_scale = config.get('log', False)

                if param_type == 'float':
                    params[param_name] = trial.suggest_float(
                        param_name, min_val, max_val, log=log_scale
                    )
                elif param_type == 'int':
                    params[param_name] = trial.suggest_int(
                        param_name, min_val, max_val, log=log_scale
                    )
                elif param_type == 'categorical':
                    choices = config['choices']
                    params[param_name] = trial.suggest_categorical(param_name, choices)
                else:
                    self.logger.warning(f"Unknown parameter type: {param_type}")

            return params

        except Exception as e:
            self.logger.error(f"Failed to suggest hyperparameters: {e}")
            raise ValidationException(f"Failed to suggest hyperparameters: {e}")

    def report_intermediate_value(self, trial: optuna.Trial, value: float, step: int):
        """
        Report intermediate value for pruning

        Args:
            trial: Optuna trial
            value: Intermediate value
            step: Current step
        """
        try:
            trial.report(value, step)
        except Exception as e:
            self.logger.error(f"Failed to report intermediate value: {e}")

    def should_prune(self, trial: optuna.Trial) -> bool:
        """
        Check if trial should be pruned

        Args:
            trial: Optuna trial

        Returns:
            True if trial should be pruned
        """
        try:
            return trial.should_prune()
        except Exception as e:
            self.logger.error(f"Failed to check pruning: {e}")
            return False

    def save_study(self, filepath: str):
        """Save study to file"""
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Save study
            with open(filepath, 'wb') as f:
                pickle.dump(self.study, f)

            self.logger.info(f"Study saved to {filepath}")

        except Exception as e:
            self.logger.error(f"Failed to save study: {e}")
            raise ValidationException(f"Failed to save study: {e}")

    def load_study(self, filepath: str):
        """Load study from file"""
        try:
            if not os.path.exists(filepath):
                raise ValidationException(f"Study file not found: {filepath}")

            with open(filepath, 'rb') as f:
                self.study = pickle.load(f)

            self.logger.info(f"Study loaded from {filepath}")

        except Exception as e:
            self.logger.error(f"Failed to load study: {e}")
            raise ValidationException(f"Failed to load study: {e}")

    def get_trial_summary(self, trial_number: int) -> Dict[str, Any]:
        """
        Get summary of a specific trial

        Args:
            trial_number: Trial number

        Returns:
            Trial summary
        """
        try:
            if trial_number >= len(self.study.trials):
                raise ValidationException(f"Trial {trial_number} not found")

            trial = self.study.trials[trial_number]
            
            summary = {
                'number': trial.number,
                'state': trial.state.name,
                'value': trial.value,
                'params': trial.params,
                'user_attrs': trial.user_attrs,
                'system_attrs': trial.system_attrs,
                'datetime_start': trial.datetime_start.isoformat() if trial.datetime_start else None,
                'datetime_complete': trial.datetime_complete.isoformat() if trial.datetime_complete else None
            }

            return summary

        except Exception as e:
            self.logger.error(f"Failed to get trial summary: {e}")
            return {}

    def get_best_trials(self, n_trials: int = 10) -> List[Dict[str, Any]]:
        """
        Get best trials

        Args:
            n_trials: Number of best trials to return

        Returns:
            List of best trial summaries
        """
        try:
            completed_trials = [t for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE]
            sorted_trials = sorted(
                completed_trials,
                key=lambda t: t.value if t.value is not None else float("-inf"),
                reverse=True
            )

            best_trials = []
            for trial in sorted_trials[:n_trials]:
                best_trials.append({
                    'number': trial.number,
                    'value': trial.value,
                    'params': trial.params
                })

            return best_trials

        except Exception as e:
            self.logger.error(f"Failed to get best trials: {e}")
            return []

    def export_study_results(self, filepath: str):
        """Export study results to JSON"""
        try:
            results = {
                'study_name': self.study_name,
                'best_trial': {
                    'number': self.study.best_trial.number,
                    'value': self.study.best_trial.value,
                    'params': self.study.best_trial.params
                },
                'trials': []
            }

            for trial in self.study.trials:
                trial_info = {
                    'number': trial.number,
                    'state': trial.state.name,
                    'value': trial.value,
                    'params': trial.params
                }
                results['trials'].append(trial_info)

            with open(filepath, 'w') as f:
                json.dump(results, f, indent=2)

            self.logger.info(f"Study results exported to {filepath}")

        except Exception as e:
            self.logger.error(f"Failed to export study results: {e}")
            raise ValidationException(f"Failed to export study results: {e}")

    def create_visualization(self, plot_type: str = "optimization_history", save_path: Optional[str] = None):
        """
        Create visualization of study results

        Args:
            plot_type: Type of plot ('optimization_history', 'param_importances', 'parallel_coordinate')
            save_path: Path to save the plot
        """
        try:
            import matplotlib.pyplot as plt

            if plot_type == "optimization_history":
                optuna.visualization.matplotlib.plot_optimization_history(self.study)
                plt.title("Optimization History")
            elif plot_type == "param_importances":
                optuna.visualization.matplotlib.plot_param_importances(self.study)
                plt.title("Parameter Importances")
            elif plot_type == "parallel_coordinate":
                optuna.visualization.matplotlib.plot_parallel_coordinate(self.study)
                plt.title("Parallel Coordinate Plot")
            else:
                self.logger.warning(f"Unknown plot type: {plot_type}")
                return

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                self.logger.info(f"Plot saved to {save_path}")
            else:
                plt.show()

        except ImportError:
            self.logger.warning("Matplotlib not available for visualization")
        except Exception as e:
            self.logger.error(f"Failed to create visualization: {e}")

    def get_study_statistics(self) -> Dict[str, Any]:
        """Get comprehensive study statistics"""
        try:
            trials = self.study.trials
            completed_trials = [t for t in trials if t.state == optuna.trial.TrialState.COMPLETE]
            failed_trials = [t for t in trials if t.state == optuna.trial.TrialState.FAIL]

            if not completed_trials:
                return {'error': 'No completed trials'}

            values = [t.value for t in completed_trials if t.value is not None]
            if not values:
                return {'error': 'No completed trials with values'}

            statistics = {
                'total_trials': len(trials),
                'completed_trials': len(completed_trials),
                'failed_trials': len(failed_trials),
                'best_value': self.study.best_value,
                'best_trial_number': self.study.best_trial.number,
                'mean_value': float(np.mean(values)),
                'std_value': float(np.std(values)),
                'min_value': float(np.min(values)),
                'max_value': float(np.max(values)),
                'median_value': float(np.median(values))
            }

            return statistics

        except Exception as e:
            self.logger.error(f"Failed to get study statistics: {e}")
            return {'error': str(e)}

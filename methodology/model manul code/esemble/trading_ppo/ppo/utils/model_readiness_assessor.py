"""
Model Readiness Assessment System
Evaluates when individual regime models are ready for production use
"""

import json
import os
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

class ModelReadinessAssessor:
    """
    Assesses model readiness based on performance criteria and stability metrics
    """
    
    def __init__(self, config_path: str = "config.py"):
        self.config = self._load_config(config_path)
        self.assessment_history = []
        self.readiness_thresholds = self._get_readiness_thresholds()
        
    def _load_config(self, config_path: str):
        """Load configuration from external file"""
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location("config", config_path)
            if spec is None or spec.loader is None:
                raise FileNotFoundError(f"Config module spec not found for {config_path}")
            config_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(config_module)
            return config_module
        except Exception as e:
            logger.warning(f"Could not load config: {e}")
            return self._get_default_config()
    
    def _get_default_config(self):
        """Default configuration if config file not found"""
        class DefaultConfig:
            REGIME_EVAL_CRITERIA = {
                'global': {
                    'sharpe_min': 0.5,
                    'profit_factor_min': 1.2,
                    'max_drawdown_max': 0.15,
                    'win_rate_min': 0.45,
                    'passes_required': 2
                },
                'low_vol': {
                    'sharpe_min': 0.3,
                    'profit_factor_min': 1.1,
                    'max_drawdown_max': 0.10,
                    'win_rate_min': 0.40,
                    'passes_required': 2
                },
                'high_vol': {
                    'sharpe_min': 0.8,
                    'profit_factor_min': 1.4,
                    'max_drawdown_max': 0.20,
                    'win_rate_min': 0.50,
                    'passes_required': 2
                },
                'trend': {
                    'sharpe_min': 0.6,
                    'profit_factor_min': 1.3,
                    'max_drawdown_max': 0.15,
                    'win_rate_min': 0.45,
                    'passes_required': 2
                },
                'mean_revert': {
                    'sharpe_min': 0.4,
                    'profit_factor_min': 1.2,
                    'max_drawdown_max': 0.12,
                    'win_rate_min': 0.42,
                    'passes_required': 2
                }
            }
        return DefaultConfig()
    
    def _get_readiness_thresholds(self) -> Dict:
        """Get readiness thresholds for each regime"""
        return {
            'low_vol': {
                'sharpe_min': 0.3,
                'profit_factor_min': 1.1,
                'max_drawdown_max': 0.10,
                'win_rate_min': 0.40,
                'stability_score_min': 0.7,
                'consistency_score_min': 0.6
            },
            'high_vol': {
                'sharpe_min': 0.8,
                'profit_factor_min': 1.4,
                'max_drawdown_max': 0.20,
                'win_rate_min': 0.50,
                'stability_score_min': 0.7,
                'consistency_score_min': 0.6
            },
            'trend': {
                'sharpe_min': 0.6,
                'profit_factor_min': 1.3,
                'max_drawdown_max': 0.15,
                'win_rate_min': 0.45,
                'stability_score_min': 0.7,
                'consistency_score_min': 0.6
            },
            'mean_revert': {
                'sharpe_min': 0.4,
                'profit_factor_min': 1.2,
                'max_drawdown_max': 0.12,
                'win_rate_min': 0.42,
                'stability_score_min': 0.7,
                'consistency_score_min': 0.6
            }
        }
    
    def assess_model_readiness(self, regime_name: str, model_metrics: Dict) -> Dict:
        """
        Assess if a model is ready for production use
        
        Args:
            regime_name: Name of the regime (low_vol, high_vol, trend, mean_revert)
            model_metrics: Dictionary containing model performance metrics
            
        Returns:
            Dictionary with readiness assessment results
        """
        thresholds = self.readiness_thresholds.get(regime_name, {})
        
        # Extract metrics
        sharpe_ratio = model_metrics.get('sharpe_ratio', 0)
        win_rate = model_metrics.get('win_rate', 0)
        profit_factor = model_metrics.get('profit_factor', 0)
        max_drawdown = abs(model_metrics.get('max_drawdown', 0))
        
        # Calculate additional metrics
        stability_score = self._calculate_stability_score(model_metrics)
        consistency_score = self._calculate_consistency_score(model_metrics)
        
        # Check each criterion
        criteria_results = {
            'sharpe_ratio': {
                'required': thresholds.get('sharpe_min', 0),
                'actual': sharpe_ratio,
                'passed': sharpe_ratio >= thresholds.get('sharpe_min', 0),
                'weight': 0.3
            },
            'profit_factor': {
                'required': thresholds.get('profit_factor_min', 0),
                'actual': profit_factor,
                'passed': profit_factor >= thresholds.get('profit_factor_min', 0),
                'weight': 0.25
            },
            'max_drawdown': {
                'required': thresholds.get('max_drawdown_max', 1),
                'actual': max_drawdown,
                'passed': max_drawdown <= thresholds.get('max_drawdown_max', 1),
                'weight': 0.2
            },
            'win_rate': {
                'required': thresholds.get('win_rate_min', 0),
                'actual': win_rate,
                'passed': win_rate >= thresholds.get('win_rate_min', 0),
                'weight': 0.15
            },
            'stability_score': {
                'required': thresholds.get('stability_score_min', 0),
                'actual': stability_score,
                'passed': stability_score >= thresholds.get('stability_score_min', 0),
                'weight': 0.05
            },
            'consistency_score': {
                'required': thresholds.get('consistency_score_min', 0),
                'actual': consistency_score,
                'passed': consistency_score >= thresholds.get('consistency_score_min', 0),
                'weight': 0.05
            }
        }
        
        # Calculate overall readiness score
        total_score = 0
        total_weight = 0
        passed_criteria = 0
        
        for criterion, result in criteria_results.items():
            if result['passed']:
                total_score += result['weight']
                passed_criteria += 1
            total_weight += result['weight']
        
        readiness_score = total_score / total_weight if total_weight > 0 else 0
        
        # Determine readiness status
        if readiness_score >= 0.8 and passed_criteria >= 5:
            status = "READY"
        elif readiness_score >= 0.6 and passed_criteria >= 4:
            status = "MARGINAL"
        elif readiness_score >= 0.4 and passed_criteria >= 3:
            status = "NEEDS_IMPROVEMENT"
        else:
            status = "NOT_READY"
        
        # Create assessment result
        assessment = {
            'regime_name': regime_name,
            'timestamp': datetime.now().isoformat(),
            'status': status,
            'readiness_score': readiness_score,
            'passed_criteria': passed_criteria,
            'total_criteria': len(criteria_results),
            'criteria_results': criteria_results,
            'recommendations': self._generate_recommendations(criteria_results, regime_name)
        }
        
        # Store assessment history
        self.assessment_history.append(assessment)
        
        return assessment
    
    def _calculate_stability_score(self, metrics: Dict) -> float:
        """Calculate stability score based on performance consistency"""
        # This would analyze performance over multiple runs
        # For now, use a simple heuristic
        sharpe = metrics.get('sharpe_ratio', 0)
        win_rate = metrics.get('win_rate', 0)
        profit_factor = metrics.get('profit_factor', 0)
        
        # Higher values indicate more stability
        stability = min(1.0, (sharpe + 1) / 2) * 0.4 + win_rate * 0.3 + min(1.0, profit_factor / 2) * 0.3
        return stability
    
    def _calculate_consistency_score(self, metrics: Dict) -> float:
        """Calculate consistency score based on performance patterns"""
        # This would analyze trade consistency
        # For now, use win rate as a proxy
        win_rate = metrics.get('win_rate', 0)
        return win_rate
    
    def _generate_recommendations(self, criteria_results: Dict, regime_name: str) -> List[str]:
        """Generate specific recommendations for improvement"""
        recommendations = []
        
        for criterion, result in criteria_results.items():
            if not result['passed']:
                if criterion == 'sharpe_ratio':
                    recommendations.append(f"Increase Sharpe ratio from {result['actual']:.3f} to {result['required']:.3f}")
                elif criterion == 'profit_factor':
                    recommendations.append(f"Increase profit factor from {result['actual']:.3f} to {result['required']:.3f}")
                elif criterion == 'max_drawdown':
                    recommendations.append(f"Reduce max drawdown from {result['actual']:.3f} to {result['required']:.3f}")
                elif criterion == 'win_rate':
                    recommendations.append(f"Increase win rate from {result['actual']:.3f} to {result['required']:.3f}")
        
        # Add regime-specific recommendations
        if regime_name == 'trend' and criteria_results['sharpe_ratio']['actual'] < 0:
            recommendations.append("Trend model is losing money - needs complete retraining with better features")
        elif regime_name == 'high_vol' and criteria_results['sharpe_ratio']['actual'] < 0.1:
            recommendations.append("High volatility model needs more aggressive training parameters")
        
        return recommendations
    
    def assess_all_models(self) -> Dict:
        """Assess readiness of all regime models"""
        regime_progress_path = "ppo/saved_models/regime_progress.json"
        
        if not os.path.exists(regime_progress_path):
            return {"error": "Regime progress file not found"}
        
        try:
            with open(regime_progress_path, 'r') as f:
                regime_data = json.load(f)
            
            assessments = {}
            overall_status = "UNKNOWN"
            
            for regime_name, regime_info in regime_data.get('regime_status', {}).items():
                if regime_info.get('status') == 'COMPLETE':
                    assessment = self.assess_model_readiness(regime_name, regime_info)
                    assessments[regime_name] = assessment
            
            # Determine overall status
            ready_models = sum(1 for a in assessments.values() if a['status'] == 'READY')
            total_models = len(assessments)
            
            if ready_models == total_models and total_models > 0:
                overall_status = "ALL_READY"
            elif ready_models >= total_models * 0.75:
                overall_status = "MOSTLY_READY"
            elif ready_models >= total_models * 0.5:
                overall_status = "PARTIALLY_READY"
            else:
                overall_status = "NOT_READY"
            
            return {
                'overall_status': overall_status,
                'ready_models': ready_models,
                'total_models': total_models,
                'assessments': assessments,
                'next_steps': self._get_next_steps(assessments)
            }
            
        except Exception as e:
            logger.error(f"Error assessing models: {e}")
            return {"error": str(e)}
    
    def _get_next_steps(self, assessments: Dict) -> List[str]:
        """Get recommended next steps based on assessments"""
        next_steps = []
        
        # Check if any models are ready
        ready_models = [name for name, assessment in assessments.items() if assessment['status'] == 'READY']
        
        if len(ready_models) >= 2:
            next_steps.append("Ready to implement regime switching logic")
            next_steps.append("Begin neural network model selector training")
        
        # Check for failing models
        failing_models = [name for name, assessment in assessments.items() if assessment['status'] == 'NOT_READY']
        
        for model_name in failing_models:
            assessment = assessments[model_name]
            if model_name == 'trend' and assessment['criteria_results']['sharpe_ratio']['actual'] < 0:
                next_steps.append(f"CRITICAL: {model_name} model needs complete retraining")
            else:
                next_steps.append(f"Improve {model_name} model performance")
        
        # Check for pending models
        if 'mean_revert' not in assessments:
            next_steps.append("Train mean_revert regime model")
        
        return next_steps
    
    def save_assessment_report(self, output_path: str = "ppo/saved_models/model_readiness_report.json"):
        """Save comprehensive assessment report"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'assessment_summary': self.assess_all_models(),
            'assessment_history': self.assessment_history[-10:],  # Last 10 assessments
            'readiness_thresholds': self.readiness_thresholds
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Model readiness report saved to {output_path}")
        return report

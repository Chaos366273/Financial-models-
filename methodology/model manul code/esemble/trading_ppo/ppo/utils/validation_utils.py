"""
Validation utilities for trading system
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from pathlib import Path
import json

logger = logging.getLogger(__name__)

class ValidationUtils:
    """Utility functions for data and model validation"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def validate_trading_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Validate trading data for required structure and quality"""
        validation_report = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'data_info': {}
        }
        
        try:
            # Check if data is empty
            if data.empty:
                validation_report['is_valid'] = False
                validation_report['errors'].append("Data is empty")
                return validation_report
            
            # Check required columns
            required_columns = ['date', 'tic', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                validation_report['is_valid'] = False
                validation_report['errors'].append(f"Missing required columns: {missing_columns}")
            
            # Check data types
            if 'date' in data.columns:
                try:
                    data['date'] = pd.to_datetime(data['date'])
                except:
                    validation_report['errors'].append("Date column cannot be converted to datetime")
            
            # Check for negative prices
            if 'close' in data.columns:
                negative_prices = (data['close'] <= 0).sum()
                if negative_prices > 0:
                    validation_report['warnings'].append(f"Found {negative_prices} negative or zero prices")
            
            # Check for missing values
            missing_counts = data.isnull().sum()
            total_missing = missing_counts.sum()
            if total_missing > 0:
                validation_report['warnings'].append(f"Found {total_missing} missing values")
            
            # Check for duplicate rows
            duplicates = data.duplicated().sum()
            if duplicates > 0:
                validation_report['warnings'].append(f"Found {duplicates} duplicate rows")
            
            # Data info
            validation_report['data_info'] = {
                'shape': data.shape,
                'date_range': {
                    'start': data['date'].min() if 'date' in data.columns else None,
                    'end': data['date'].max() if 'date' in data.columns else None
                },
                'unique_tickers': data['tic'].nunique() if 'tic' in data.columns else 0,
                'memory_usage': data.memory_usage(deep=True).sum()
            }
            
            self.logger.info(f" Data validation completed: {data.shape}")
            
        except Exception as e:
            validation_report['is_valid'] = False
            validation_report['errors'].append(f"Validation error: {str(e)}")
            self.logger.error(f" Data validation failed: {e}")
        
        return validation_report
    
    def validate_model_predictions(self, predictions: np.ndarray, 
                                 expected_shape: Tuple[int, ...]) -> Dict[str, Any]:
        """Validate model predictions"""
        validation_report = {
            'is_valid': True,
            'errors': [],
            'warnings': []
        }
        
        try:
            # Check shape
            if predictions.shape != expected_shape:
                validation_report['is_valid'] = False
                validation_report['errors'].append(
                    f"Prediction shape {predictions.shape} doesn't match expected {expected_shape}"
                )
            
            # Check for NaN values
            if np.isnan(predictions).any():
                validation_report['is_valid'] = False
                validation_report['errors'].append("Predictions contain NaN values")
            
            # Check for infinite values
            if np.isinf(predictions).any():
                validation_report['is_valid'] = False
                validation_report['errors'].append("Predictions contain infinite values")
            
            # Check prediction range (for actions, should be reasonable)
            if predictions.min() < -10 or predictions.max() > 10:
                validation_report['warnings'].append("Predictions outside expected range [-10, 10]")
            
            self.logger.info(f" Model predictions validation completed")
            
        except Exception as e:
            validation_report['is_valid'] = False
            validation_report['errors'].append(f"Prediction validation error: {str(e)}")
            self.logger.error(f" Prediction validation failed: {e}")
        
        return validation_report
    
    def validate_environment(self, env) -> Dict[str, Any]:
        """Validate trading environment"""
        validation_report = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'env_info': {}
        }
        
        try:
            # Check observation space
            if hasattr(env, 'observation_space'):
                obs_shape = env.observation_space.shape
                validation_report['env_info']['observation_shape'] = obs_shape
                
                if len(obs_shape) == 0 or obs_shape[0] <= 0:
                    validation_report['is_valid'] = False
                    validation_report['errors'].append("Invalid observation space shape")
            else:
                validation_report['errors'].append("Environment missing observation_space")
            
            # Check action space
            if hasattr(env, 'action_space'):
                action_shape = env.action_space.shape
                validation_report['env_info']['action_shape'] = action_shape
                
                if len(action_shape) == 0 or action_shape[0] <= 0:
                    validation_report['is_valid'] = False
                    validation_report['errors'].append("Invalid action space shape")
            else:
                validation_report['errors'].append("Environment missing action_space")
            
            # Check reset method
            if not hasattr(env, 'reset'):
                validation_report['is_valid'] = False
                validation_report['errors'].append("Environment missing reset method")
            
            # Check step method
            if not hasattr(env, 'step'):
                validation_report['is_valid'] = False
                validation_report['errors'].append("Environment missing step method")
            
            # Test reset
            try:
                obs = env.reset()
                if isinstance(obs, tuple):
                    obs = obs[0]  # Handle new gym API
                
                if not isinstance(obs, (np.ndarray, list)):
                    validation_report['warnings'].append("Reset returned unexpected observation type")
                
                validation_report['env_info']['reset_successful'] = True
            except Exception as e:
                validation_report['is_valid'] = False
                validation_report['errors'].append(f"Reset failed: {str(e)}")
                validation_report['env_info']['reset_successful'] = False
            
            self.logger.info(f" Environment validation completed")
            
        except Exception as e:
            validation_report['is_valid'] = False
            validation_report['errors'].append(f"Environment validation error: {str(e)}")
            self.logger.error(f" Environment validation failed: {e}")
        
        return validation_report
    
    def validate_configuration(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate configuration parameters"""
        validation_report = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'config_info': {}
        }
        
        try:
            # Required parameters
            required_params = [
                'TOTAL_TIMESTEPS', 'LEARNING_RATE', 'BATCH_SIZE', 
                'WINDOW_SIZE', 'INITIAL_BALANCE', 'TRANSACTION_COST'
            ]
            
            for param in required_params:
                if param not in config:
                    validation_report['is_valid'] = False
                    validation_report['errors'].append(f"Missing required parameter: {param}")
                else:
                    validation_report['config_info'][param] = config[param]
            
            # Validate parameter ranges
            if 'TOTAL_TIMESTEPS' in config:
                if config['TOTAL_TIMESTEPS'] <= 0:
                    validation_report['errors'].append("TOTAL_TIMESTEPS must be positive")
                elif config['TOTAL_TIMESTEPS'] < 1000:
                    validation_report['warnings'].append("TOTAL_TIMESTEPS seems low for training")
            
            if 'LEARNING_RATE' in config:
                if config['LEARNING_RATE'] <= 0 or config['LEARNING_RATE'] > 1:
                    validation_report['errors'].append("LEARNING_RATE must be between 0 and 1")
            
            if 'BATCH_SIZE' in config:
                if config['BATCH_SIZE'] <= 0:
                    validation_report['errors'].append("BATCH_SIZE must be positive")
                elif config['BATCH_SIZE'] > 1000:
                    validation_report['warnings'].append("BATCH_SIZE seems high")
            
            if 'WINDOW_SIZE' in config:
                if config['WINDOW_SIZE'] <= 0:
                    validation_report['errors'].append("WINDOW_SIZE must be positive")
                elif config['WINDOW_SIZE'] > 100:
                    validation_report['warnings'].append("WINDOW_SIZE seems high")
            
            if 'INITIAL_BALANCE' in config:
                if config['INITIAL_BALANCE'] <= 0:
                    validation_report['errors'].append("INITIAL_BALANCE must be positive")
            
            if 'TRANSACTION_COST' in config:
                if config['TRANSACTION_COST'] < 0:
                    validation_report['errors'].append("TRANSACTION_COST cannot be negative")
                elif config['TRANSACTION_COST'] > 0.1:
                    validation_report['warnings'].append("TRANSACTION_COST seems high")
            
            self.logger.info(f" Configuration validation completed")
            
        except Exception as e:
            validation_report['is_valid'] = False
            validation_report['errors'].append(f"Configuration validation error: {str(e)}")
            self.logger.error(f" Configuration validation failed: {e}")
        
        return validation_report
    
    def save_validation_report(self, report: Dict[str, Any], filepath: str):
        """Save validation report to file"""
        try:
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            self.logger.info(f" Validation report saved to {filepath}")
            
        except Exception as e:
            self.logger.error(f" Failed to save validation report: {e}")
    
    def print_validation_summary(self, report: Dict[str, Any]):
        """Print validation summary to console"""
        print("\n" + "="*50)
        print("VALIDATION SUMMARY")
        print("="*50)
        
        if report['is_valid']:
            print(" Validation PASSED")
        else:
            print(" Validation FAILED")
        
        if report['errors']:
            print("\n ERRORS:")
            for error in report['errors']:
                print(f"  - {error}")
        
        if report['warnings']:
            print("\n  WARNINGS:")
            for warning in report['warnings']:
                print(f"  - {warning}")
        
        if 'data_info' in report:
            print("\n DATA INFO:")
            for key, value in report['data_info'].items():
                print(f"  - {key}: {value}")
        
        if 'env_info' in report:
            print("\n ENVIRONMENT INFO:")
            for key, value in report['env_info'].items():
                print(f"  - {key}: {value}")
        
        if 'config_info' in report:
            print("\n  CONFIG INFO:")
            for key, value in report['config_info'].items():
                print(f"  - {key}: {value}")
        
        print("="*50)

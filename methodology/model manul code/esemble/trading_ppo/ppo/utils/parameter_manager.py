"""
Parameter management functionality for the PPO Trading Agent
Extracted from TG4.1.py - parameter management functions and utilities
"""

import os
import sys
import warnings
import logging
import pickle
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd

__all__ = [
    "save_top_params",
    "load_top_params",
    "save_hyperparameters",
    "load_hyperparameters",
    "validate_parameters",
    "create_default_parameters",
    "merge_parameters",
    "export_parameters_to_json",
    "import_parameters_from_json",
    "test_parameter_management",
    "ParameterManager",
]
# Global constants
TOP_PARAMS_PATH = 'ppo/saved_models/top_params.pkl'

def save_top_params(params_list, filepath: str = TOP_PARAMS_PATH):
    """Persist a list of top-K parameter dictionaries for later full training."""
    try:
        import os, pickle
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(params_list, f)
        print(f"[SUCCESS] Saved top-{len(params_list)} parameter sets to {filepath}")
    except Exception as e:
        print(f"[ERROR] Failed to save top parameters: {e}")

def load_top_params(filepath: str = TOP_PARAMS_PATH):
    """Load previously saved top-K parameters. Returns an empty list if none exist."""
    try:
        import os, pickle
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            if isinstance(data, list):
                print(f"[SUCCESS] Loaded {len(data)} top parameter sets from {filepath}")
                return data
        print(f"[WARNING] No top parameters file found at {filepath}")
        return []
    except Exception as e:
        print(f"[ERROR] Failed to load top parameters: {e}")
        return []

def save_hyperparameters(params, filepath: str = 'ppo/saved_models/best_hyperparameters.pkl'):
    """Save hyperparameters to file"""
    try:
        import os, pickle
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(params, f)
        print(f"[SUCCESS] Saved hyperparameters to {filepath}")
    except Exception as e:
        print(f"[ERROR] Failed to save hyperparameters: {e}")

def load_hyperparameters(filepath: str = 'ppo/saved_models/best_hyperparameters.pkl'):
    """Load hyperparameters from file"""
    try:
        import os, pickle
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                params = pickle.load(f)
            print(f"[SUCCESS] Loaded hyperparameters from {filepath}")
            return params
        print(f"[WARNING] No hyperparameters file found at {filepath}")
        return None
    except Exception as e:
        print(f"[ERROR] Failed to load hyperparameters: {e}")
        return None

def validate_parameters(params):
    """Validate parameter dictionary"""
    errors = []
    
    # Required parameter categories
    required_categories = ['model_params', 'training_params', 'trading_params']
    
    for category in required_categories:
        if category not in params:
            errors.append(f"Missing required category: {category}")
    
    if errors:
        raise ValueError(f"Parameter validation failed: {'; '.join(errors)}")
    
    # Validate model parameters
    model_params = params.get('model_params', {})
    required_model_params = ['learning_rate', 'batch_size', 'n_steps']
    
    for param in required_model_params:
        if param not in model_params:
            errors.append(f"Missing required model parameter: {param}")
    
    # Validate training parameters
    training_params = params.get('training_params', {})
    required_training_params = ['total_timesteps', 'n_envs']
    
    for param in required_training_params:
        if param not in training_params:
            errors.append(f"Missing required training parameter: {param}")
    
    # Validate trading parameters
    trading_params = params.get('trading_params', {})
    required_trading_params = ['action_threshold', 'action_multiplier', 'min_trade_size']
    
    for param in required_trading_params:
        if param not in trading_params:
            errors.append(f"Missing required trading parameter: {param}")
    
    if errors:
        raise ValueError(f"Parameter validation failed: {'; '.join(errors)}")
    
    print("[PARAMS] Parameter validation passed")
    return True

def create_default_parameters():
    """Create default parameter set"""
    default_params = {
        'model_params': {
            'learning_rate': 3e-4,
            'batch_size': 64,
            'n_steps': 2048,
            'n_epochs': 10,
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'clip_range': 0.2,
            'clip_range_vf': None,
            'ent_coef': 0.01,
            'vf_coef': 0.5,
            'max_grad_norm': 0.5,
            'target_kl': None,
            'policy_kwargs': {
                'net_arch': [dict(pi=[64, 64], vf=[64, 64])]
            }
        },
        'training_params': {
            'total_timesteps': 1000000,
            'n_envs': 1,
            'eval_freq': 10000,
            'save_freq': 50000,
            'log_interval': 10
        },
        'trading_params': {
            'action_threshold': 0.5,
            'action_multiplier': 0.1,
            'min_trade_size': 100.0,
            'max_position_size': 0.2,
            'commission_rate': 0.0029,
            'max_position_risk_pct': 0.2,
            'max_portfolio_risk_pct': 0.7,
            'stop_loss_pct': 0.05,
            'take_profit_pct': 0.1
        }
    }
    
    print("[PARAMS] Created default parameter set")
    return default_params

def merge_parameters(base_params, override_params):
    """Merge parameter dictionaries with override taking precedence"""
    merged_params = {}
    
    for category in ['model_params', 'training_params', 'trading_params']:
        merged_params[category] = {}
        
        # Start with base parameters
        if category in base_params:
            merged_params[category].update(base_params[category])
        
        # Override with new parameters
        if category in override_params:
            merged_params[category].update(override_params[category])
    
    print("[PARAMS] Merged parameter sets")
    return merged_params

def export_parameters_to_json(params, filepath: str = 'ppo/saved_models/parameters.json'):
    """Export parameters to JSON format for human readability"""
    try:
        import json
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            else:
                return obj
        
        json_params = convert_numpy_types(params)
        
        with open(filepath, 'w') as f:
            json.dump(json_params, f, indent=2)
        
        print(f"[SUCCESS] Exported parameters to {filepath}")
        
    except Exception as e:
        print(f"[ERROR] Failed to export parameters to JSON: {e}")

def import_parameters_from_json(filepath: str = 'ppo/saved_models/parameters.json'):
    """Import parameters from JSON format"""
    try:
        import json
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                params = json.load(f)
            print(f"[SUCCESS] Imported parameters from {filepath}")
            return params
        else:
            print(f"[WARNING] No parameters file found at {filepath}")
            return None
    except Exception as e:
        print(f"[ERROR] Failed to import parameters from JSON: {e}")
        return None

def test_parameter_management():
    """Test parameter management functionality"""
    print("\n" + "="*50)
    print(" TESTING PARAMETER MANAGEMENT")
    print("="*50)
    
    try:
        # Create default parameters
        params = create_default_parameters()
        print(" Created default parameters")
        
        # Validate parameters
        validate_parameters(params)
        print(" Parameter validation passed")
        
        # Test saving and loading
        test_file = 'ppo/saved_models/test_params.pkl'
        save_top_params([params], test_file)
        loaded_params = load_top_params(test_file)
        
        if loaded_params and len(loaded_params) > 0:
            print(" Parameter save/load test passed")
        else:
            print(" Parameter save/load test failed")
        
        # Test JSON export/import
        json_file = 'ppo/saved_models/test_params.json'
        export_parameters_to_json(params, json_file)
        imported_params = import_parameters_from_json(json_file)
        
        if imported_params:
            print(" JSON export/import test passed")
        else:
            print(" JSON export/import test failed")
        
        # Clean up test files
        if os.path.exists(test_file):
            os.remove(test_file)
        if os.path.exists(json_file):
            os.remove(json_file)
        
        print("[SUCCESS] Parameter management test passed")
        return True
        
    except Exception as e:
        print(f"[ERROR] Parameter management test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("="*50)
    return True


class ParameterManager:
    """
    Minimal wrapper class to preserve backward compatibility with code that expects
    `from ppo.utils.parameter_manager import ParameterManager`.
    Methods delegate to the module-level implementations above.
    """

    def save_top_params(self, params_list, filepath: str = TOP_PARAMS_PATH):
        return save_top_params(params_list, filepath)

    def load_top_params(self, filepath: str = TOP_PARAMS_PATH):
        return load_top_params(filepath)

    def save_hyperparameters(self, params, filepath: str = 'ppo/saved_models/best_hyperparameters.pkl'):
        return save_hyperparameters(params, filepath)

    def load_hyperparameters(self, filepath: str = 'ppo/saved_models/best_hyperparameters.pkl'):
        return load_hyperparameters(filepath)

    def validate_parameters(self, params):
        return validate_parameters(params)

    def create_default_parameters(self):
        return create_default_parameters()

    def merge_parameters(self, base_params, override_params):
        return merge_parameters(base_params, override_params)

    def export_parameters_to_json(self, params, filepath: str = 'ppo/saved_models/parameters.json'):
        return export_parameters_to_json(params, filepath)

    def import_parameters_from_json(self, filepath: str = 'ppo/saved_models/parameters.json'):
        return import_parameters_from_json(filepath)

    def test_parameter_management(self):
        return test_parameter_management()

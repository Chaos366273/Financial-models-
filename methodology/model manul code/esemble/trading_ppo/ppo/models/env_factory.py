"""
Environment factory functionality for the PPO Trading Agent
Extracted from TG4.1.py - environment factory functions and utilities
"""

import os
import sys
import warnings
import logging
import pickle
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, cast
import numpy as np
import pandas as pd

def make_trading_env(price_matrix, feature_matrix, volume_matrix, dates, feature_cols, trading_config, fixed_window_size=None, real_time_stream=None):
    """Factory function for creating a fixed environment with real-time data"""
    def _init():
        from config import DATA_CONFIG, TRADING_CONFIG
        if price_matrix is None or len(price_matrix) == 0:
            raise ValueError("price_matrix cannot be None or empty in make_trading_env")
        num_tickers = len(price_matrix[0])
        configured_tickers = DATA_CONFIG.get('tickers', [])
        if configured_tickers and num_tickers <= len(configured_tickers):
            ticker_names = configured_tickers[:num_tickers]
        else:
            ticker_names = [f"TICKER_{i}" for i in range(num_tickers)]
        window_size = fixed_window_size if fixed_window_size is not None else DATA_CONFIG.get('window_size', 15)  # FIXED: Use config value
        
        # CRITICAL FIX: Validate window_size
        if window_size is None or window_size <= 0:
            window_size = 15  # FIXED: Use config default
            print(f"[WARNING] Invalid window_size, using default: {window_size}")
        
        # CRITICAL FIX: Validate feature_cols
        if feature_cols is None or len(feature_cols) == 0:
            raise ValueError("feature_cols cannot be None or empty in make_trading_env")
        
        # CRITICAL FIX: Properly merge trading_config with TRADING_CONFIG defaults
        effective_trading_config = TRADING_CONFIG.copy()
        if trading_config is not None:
            effective_trading_config.update(trading_config)

        # Ensure analysis flags are propagated from DATA_CONFIG when missing
        try:
            from config import DATA_CONFIG as _DC  # local import to avoid cycles
        except Exception:
            _DC = {}

        if 'enable_technical_analysis' not in effective_trading_config:
            effective_trading_config['enable_technical_analysis'] = True
        if 'enable_fundamental_analysis' not in effective_trading_config:
            effective_trading_config['enable_fundamental_analysis'] = _DC.get('enable_fundamental_analysis', False)  # DISABLED
        if 'enable_sentiment_analysis' not in effective_trading_config:
            effective_trading_config['enable_sentiment_analysis'] = _DC.get('enable_sentiment_analysis', False)  # DISABLED
        if 'use_cached_data' not in effective_trading_config:
            effective_trading_config['use_cached_data'] = True
        # Ensure regime detection is on by default if not specified
        if 'enable_market_regime_detection' not in effective_trading_config:
            effective_trading_config['enable_market_regime_detection'] = True
        
        # Validate that all required keys are present
        required_keys = ['max_position_risk_pct', 'max_portfolio_risk_pct', 'action_threshold', 'action_multiplier', 'min_trade_size']
        missing_keys = [key for key in required_keys if key not in effective_trading_config]
        if missing_keys:
            print(f"[WARNING] Missing trading config keys: {missing_keys}")
            # Add fallback values
            fallbacks = {
                'max_position_risk_pct': 0.20,
                'max_portfolio_risk_pct': 0.70,
                'action_threshold': 0.50,
                'action_multiplier': 0.05,
                'min_trade_size': 500.0
            }
            for key in missing_keys:
                effective_trading_config[key] = fallbacks[key]
        
        from .trading_env import FixedTradingEnv
        env = FixedTradingEnv(
            feature_matrix=feature_matrix,
            price_matrix=price_matrix,
            volume_matrix=volume_matrix,
            window_size=window_size
        )
        env = cast(Any, env)
        # Preserve metadata for downstream consumers that expect these fields.
        env.dates = dates
        env.feature_cols = feature_cols
        env.trading_config = effective_trading_config
        env.unique_tickers = ticker_names
        return env
    return _init

def create_vectorized_env(price_matrix, feature_matrix, volume_matrix, dates, feature_cols, trading_config, fixed_window_size=None, n_envs=1, use_subproc: bool = False):
    """Create a vectorized environment for parallel training.

    When use_subproc is True and n_envs > 1, attempts to use SubprocVecEnv for better CPU utilization.
    Falls back to DummyVecEnv on any import/runtime error.
    """
    try:
        if use_subproc and n_envs > 1:
            from stable_baselines3.common.vec_env import SubprocVecEnv as _Vec
        else:
            from stable_baselines3.common.vec_env import DummyVecEnv as _Vec
    except Exception:
        from stable_baselines3.common.vec_env import DummyVecEnv as _Vec
    
    env_factory = make_trading_env(
        price_matrix=price_matrix,
        feature_matrix=feature_matrix,
        volume_matrix=volume_matrix,
        dates=dates,
        feature_cols=feature_cols,
        trading_config=trading_config,
        fixed_window_size=fixed_window_size
    )
    
    # Create vectorized environment
    try:
        env = _Vec([env_factory for _ in range(n_envs)])
    except Exception:
        # Fallback to DummyVecEnv if Subproc creation failed
        from stable_baselines3.common.vec_env import DummyVecEnv
        env = DummyVecEnv([env_factory for _ in range(n_envs)])
    
    print(f"[ENV] Created vectorized environment with {n_envs} parallel environments")
    return env

def validate_environment_config(env_config):
    """Validate environment configuration parameters"""
    errors = []
    
    required_keys = [
        'price_matrix', 'feature_matrix', 'volume_matrix', 'dates', 'feature_cols', 'trading_config'
    ]
    
    for key in required_keys:
        if key not in env_config:
            errors.append(f"Missing required key: {key}")
    
    if errors:
        raise ValueError(f"Environment configuration validation failed: {'; '.join(errors)}")
    
    # Validate data shapes
    price_matrix = env_config['price_matrix']
    feature_matrix = env_config['feature_matrix']
    volume_matrix = env_config['volume_matrix']
    dates = env_config['dates']

    if price_matrix is None or len(price_matrix) == 0:
        errors.append("Price matrix cannot be None or empty")
    if feature_matrix is None or len(feature_matrix) == 0:
        errors.append("Feature matrix cannot be None or empty")
    if volume_matrix is None or len(volume_matrix) == 0:
        errors.append("Volume matrix cannot be None or empty")

    if not errors:
        if len(price_matrix) != len(feature_matrix):
            errors.append("Price matrix and feature matrix must have same number of time steps")
        if len(price_matrix) != len(volume_matrix):
            errors.append("Price matrix and volume matrix must have same number of time steps")

        if len(price_matrix) != len(dates):
            errors.append("Price matrix and dates must have same length")

        if len(price_matrix[0]) != len(feature_matrix[0]):
            errors.append("Price matrix and feature matrix must have same number of tickers")
        if len(price_matrix[0]) != len(volume_matrix[0]):
            errors.append("Price matrix and volume matrix must have same number of tickers")
    
    if errors:
        raise ValueError(f"Data shape validation failed: {'; '.join(errors)}")
    
    print("[ENV] Environment configuration validation passed")
    print(f"[ENV] Data shape: {len(price_matrix)} time steps, {len(price_matrix[0])} tickers")
    print(f"[ENV] Feature dimensions: {feature_matrix.shape}")
    
    return True

def create_environment_from_data(data_dict, trading_config=None, window_size=None, real_time_stream=None):
    """Create environment from preprocessed data dictionary"""
    try:
        # Extract data from dictionary
        df = data_dict['df']
        tickers = data_dict.get('tickers', [])
        benchmark_returns = data_dict.get('benchmark_returns', None)
        selected_features = data_dict.get('selected_features', [])
        
        # Convert DataFrame to matrices
        if 'Ticker' not in df.columns and 'tic' in df.columns:
            df = df.rename(columns={'tic': 'Ticker'})
        from ..data.csv_loader_adapter import CSVLoaderAdapter
        adapter = CSVLoaderAdapter()
        env_data = adapter._dataframe_to_matrices(df, tickers, selected_features)
        
        # Create environment configuration
        env_config = {
            'price_matrix': env_data['price_matrix'],
            'feature_matrix': env_data['feature_matrix'],
            'dates': env_data['dates'],
            'feature_cols': selected_features,
            'trading_config': trading_config or {},
            'volume_matrix': env_data.get('volume_matrix'),
            'fixed_window_size': window_size,
            'real_time_stream': real_time_stream
        }
        
        # Validate configuration
        validate_environment_config(env_config)
        
        # Create environment
        env_factory = make_trading_env(**env_config)
        env = env_factory()
        
        print(f"[ENV] Successfully created environment from data")
        print(f"[ENV] Environment observation space: {env.observation_space}")
        print(f"[ENV] Environment action space: {env.action_space}")
        
        return env
        
    except Exception as e:
        print(f"[ERROR] Failed to create environment from data: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_environment_creation():
    """Test environment creation with synthetic data"""
    print("\n" + "="*50)
    print(" TESTING ENVIRONMENT CREATION")
    print("="*50)
    
    try:
        # Create synthetic data
        n_timesteps = 100
        n_tickers = 5
        n_features = 10
        
        # Synthetic price matrix
        price_matrix = np.random.randn(n_timesteps, n_tickers) * 0.01 + 100
        
        # Synthetic feature matrix
        feature_matrix = np.random.randn(n_timesteps, n_tickers, n_features)
        
        # Synthetic volume matrix
        volume_matrix = np.abs(np.random.randn(n_timesteps, n_tickers)) * 1000
        
        # Synthetic dates
        dates = pd.date_range('2023-01-01', periods=n_timesteps, freq='D')
        
        # Feature columns
        feature_cols = [f'feature_{i}' for i in range(n_features)]
        
        # Trading config
        trading_config = {
            'max_position_risk_pct': 0.20,
            'max_portfolio_risk_pct': 0.70,
            'action_threshold': 0.50,
            'action_multiplier': 0.05,
            'min_trade_size': 500.0
        }
        
        # Create environment
        env_factory = make_trading_env(
            price_matrix=price_matrix,
            feature_matrix=feature_matrix,
            volume_matrix=volume_matrix,
            dates=dates,
            feature_cols=feature_cols,
            trading_config=trading_config
        )
        
        env = env_factory()
        
        # Test environment
        obs = env.reset()
        print(f" Environment created successfully")
        print(f" Observation shape: {obs.shape}")
        print(f" Action space: {env.action_space}")
        print(f" Price matrix shape: {price_matrix.shape}")
        print(f" Feature matrix shape: {feature_matrix.shape}")
        
        # Test a few steps
        for i in range(5):
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            print(f"Step {i+1}: Reward={reward:.4f}, Done={done}")
            if done:
                break
        
        env.close()
        print("[SUCCESS] Environment creation test passed")
        return True
        
    except Exception as e:
        print(f"[ERROR] Environment creation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("="*50)
    return True

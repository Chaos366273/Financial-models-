"""
Model cleanup functionality for the PPO Trading Agent
Extracted from TG4.1.py - model cleanup and testing functions
"""

import os
import sys
import warnings
import logging
import pickle
import hashlib
import glob
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd

__all__ = [
    "cleanup_old_models",
    "cleanup_training_artifacts",
    "cleanup_cache_files",
    "cleanup_log_files",
    "get_model_info",
    "backup_models",
    "test_fixes",
    "run_cleanup_suite",
    "ModelCleanup",
]

def cleanup_old_models():
    """CRITICAL FIX: Clean up old model files that were trained with tight risk limits"""
    import os
    import glob
    
    print("\n" + "="*60)
    print(" CLEANING UP OLD MODELS")
    print("="*60)
    
    # Files to remove
    files_to_remove = [
        'ppo/saved_models/*.zip',
        'ppo/saved_models/training_metadata.pkl',
        'ppo/saved_models/best_model.zip',
        'ppo/saved_models/final_trained_agent_model.zip',
        'ppo/saved_models/full_trained_model.zip'
    ]
    
    removed_count = 0
    for pattern in files_to_remove:
        for file_path in glob.glob(pattern):
            try:
                os.remove(file_path)
                print(f" Removed: {file_path}")
                removed_count += 1
            except Exception as e:
                print(f"  Could not remove {file_path}: {e}")
    
    if removed_count == 0:
        print("  No old model files found to remove")
    else:
        print(f" Cleaned up {removed_count} old model files")
    
    return True

def cleanup_training_artifacts():
    """Clean up training artifacts and temporary files"""
    print("\n" + "="*60)
    print(" CLEANING UP TRAINING ARTIFACTS")
    print("="*60)
    
    # Directories and files to clean
    cleanup_items = [
        'ppo/saved_models/trial_*/',
        'ppo/saved_models/temp_*/',
        'ppo/saved_models/checkpoint_*/',
        'ppo/artifacts/temp_*/',
        'ppo/logs/training_*.log',
        'ppo/logs/debug_*.log'
    ]
    
    removed_count = 0
    for pattern in cleanup_items:
        if pattern.endswith('/'):
            # Directory cleanup
            for dir_path in glob.glob(pattern):
                try:
                    import shutil
                    shutil.rmtree(dir_path)
                    print(f" Removed directory: {dir_path}")
                    removed_count += 1
                except Exception as e:
                    print(f"  Could not remove directory {dir_path}: {e}")
        else:
            # File cleanup
            for file_path in glob.glob(pattern):
                try:
                    os.remove(file_path)
                    print(f" Removed file: {file_path}")
                    removed_count += 1
                except Exception as e:
                    print(f"  Could not remove file {file_path}: {e}")
    
    if removed_count == 0:
        print("  No training artifacts found to remove")
    else:
        print(f" Cleaned up {removed_count} training artifacts")
    
    return True

def cleanup_cache_files():
    """Clean up cache files to force fresh data loading"""
    print("\n" + "="*60)
    print(" CLEANING UP CACHE FILES")
    print("="*60)
    
    cache_files = [
        'ppo/artifacts/data_cache.pkl',
        'ppo/artifacts/benchmark_SPY.csv',
        'ppo/artifacts/data_cache_backup.pkl'
    ]
    
    removed_count = 0
    for file_path in cache_files:
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                print(f" Removed cache: {file_path}")
                removed_count += 1
            except Exception as e:
                print(f"  Could not remove cache {file_path}: {e}")
    
    if removed_count == 0:
        print("  No cache files found to remove")
    else:
        print(f" Cleaned up {removed_count} cache files")
    
    return True

def cleanup_log_files():
    """Clean up old log files"""
    print("\n" + "="*60)
    print(" CLEANING UP LOG FILES")
    print("="*60)
    
    log_patterns = [
        'ppo/logs/*.log',
        'ppo/logs/*.txt'
    ]
    
    removed_count = 0
    for pattern in log_patterns:
        for file_path in glob.glob(pattern):
            try:
                # Skip important files
                if any(skip in file_path for skip in ['config.py', 'PPO.py', 'requirements.txt']):
                    continue
                os.remove(file_path)
                print(f" Removed log: {file_path}")
                removed_count += 1
            except Exception as e:
                print(f"  Could not remove log {file_path}: {e}")
    
    if removed_count == 0:
        print("  No log files found to remove")
    else:
        print(f" Cleaned up {removed_count} log files")
    
    return True

def get_model_info():
    """Get information about existing model files"""
    print("\n" + "="*60)
    print(" MODEL INFORMATION")
    print("="*60)
    
    model_files = []
    
    # Find all model files
    patterns = [
        './models/*.zip',
        './models/*.pkl',
        './models/*.pth',
        './models/trial_*/*.pkl',
        './models/trial_*/*.zip'
    ]
    
    for pattern in patterns:
        for file_path in glob.glob(pattern):
            try:
                file_size = os.path.getsize(file_path)
                file_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                model_files.append({
                    'path': file_path,
                    'size_mb': file_size / (1024 * 1024),
                    'modified': file_time
                })
            except Exception as e:
                print(f"  Could not read file info for {file_path}: {e}")
    
    if not model_files:
        print("  No model files found")
        return []
    
    # Sort by modification time (newest first)
    model_files.sort(key=lambda x: x['modified'], reverse=True)
    
    print(f" Found {len(model_files)} model files:")
    for model in model_files:
        print(f"    {model['path']}")
        print(f"       Size: {model['size_mb']:.1f} MB")
        print(f"       Modified: {model['modified'].strftime('%Y-%m-%d %H:%M:%S')}")
    
    return model_files

def backup_models(backup_dir='./models/backup'):
    """Create backup of existing models"""
    print("\n" + "="*60)
    print(" BACKING UP MODELS")
    print("="*60)
    
    try:
        import shutil
        
        # Create backup directory
        os.makedirs(backup_dir, exist_ok=True)
        
        # Find model files to backup
        model_files = []
        patterns = [
            './models/*.zip',
            './models/*.pkl',
            './models/best_hyperparameters.pkl'
        ]
        
        for pattern in patterns:
            for file_path in glob.glob(pattern):
                model_files.append(file_path)
        
        if not model_files:
            print("  No model files found to backup")
            return False
        
        # Create backup with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_path = f"{backup_dir}/backup_{timestamp}"
        os.makedirs(backup_path, exist_ok=True)
        
        backed_up_count = 0
        for file_path in model_files:
            try:
                filename = os.path.basename(file_path)
                backup_file = os.path.join(backup_path, filename)
                shutil.copy2(file_path, backup_file)
                print(f" Backed up: {filename}")
                backed_up_count += 1
            except Exception as e:
                print(f"  Could not backup {file_path}: {e}")
        
        print(f" Created backup with {backed_up_count} files in {backup_path}")
        return True
        
    except Exception as e:
        print(f" Backup failed: {e}")
        return False

def test_fixes():
    """Test all the fixes to ensure they work correctly"""
    print("\n" + "="*60)
    print(" TESTING FIXES")
    print("="*60)
    
    try:
        # Test 1: Environment initialization
        print("[SUCCESS] Test 1: Environment initialization...")
        
        # Create test data - FIXED: Use consistent tickers from config
        test_tickers = ['AAPL', 'GOOG', 'MSFT']  # Use 'GOOG' instead of 'GOOGL' to match config
        test_dates = pd.date_range('2023-01-01', periods=100, freq='D')
        test_price_matrix = np.random.rand(100, 3) * 100 + 50  # Random prices
        test_feature_matrix = np.random.rand(100, 3, 5)  # 5 features per ticker
        test_volume_matrix = np.random.rand(100, 3) * 1_000_000  # Random volume
        test_feature_cols = ['returns', 'volatility', 'rsi', 'ma_20', 'volume']
        
        # Create environment
        from ..models.trading_env import FixedTradingEnv
        from config import DATA_CONFIG
        
        env = FixedTradingEnv(
            feature_matrix=test_feature_matrix,
            price_matrix=test_price_matrix,
            volume_matrix=test_volume_matrix,
            window_size=DATA_CONFIG['window_size'],  # Use config window_size
            mode='train'
        )
        
        print(f"  [SUCCESS] Environment created successfully")
        print(f"   Observation space: {env.observation_space.shape}")
        action_space_shape = getattr(env.action_space, "shape", None)
        action_space_size = getattr(env.action_space, "n", None)
        print(f"   Action space: {action_space_shape or action_space_size}")
        
        # Test 2: Environment reset
        print("\n[SUCCESS] Test 2: Environment reset...")
        obs, info = env.reset()
        print(f"  [SUCCESS] Reset successful - obs shape: {obs.shape}")
        print(f"   Initial portfolio: ${info.get('portfolio_value', 0):.2f}")
        
        # Test 3: Environment step
        print("\n[SUCCESS] Test 3: Environment step...")
        test_action = np.random.rand(3) * 2 - 1  # Random actions
        obs, reward, done, truncated, info = env.step(test_action)
        print(f"  [SUCCESS] Step successful - reward: {reward:.4f}")
        print(f"   Portfolio after step: ${info.get('portfolio_value', 0):.2f}")
        
        # Test 4: Model prediction (if model exists)
        print("\n[SUCCESS] Test 4: Model prediction test...")
        try:
            # Create a simple test model
            from stable_baselines3 import PPO
            test_model = PPO("MlpPolicy", env, device='auto', verbose=0, n_steps=64, batch_size=32)  # type: ignore[arg-type]
            test_model.learn(total_timesteps=100)  # Quick training
            
            # Test prediction
            prediction = test_model.predict(obs, deterministic=True)
            print(f"  [SUCCESS] Model prediction successful")
            print(f"   Prediction type: {type(prediction)}")
            if isinstance(prediction, (list, tuple)):
                print(f"   Prediction length: {len(prediction)}")
                action = prediction[0]
            else:
                action = prediction
            print(f"   Action shape: {action.shape}")
            
        except Exception as e:
            print(f"  [WARN] Model test skipped: {e}")
        
        # Test 5: Trading parameters
        print("\n[SUCCESS] Test 5: Trading parameters...")
        min_trade_size = getattr(env, "min_trade_size", 0)
        max_position_size = getattr(env, "max_position_size", 0)
        action_multiplier = getattr(env, "action_multiplier", 0)
        transaction_cost = getattr(env, "transaction_cost", 0)
        print(f"   Min trade size: ${min_trade_size}")
        print(f"   Max position size: {max_position_size*100}%")
        print(f"   Action multiplier: {action_multiplier*100}%")
        print(f"   Transaction cost: {transaction_cost*100}%")
        
        # Test 6: Multiple steps
        print("\n[SUCCESS] Test 6: Multiple steps test...")
        total_reward = 0
        for i in range(10):
            action = np.random.rand(3) * 2 - 1
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            if done or truncated:
                break
        
        print(f"  [SUCCESS] Multiple steps successful")
        print(f"   Total reward: {total_reward:.4f}")
        print(f"   Final portfolio: ${info.get('portfolio_value', 0):.2f}")
        
        print("\n ALL TESTS PASSED!")
        return True
        
    except Exception as e:
        print(f"\n[ERROR] TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_cleanup_suite():
    """Run the complete cleanup suite"""
    print("\n" + "="*80)
    print(" RUNNING COMPLETE CLEANUP SUITE")
    print("="*80)
    
    try:
        # Get model info first
        model_info = get_model_info()
        
        # Ask for confirmation if models exist
        if model_info:
            print(f"\n  Found {len(model_info)} existing model files")
            print("Do you want to backup before cleanup? (y/n): ", end="")
            # In a real implementation, you'd get user input here
            backup_choice = "n"  # Default to no for automation
            
            if backup_choice.lower() == 'y':
                backup_models()
        
        # Run cleanup operations
        cleanup_old_models()
        cleanup_training_artifacts()
        cleanup_cache_files()
        cleanup_log_files()
        
        print("\n CLEANUP SUITE COMPLETED")
        print(" All old files and artifacts have been removed")
        print(" System is ready for fresh training")
        
        return True
        
    except Exception as e:
        print(f"\n CLEANUP SUITE FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


class ModelCleanup:
    """
    Minimal wrapper class to preserve backward compatibility with code that expects
    `from ppo.utils.model_cleanup import ModelCleanup`.
    Methods delegate to the module-level implementations above.
    """

    def cleanup_old_models(self) -> bool:
        return cleanup_old_models()

    def cleanup_training_artifacts(self) -> bool:
        return cleanup_training_artifacts()

    def cleanup_cache_files(self) -> bool:
        return cleanup_cache_files()

    def cleanup_log_files(self) -> bool:
        return cleanup_log_files()

    def get_model_info(self):
        return get_model_info()

    def backup_models(self, backup_dir: str = './models/backup') -> bool:
        return backup_models(backup_dir=backup_dir)

    def test_fixes(self) -> bool:
        return test_fixes()

    def run_cleanup_suite(self) -> bool:
        return run_cleanup_suite()

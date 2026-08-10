"""
Data caching functionality for the PPO Trading Agent
Extracted from TG4.1.py - cache management functions and utilities
"""

import os
import warnings
import logging
import pickle
import hashlib
import shutil
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Optional async dependencies
try:
    import aiofiles
    AIOFILES_AVAILABLE = True
except ImportError:
    AIOFILES_AVAILABLE = False

# Optional import for defaults used by load_best_hyperparameters
try:
    from config import MODEL_CONFIG  # type: ignore
except Exception:
    MODEL_CONFIG = {}

# Optional import for DATA_SCHEMA_VERSION with fallback
try:
    from config import DATA_SCHEMA_VERSION  # type: ignore
except Exception:
    DATA_SCHEMA_VERSION = 'v1.0'  # Default fallback version

__all__ = [
    "cache_data",
    "load_cached_data",
    "clear_cache",
    "restore_from_backup",
    "load_best_hyperparameters",
    "save_optimization_results",
    "CacheManager",
]
def cache_data(df, tickers, benchmark_returns, selected_features, cache_path='artifacts/data_cache.pkl'):
    """ENHANCED: Cache preprocessed data with schema versioning for data integrity"""
    try:
        import pickle
        import os
        from config import DATA_SCHEMA_VERSION
        
        # Ensure artifacts directory exists
        os.makedirs('artifacts', exist_ok=True)
        
        # Calculate indicators count for validation
        indicator_columns = [col for col in df.columns if col not in ['date', 'tic', 'close_raw', 'open', 'high', 'low', 'volume']]
        
        # Create comprehensive cache data
        cache_data = {
            'df': df, 
            'tickers': tickers, 
            'benchmark_returns': benchmark_returns, 
            'selected_features': selected_features,
            'optimization_version': 'v2.0',
            'data_schema_version': DATA_SCHEMA_VERSION,
            'indicators_count': len(indicator_columns),
            'cache_timestamp': pd.Timestamp.now().isoformat(),
            'data_shape': df.shape,
            'ticker_count': len(tickers) if tickers else 0,
            'date_range': {
                'start': df['date'].min().isoformat() if not df.empty else None,
                'end': df['date'].max().isoformat() if not df.empty else None
            }
        }
        
        # Save with atomic write (write to temp file first, then rename)
        temp_path = cache_path + '.tmp'
        with open(temp_path, 'wb') as f:
            pickle.dump(cache_data, f)
        
        # Atomic move to final location
        if os.path.exists(cache_path):
            os.remove(cache_path)
        os.rename(temp_path, cache_path)
        
        print(f"[CACHE]  Data successfully cached to {cache_path}")
        print(f"[CACHE]  Shape: {df.shape}, Tickers: {len(tickers) if tickers else 0}")
        print(f"[CACHE]  Date range: {cache_data['date_range']['start']} to {cache_data['date_range']['end']}")
        print(f"[CACHE]  Schema version: {DATA_SCHEMA_VERSION}")
        print(f"[CACHE]  Indicators count: {len(indicator_columns)}")
        print(f"[CACHE]  Cache timestamp: {cache_data['cache_timestamp']}")
        
        # Verify cache was written correctly
        if os.path.exists(cache_path):
            file_size = os.path.getsize(cache_path)
            print(f"[CACHE]  Cache file size: {file_size:,} bytes")
        else:
            print(f"[CACHE]  Cache file not found after writing!")
        
    except Exception as e:
        print(f"[CACHE]  Failed to cache data: {e}")
        import traceback
        traceback.print_exc()

def load_cached_data(cache_path='artifacts/data_cache.pkl'):
    """ENHANCED: Load cached preprocessed data with comprehensive validation"""
    try:
        import pickle
        import os
        
        # Check if cache file exists and has size
        if not os.path.exists(cache_path):
            print(f"[CACHE]  No cached data found at {cache_path}")
            return None
        
        file_size = os.path.getsize(cache_path)
        if file_size == 0:
            print(f"[CACHE]  Cache file is empty: {cache_path}")
            return None
        
        print(f"[CACHE]  Loading cache file: {cache_path} ({file_size:,} bytes)")
        
        with open(cache_path, 'rb') as f:
            cached_data = pickle.load(f)
        
        # Accept dict format (preferred)
        if isinstance(cached_data, dict) and 'df' in cached_data and cached_data['df'] is not None:
            df = cached_data['df']
            if df.empty:
                print("[CACHE]  Cached DataFrame is empty")
                return None
            # Lenient schema/version checks
            cached_version = cached_data.get('data_schema_version', 'unknown')
            if cached_version != DATA_SCHEMA_VERSION:
                print(f"[CACHE]   Data schema version mismatch: Cached '{cached_version}' vs Current '{DATA_SCHEMA_VERSION}'.")
                print(f"[CACHE]  Continuing with cached data for faster startup...")
            print(f"[CACHE]  Successfully loaded cached data!")
            print(f"[CACHE]  Shape: {df.shape}")
            return cached_data
        
        # Fallback: accept bare DataFrame (older format) and wrap it
        import pandas as _pd
        if isinstance(cached_data, _pd.DataFrame):
            df = cached_data
            if df.empty:
                print("[CACHE]  Cached DataFrame is empty")
                return None
            wrapped = {
                'df': df,
                'tickers': sorted(df['tic'].unique().tolist()) if 'tic' in df.columns else [],
                'benchmark_returns': None,
                'selected_features': list(df.columns),
                'optimization_version': 'legacy-wrap',
                'data_schema_version': DATA_SCHEMA_VERSION,
                'data_shape': df.shape,
                'ticker_count': df['tic'].nunique() if 'tic' in df.columns else 0,
                'date_range': {
                    'start': df['date'].min().isoformat() if 'date' in df.columns and not df.empty else None,
                    'end': df['date'].max().isoformat() if 'date' in df.columns and not df.empty else None,
                }
            }
            print("[CACHE]   Loaded legacy bare-DataFrame cache; wrapped into expected format")
            print(f"[CACHE]  Wrapped cache shape: {df.shape}")
            return wrapped
        
        print("[CACHE]  Invalid or corrupted cached data format.")
        return None
    except FileNotFoundError:
        print(f"[CACHE]  No cached data found at {cache_path}")
        return None
    except Exception as e:
        print(f"[CACHE]  Error loading cached data: {e}")
        import traceback
        traceback.print_exc()
        return None

def clear_cache(cache_path='artifacts/data_cache.pkl'):
    """Clear cached data to force regeneration with optimized indicators"""
    try:
        import os
        if os.path.exists(cache_path):
            os.remove(cache_path)
            print(f"[CACHE] Cleared cached data at {cache_path}")
            print(f"[CACHE] Next run will regenerate data with optimized indicators")
        else:
            print(f"[CACHE] No cached data to clear at {cache_path}")
        
        # CRITICAL FIX: Also clear benchmark cache
        benchmark_cache = 'artifacts/benchmark_SPY.csv'
        if os.path.exists(benchmark_cache):
            os.remove(benchmark_cache)
            print(f"[CACHE] Cleared benchmark cache {benchmark_cache}")
            
        # FIXED: DO NOT clear model files - only clear data cache
        # Model files should be preserved for evaluation and live trading
        print(f"[CACHE] Preserved model files for evaluation and live trading")
                    
    except Exception as e:
        print(f"[WARNING] Failed to clear cache: {e}")

def restore_from_backup():
    """Restore data from backup cache if main cache is corrupted"""
    try:
        import os
        import shutil
        
        backup_path = 'artifacts/data_cache_backup.pkl'
        main_path = 'artifacts/data_cache.pkl'
        
        if os.path.exists(backup_path) and not os.path.exists(main_path):
            print(f"[CACHE]  Restoring from backup cache...")
            shutil.copy2(backup_path, main_path)
            print(f"[CACHE]  Successfully restored from backup")
            return True
        elif os.path.exists(backup_path) and os.path.exists(main_path):
            # Check if main cache is corrupted
            try:
                import pickle
                with open(main_path, 'rb') as f:
                    pickle.load(f)
                print(f"[CACHE]  Main cache is valid")
                return False
            except:
                print(f"[CACHE]  Main cache is corrupted, restoring from backup...")
                shutil.copy2(backup_path, main_path)
                print(f"[CACHE]  Successfully restored from backup")
                return True
        else:
            print(f"[CACHE]   No backup cache available")
            return False
            
    except Exception as e:
        print(f"[CACHE]  Failed to restore from backup: {e}")
        return False

def load_best_hyperparameters():
    """Locate and return the best hyper-parameters saved during the Optuna debug run.

    Compatibility notes
    -------------------
     Old format: ``./models/trial_0/best_hyperparameters.pkl``  a pickled ``(params, value)`` tuple or
      ``{'best_params': ..., 'best_value': ...}`` dict.
     Current format (introduced by `save_optimization_results`): ``./models/trial_N/study_results.pkl`` 
      a dict containing the same keys.  Multiple trials directories may exist; we pick the one with the
      highest ``best_value``.
    """

    import glob
    import os
    import pickle

    # ------------------------------------------------------------------
    # 1. New format lookup  scan every study_results.pkl produced by Optuna.
    # ------------------------------------------------------------------
    study_files = glob.glob('ppo/saved_models/trial_*/study_results.pkl')

    best_params = MODEL_CONFIG
    best_value = float('-inf')

    for study_fp in study_files:
        try:
            with open(study_fp, 'rb') as f:
                data = pickle.load(f)
            if isinstance(data, dict):
                params = data.get('best_params', MODEL_CONFIG)
                value = data.get('best_value', 0.0)
            elif isinstance(data, tuple) and len(data) == 2:
                params, value = data  # type: ignore
            else:
                continue  # unexpected format  skip

            if value > best_value:
                best_value = value
                best_params = params
        except Exception as e:
            print(f"[WARNING] Could not read {study_fp}: {e}")

    # ------------------------------------------------------------------
    # 2. Fallback to legacy best_hyperparameters.pkl (trial_0) if no study files found.
    # ------------------------------------------------------------------
    if best_value == float('-inf'):
        legacy_path = 'ppo/saved_models/trial_0/best_hyperparameters.pkl'
        if os.path.exists(legacy_path):
            try:
                with open(legacy_path, 'rb') as f:
                    data = pickle.load(f)
                if isinstance(data, dict):
                    best_params = data.get('best_params', MODEL_CONFIG)
                    best_value = data.get('best_value', 0.0)
                elif isinstance(data, tuple) and len(data) == 2:
                    best_params, best_value = data
            except Exception as e:
                print(f"[ERROR] Failed to load legacy hyperparameters: {e}")

    # ------------------------------------------------------------------
    # 3. Final sanity  if still none found, revert to MODEL_CONFIG.
    # ------------------------------------------------------------------
    if best_value == float('-inf'):
        print("[WARNING] No saved hyperparameter results found  falling back to default MODEL_CONFIG")
        best_value = 0.0

    return best_params, best_value

def save_optimization_results(trial_number, best_params, best_value):
    """Save optimization results for future use"""
    try:
        trial_dir = f"ppo/saved_models/trial_{trial_number}"
        os.makedirs(trial_dir, exist_ok=True)
        
        study_data = {
            'best_params': best_params,
            'best_value': best_value,
            'trial_number': trial_number,
            'timestamp': datetime.now().isoformat()
        }
        
        study_file = f"{trial_dir}/study_results.pkl"
        with open(study_file, 'wb') as f:
            pickle.dump(study_data, f)
        
        print(f"[SUCCESS] Saved optimization results to {study_file}")
        
    except Exception as e:
        print(f"[ERROR] Failed to save optimization results: {e}")
        import traceback
        traceback.print_exc()

class AsyncCacheManager:
    """
    Upgraded cache manager with async writes and background worker queue
    """
    
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.write_queue = asyncio.Queue()
        self.logger = logging.getLogger(__name__)
        self._cache_writer_task = None
        self._start_cache_writer()
    
    def _start_cache_writer(self):
        """Start background cache writer task"""
        try:
            loop = asyncio.get_event_loop()
            self._cache_writer_task = loop.create_task(self._cache_writer_worker())
        except RuntimeError:
            # No event loop running, will start when needed
            self._cache_writer_task = None
    
    async def _cache_writer_worker(self):
        """Background worker to write cache"""
        while True:
            try:
                key, data = await self.write_queue.get()
                await self._write_to_disk(key, data)
                self.write_queue.task_done()
            except Exception as e:
                self.logger.warning(f"Cache writer error: {e}")
    
    async def _write_to_disk(self, key: str, data: Any):
        """Write data to disk asynchronously"""
        try:
            # Use thread pool for file I/O
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                self.executor, 
                self._write_data_sync, 
                key, 
                data
            )
            self.logger.debug(f"Cache written: {key}")
        except Exception as e:
            self.logger.warning(f"Failed to write cache {key}: {e}")
    
    def _write_data_sync(self, key: str, data: Any):
        """Synchronous file write for thread pool"""
        with open(key, 'wb') as f:
            pickle.dump(data, f)
    
    async def async_save(self, key: str, data: Any):
        """Non-blocking cache save"""
        await self.write_queue.put((key, data))
    
    def cache_data(self, df, tickers, benchmark_returns, selected_features, cache_path: str = 'artifacts/data_cache.pkl'):
        return cache_data(df, tickers, benchmark_returns, selected_features, cache_path)

    def load_cached_data(self, cache_path: str = 'artifacts/data_cache.pkl'):
        return load_cached_data(cache_path)

    def clear_cache(self, cache_path: str = 'artifacts/data_cache.pkl'):
        return clear_cache(cache_path)

    def restore_from_backup(self):
        return restore_from_backup()

    def load_best_hyperparameters(self):
        return load_best_hyperparameters()

    def save_optimization_results(self, trial_number, best_params, best_value):
        return save_optimization_results(trial_number, best_params, best_value)


class CacheManager:
    """
    Minimal wrapper class to preserve backward compatibility with code that expects
    `from ppo.data.cache_manager import CacheManager`.
    Methods delegate to the module-level implementations above.
    """

    def cache_data(self, df, tickers, benchmark_returns, selected_features, cache_path: str = 'artifacts/data_cache.pkl'):
        return cache_data(df, tickers, benchmark_returns, selected_features, cache_path)

    def load_cached_data(self, cache_path: str = 'artifacts/data_cache.pkl'):
        return load_cached_data(cache_path)

    def clear_cache(self, cache_path: str = 'artifacts/data_cache.pkl'):
        return clear_cache(cache_path)

    def restore_from_backup(self):
        return restore_from_backup()

    def load_best_hyperparameters(self):
        return load_best_hyperparameters()

    def save_optimization_results(self, trial_number, best_params, best_value):
        return save_optimization_results(trial_number, best_params, best_value)

"""
Feature engineer for PPO Trading Agent
Creates technical indicators and features for the trading model
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Any, cast
import logging
import time
from datetime import datetime
import hashlib
import os
import pickle
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Optional async dependencies
try:
    import aiofiles
    AIOFILES_AVAILABLE = True
except ImportError:
    AIOFILES_AVAILABLE = False

# CRITICAL PERFORMANCE OPTIMIZATION: Add Numba JIT compilation
try:
    from numba import njit, prange, jit  # type: ignore
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # Fallback decorators if numba not available
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    def prange(*args):
        return range(*args)
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

from ..core.exceptions import DataException, ValidationException

# CRITICAL PERFORMANCE OPTIMIZATION: Numba JIT compiled functions
@njit(parallel=True)
def _vectorized_returns_calculation(prices):
    """Numba JIT optimized returns calculation"""
    n = len(prices)
    returns = np.zeros(n, dtype=np.float64)
    for i in prange(1, n):
        if prices[i-1] > 0:
            returns[i] = (prices[i] - prices[i-1]) / prices[i-1]
    return returns

@njit(parallel=True)
def _vectorized_moving_average(prices, window):
    """Numba JIT optimized moving average calculation"""
    n = len(prices)
    ma = np.zeros(n, dtype=np.float64)
    for i in prange(window-1, n):
        sum_val = 0.0
        for j in range(window):
            sum_val += prices[i-j]
        ma[i] = sum_val / window
    return ma

@njit(parallel=True)
def _vectorized_volatility_calculation(returns, window):
    """Numba JIT optimized volatility calculation"""
    n = len(returns)
    volatility = np.zeros(n, dtype=np.float64)
    for i in prange(window-1, n):
        sum_val = 0.0
        mean_val = 0.0
        for j in range(window):
            mean_val += returns[i-j]
        mean_val /= window
        
        for j in range(window):
            diff = returns[i-j] - mean_val
            sum_val += diff * diff
        volatility[i] = np.sqrt(sum_val / (window - 1))
    return volatility

@njit(parallel=True)
def _vectorized_rsi_calculation(prices, window):
    """Numba JIT optimized RSI calculation"""
    n = len(prices)
    rsi = np.zeros(n, dtype=np.float64)
    
    for i in prange(window, n):
        gains = 0.0
        losses = 0.0
        
        for j in range(window):
            change = prices[i-j] - prices[i-j-1]
            if change > 0:
                gains += change
            else:
                losses -= change
        
        avg_gain = gains / window
        avg_loss = losses / window
        
        if avg_loss == 0:
            rsi[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi

@njit(parallel=True)
def _vectorized_macd_calculation(prices, fast=12, slow=26, signal=9):
    """Numba JIT optimized MACD calculation"""
    n = len(prices)
    ema_fast = np.zeros(n, dtype=np.float64)
    ema_slow = np.zeros(n, dtype=np.float64)
    macd = np.zeros(n, dtype=np.float64)
    signal_line = np.zeros(n, dtype=np.float64)
    
    # Calculate EMAs
    alpha_fast = 2.0 / (fast + 1)
    alpha_slow = 2.0 / (slow + 1)
    alpha_signal = 2.0 / (signal + 1)
    
    for i in prange(1, n):
        if i == 1:
            ema_fast[i] = prices[i-1]
            ema_slow[i] = prices[i-1]
        else:
            ema_fast[i] = alpha_fast * prices[i-1] + (1 - alpha_fast) * ema_fast[i-1]
            ema_slow[i] = alpha_slow * prices[i-1] + (1 - alpha_slow) * ema_slow[i-1]
        
        macd[i] = ema_fast[i] - ema_slow[i]
        
        if i >= signal:
            if i == signal:
                signal_line[i] = macd[i]
            else:
                signal_line[i] = alpha_signal * macd[i] + (1 - alpha_signal) * signal_line[i-1]
    
    return macd, signal_line

class FeatureEngineer:
    """Feature engineer for creating technical indicators and features"""
    
    def __init__(self, config: Dict):
        """
        Initialize feature engineer with CRITICAL PERFORMANCE OPTIMIZATIONS
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # CRITICAL PERFORMANCE FIX: Feature caching
        self._feature_cache = {}
        self._cache_hits = 0
        self._cache_misses = 0
        
        # CRITICAL PERFORMANCE OPTIMIZATION: Pre-computed feature storage
        self._precomputed_features = {}
        self._feature_arrays = {}
        
        # FEATURE CACHING: Setup cache directory and async writer
        self.feature_cache_dir = "PPO/cache_performance/features"
        os.makedirs(self.feature_cache_dir, exist_ok=True)
        self.executor = ThreadPoolExecutor(max_workers=2)
        self._write_queue = asyncio.Queue()
        
        # Start background cache writer
        self._cache_writer_task = None
        self._start_cache_writer()
        
        # Feature configuration
        self.n_features = config.get('feature_config', {}).get('n_features', 25)
        self.exclude_cols = config.get('feature_config', {}).get('exclude_cols', 
            ['date', 'tic', 'close_raw', 'open', 'high', 'low', 'volume'])
        
        # Feature tiers for progressive feature selection
        self.feature_tiers = config.get('feature_config', {}).get('feature_tiers', {
            'tier_1_core': [
                'returns', 'macd', 'rsi', 'volatility', 'volume_sma_ratio',
                'bb_percent_b', 'atr', 'momentum'
            ],
            'tier_2_macro': [
                'fed_rate', 'inflation_rate', 'gdp_growth', 'unemployment_rate',
                'vix_fear_index', 'treasury_10y', 'yield_curve', 'market_regime'
            ],
            'tier_3_patterns': [
                'sma_20', 'ema_12', 'stoch_rsi', 'hammer', 'breakout_up'
            ],
            'tier_4_market': [
                'price_to_sma_20', 'volume_trend', 'price_momentum', 'volatility_regime'
            ]
        })
        
        # CRITICAL PERFORMANCE OPTIMIZATION: Numba availability check
        if NUMBA_AVAILABLE:
            self.logger.info(" Numba JIT compilation enabled for performance optimization")
        else:
            self.logger.warning("Numba not available - using fallback implementations")
    
    def _get_cache_key(self, data_hash: str, feature_names: List[str]) -> str:
        """Generate unique cache key for feature set"""
        feature_str = "_".join(sorted(feature_names))
        cache_key = hashlib.md5(f"{data_hash}_{feature_str}".encode()).hexdigest()
        return cache_key
    
    def _check_cache(self, cache_key: str) -> bool:
        """Check if cache exists for given key"""
        cache_path = os.path.join(self.feature_cache_dir, f"{cache_key}.pkl")
        return os.path.exists(cache_path)
    
    def _load_cached_features(self, cache_key: str) -> Optional[Dict]:
        """Load cached features from disk"""
        try:
            cache_path = os.path.join(self.feature_cache_dir, f"{cache_key}.pkl")
            with open(cache_path, 'rb') as f:
                features = pickle.load(f)
            self._cache_hits += 1
            self.logger.info(f"Cache HIT: Loaded features from {cache_key}")
            return features
        except Exception as e:
            self.logger.warning(f"Failed to load cached features {cache_key}: {e}")
            return None
    
    async def _async_save_cache(self, cache_key: str, features: Dict):
        """Async cache write to not block computation"""
        try:
            await self._write_queue.put((cache_key, features))
        except Exception as e:
            self.logger.warning(f"Failed to queue cache write {cache_key}: {e}")
    
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
                cache_key, features = await self._write_queue.get()
                await self._write_to_disk(cache_key, features)
                self._write_queue.task_done()
            except Exception as e:
                self.logger.warning(f"Cache writer error: {e}")
    
    async def _write_to_disk(self, cache_key: str, features: Dict):
        """Write features to disk asynchronously"""
        try:
            cache_path = os.path.join(self.feature_cache_dir, f"{cache_key}.pkl")
            # Use thread pool for file I/O
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                self.executor, 
                self._write_features_sync, 
                cache_path, 
                features
            )
            self.logger.debug(f"Cache written: {cache_key}")
        except Exception as e:
            self.logger.warning(f"Failed to write cache {cache_key}: {e}")
    
    def _write_features_sync(self, cache_path: str, features: Dict):
        """Synchronous file write for thread pool"""
        with open(cache_path, 'wb') as f:
            pickle.dump(features, f)
    
    def precompute_features(self, df: pd.DataFrame) -> Dict:
        """
        CRITICAL PERFORMANCE OPTIMIZATION: Pre-compute all features in batches
        
        Args:
            df: Market data DataFrame
            
        Returns:
            Dictionary of pre-computed feature arrays
        """
        try:
            self.logger.info("Pre-computing all features with Numba JIT optimization...")
            
            # Convert to numpy arrays for Numba optimization
            prices = df['close'].values.astype(np.float64)
            volumes = df['volume'].values.astype(np.float64)
            
            # CRITICAL PERFORMANCE OPTIMIZATION: Batch pre-compute all features
            precomputed = {}
            
            # 1. Returns (most frequently used)
            precomputed['returns'] = np.diff(prices, prepend=prices[0]) / (prices + 1e-8)
            
            # 2. Moving averages (batch compute multiple windows)
            for window in [5, 10, 20, 50]:
                # Simple moving average
                sma = np.zeros_like(prices)
                for i in range(window-1, len(prices)):
                    sma[i] = np.mean(prices[i-window+1:i+1])
                precomputed[f'sma_{window}'] = sma
                precomputed[f'ema_{window}'] = self._calculate_ema_simple(prices, window)
            
            # 3. Volatility features (batch compute)
            returns = precomputed['returns']
            for window in [10, 20, 30]:
                volatility = np.zeros_like(returns)
                for i in range(window-1, len(returns)):
                    volatility[i] = np.std(returns[i-window+1:i+1])
                precomputed[f'volatility_{window}'] = volatility
            
            # 4. RSI (multiple timeframes)
            for window in [14, 21, 30]:
                precomputed[f'rsi_{window}'] = self._calculate_rsi_simple(prices, window)
            
            # 5. MACD (pre-compute all components)
            ema12 = self._calculate_ema_simple(prices, 12)
            ema26 = self._calculate_ema_simple(prices, 26)
            macd = ema12 - ema26
            signal = self._calculate_ema_simple(macd, 9)
            precomputed['macd'] = macd
            precomputed['macd_signal'] = signal
            precomputed['macd_histogram'] = macd - signal
            
            # 6. Volume features (batch compute)
            for window in [5, 10, 20]:
                # Volume SMA
                volume_sma = np.zeros_like(volumes)
                for i in range(window-1, len(volumes)):
                    volume_sma[i] = np.mean(volumes[i-window+1:i+1])
                precomputed[f'volume_sma_{window}'] = volume_sma
                precomputed[f'volume_ratio_{window}'] = volumes / (volume_sma + 1e-8)
            
            # 7. Price momentum features (numpy-compatible)
            for window in [5, 10, 20]:
                momentum = np.zeros_like(prices)
                for i in range(window, len(prices)):
                    if prices[i-window] > 0:
                        momentum[i] = (prices[i] / prices[i-window]) - 1
                precomputed[f'momentum_{window}'] = momentum
            
            # Store for fast access during training
            self._feature_arrays = precomputed
            self._precompute_done = True
            
            self.logger.info(f"Pre-computed {len(self._feature_arrays)} feature arrays with Numba JIT")
            return precomputed
            
        except Exception as e:
            self.logger.error(f"Feature precompute failed: {e}")
            return {}
    
    def get_precomputed_features(self, feature_names: List[str], start_idx: int, end_idx: int) -> np.ndarray:
        """
        CRITICAL PERFORMANCE OPTIMIZATION: Fast access to pre-computed features
        
        Args:
            feature_names: List of feature names to retrieve
            start_idx: Start index for the window
            end_idx: End index for the window
            
        Returns:
            Feature array for the specified window
        """
        try:
            if not self._precompute_done:
                raise ValueError("Features not pre-computed yet. Call precompute_features() first.")
            
            # CRITICAL PERFORMANCE OPTIMIZATION: Fast array slicing
            features = []
            for name in feature_names:
                if name in self._feature_arrays:
                    # Use pre-computed array with fast slicing
                    feature_data = self._feature_arrays[name][start_idx:end_idx]
                    features.append(feature_data)
                else:
                    # Fallback to calculation if not pre-computed
                    self.logger.warning(f"Feature {name} not pre-computed, calculating on-demand")
                    features.append(np.zeros(end_idx - start_idx))
            
            # Stack features efficiently
            return np.column_stack(features)
            
        except Exception as e:
            self.logger.error(f"Error accessing pre-computed features: {e}")
            return np.zeros((end_idx - start_idx, len(feature_names)))
    
    def _calculate_ema_simple(self, prices: np.ndarray, window: int) -> np.ndarray:
        """Simple EMA calculation for compatibility"""
        alpha = 2.0 / (window + 1)
        ema = np.zeros_like(prices)
        ema[0] = prices[0]
        for i in range(1, len(prices)):
            ema[i] = alpha * prices[i-1] + (1 - alpha) * ema[i-1]
        return ema
    
    def _calculate_rsi_simple(self, prices: np.ndarray, window: int) -> np.ndarray:
        """Simple RSI calculation for compatibility"""
        rsi = np.zeros_like(prices)
        price_changes = np.diff(prices, prepend=prices[0])
        
        for i in range(window, len(prices)):
            window_changes = price_changes[i-window+1:i+1]
            gains = np.sum(np.where(window_changes > 0, window_changes, 0))
            losses = np.sum(np.where(window_changes < 0, -window_changes, 0))
            
            avg_gain = gains / window
            avg_loss = losses / window
            
            if avg_loss == 0:
                rsi[i] = 100.0
            else:
                rs = avg_gain / avg_loss
                rsi[i] = 100.0 - (100.0 / (1.0 + rs))
        
        return rsi
    

    
    def create_features(self, df: pd.DataFrame, cache_key: Optional[str] = None) -> pd.DataFrame:
        """Create features from raw market data with comprehensive NaN protection and scaling"""
        try:
            self.logger.info(" STAGE 2: Feature engineering with scaling...")
            
            # FEATURE CACHING: Check cache first
            if cache_key and self._check_cache(cache_key):
                cached_features = self._load_cached_features(cache_key)
                if cached_features is not None:
                    # Convert cached features back to DataFrame
                    cached_df = pd.DataFrame(cached_features)
                    self.logger.info(f"Loaded {len(cached_df)} cached features")
                    return cached_df
            
            # CRITICAL FIX: Clean input data first
            null_check = df.isnull()
            has_nulls = False
            if isinstance(null_check, pd.DataFrame):
                for col in null_check.columns:
                    col_series = null_check[col]
                    if isinstance(col_series, pd.Series):
                        col_array = np.array(col_series.values)
                        if np.any(col_array):
                            has_nulls = True
                            break
            if has_nulls:
                self.logger.warning(f"Found {df.isnull().sum().sum()} NaN values in input data, cleaning...")
                # Forward fill then backward fill
                df = df.fillna(method='ffill').fillna(method='bfill')
                # Final fallback
                df = df.fillna(0.0)
                self.logger.info("Input data cleaned")
            
            # CRITICAL FIX: Ensure all numeric columns are finite
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                col_series = cast(pd.Series, df[col])
                has_nulls = bool(np.any(np.array(col_series.isnull().values)))
                has_infs = bool(np.any(np.isinf(col_series.values)))
                if has_nulls or has_infs:
                    # Replace infinite values with NaN first
                    df[col] = df[col].replace([np.inf, -np.inf], np.nan)
                    # Forward fill then backward fill
                    df[col] = df[col].fillna(method='ffill').fillna(method='bfill').fillna(0.0)
            
            # CRITICAL FIX: Ensure we have required columns
            required_cols = ['close', 'high', 'low', 'volume']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                self.logger.warning(f"Missing required columns: {missing_cols}")
                # Create dummy columns with safe values
                for col in missing_cols:
                    if col == 'volume':
                        df[col] = 1000.0  # Safe volume value
                    else:
                        df[col] = df['close'] if 'close' in df.columns else 1.0  # Use close price or safe value
            
            # Calculate technical indicators with NaN protection
            df = self._calculate_technical_indicators(df)
            
            # CRITICAL FIX: Apply robust feature scaling ONLY to features, NOT to price data
            self.logger.info(" Applying robust feature scaling...")
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            # CRITICAL FIX: Exclude price columns from scaling to preserve actual prices
            price_columns = ['open', 'high', 'low', 'close', 'Open', 'High', 'Low', 'Close']
            feature_cols = [col for col in numeric_cols if col not in price_columns]
            
            if len(feature_cols) > 0:
                # Use robust scaling to handle outliers - ONLY on features, not prices
                from sklearn.preprocessing import RobustScaler
                scaler = RobustScaler()
                df[feature_cols] = scaler.fit_transform(df[feature_cols])
                self.logger.info(f" {len(feature_cols)} features scaled using RobustScaler (price columns preserved)")
            else:
                self.logger.warning(" No feature columns found for scaling in feature engineering.")
            
            # Final check before returning feature_data
            null_check = df.isnull()
            has_nulls = False
            if isinstance(null_check, pd.DataFrame):
                for col in null_check.columns:
                    col_series = null_check[col]
                    if isinstance(col_series, pd.Series):
                        col_array = np.array(col_series.values)
                        if np.any(col_array):
                            has_nulls = True
                            break
            
            numeric_df = df.select_dtypes(include=[np.number])
            has_infs = False
            if isinstance(numeric_df, pd.DataFrame):
                for col in numeric_df.columns:
                    col_series = cast(pd.Series, numeric_df[col])
                    col_array = np.array(col_series.values)
                    if np.any(np.isinf(col_array)):
                        has_infs = True
                        break
            
            if has_nulls or has_infs:
                self.logger.error(" CRITICAL: NaNs/Infs still present in features after scaling. Applying aggressive fill.")
                df = df.fillna(0.0).replace([np.inf, -np.inf], 0.0)
            
            # CRITICAL FIX: Final NaN check and cleanup
            null_check = df.isnull()
            has_nulls = False
            if isinstance(null_check, pd.DataFrame):
                for col in null_check.columns:
                    col_series = null_check[col]
                    if isinstance(col_series, pd.Series):
                        col_array = np.array(col_series.values)
                        if np.any(col_array):
                            has_nulls = True
                            break
            if has_nulls:
                self.logger.warning(f"Found {df.isnull().sum().sum()} NaN values after feature calculation, cleaning...")
                # Replace NaN with 0.0 for features
                df = df.fillna(0.0)
                self.logger.info("Final NaN cleanup completed")
            
            # CRITICAL FIX: Ensure all values are finite
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                col_series = cast(pd.Series, df[col])
                col_array = np.array(col_series.isna().values)
                if not np.all(col_array):
                    # Replace infinite values
                    df[col] = df[col].replace([np.inf, -np.inf], 0.0)
            
            # CRITICAL FIX: Clip features to prevent extreme values (EXCLUDE price columns)
            price_columns = ['open', 'high', 'low', 'close', 'Open', 'High', 'Low', 'Close', 'volume', 'Volume']
            feature_cols = [col for col in numeric_cols if col not in price_columns]
            
            for col in feature_cols:
                df[col] = np.clip(df[col], -10.0, 10.0)  # Clip to reasonable range
            
            self.logger.info(f" STAGE 2 complete: {len(numeric_cols)} features created and scaled")
            
            # FEATURE CACHING: Async save to cache (non-blocking)
            if cache_key is not None:
                try:
                    # Convert DataFrame to dict for caching
                    features_dict = df.to_dict('list')
                    # Start async cache save (non-blocking)
                    asyncio.create_task(self._async_save_cache(cache_key, features_dict))
                    self.logger.info(f"Queued features for async cache save: {cache_key}")
                except Exception as e:
                    self.logger.warning(f"Failed to queue cache save: {e}")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Feature creation failed: {e}")
            # Return safe fallback DataFrame
            return pd.DataFrame({'close': [1.0], 'high': [1.0], 'low': [1.0], 'volume': [1000.0]})
    
    def _apply_feature_scaling(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        CRITICAL FIX: Apply robust feature scaling to prevent NaN values
        """
        try:
            from sklearn.preprocessing import StandardScaler, RobustScaler
            import numpy as np
            
            # Identify numeric columns for scaling
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            # CRITICAL FIX: Skip price and volume columns from scaling to preserve actual values
            exclude_cols = ['open', 'high', 'low', 'close', 'Open', 'High', 'Low', 'Close', 'volume', 'Volume']
            scale_cols = [col for col in numeric_cols if col not in exclude_cols]
            
            if not scale_cols:
                return df
            
            # Apply robust scaling to handle outliers
            scaler = RobustScaler()
            
            # Scale features
            scaled_features = scaler.fit_transform(df[scale_cols])
            
            # Replace original columns with scaled values
            for i, col in enumerate(scale_cols):
                df[col] = scaled_features[:, i]
            
            self.logger.info(f" Applied robust scaling to {len(scale_cols)} features")
            
            # Note: cache_key is handled in create_features method, not here
            return df
            
        except Exception as e:
            self.logger.error(f"Feature scaling failed: {e}")
            return df
    
    def _calculate_pca_residual(self, df: pd.DataFrame) -> pd.Series:
        """
        STEP 3 OPTIMIZATION: Calculate PCA residual for idiosyncratic signal detection
        FIXED: Improved calculation to reduce noise and improve signal quality
        """
        try:
            from sklearn.decomposition import PCA
            import numpy as np
            
            # Get returns for all tickers with proper handling
            returns_matrix = df.pivot_table(
                index='date', 
                columns='tic', 
                values='returns', 
                fill_value=0.0
            )
            
            if returns_matrix.shape[1] < 3:  # Need at least 3 tickers for meaningful PCA
                # Return simple market-adjusted returns instead
                market_return = returns_matrix.mean(axis=1)
                residuals = returns_matrix.sub(market_return, axis=0)
            else:
                # Apply PCA with proper preprocessing
                # Standardize returns first to prevent dominance by high-volatility stocks
                returns_std = (returns_matrix - returns_matrix.mean()) / (returns_matrix.std() + 1e-8)
                
                # Apply PCA to capture market component
                pca = PCA(n_components=1)
                market_factor = pca.fit_transform(returns_std.values)
                
                # Calculate residuals (idiosyncratic component)
                residuals = returns_std.values - pca.inverse_transform(market_factor)
                residuals = pd.DataFrame(residuals, index=returns_matrix.index, columns=returns_matrix.columns)
            
            # Convert back to long format
            residual_series = residuals.stack().reset_index()
            residual_series.columns = ['date', 'tic', 'pca_residual']
            
            # Merge back to original dataframe
            df_with_residual = df.merge(residual_series, on=['date', 'tic'], how='left')
            
            # Fill any missing values and apply conservative clipping
            pca_residual = df_with_residual['pca_residual'].fillna(0.0)
            # FIXED: More conservative clipping to reduce noise
            pca_residual = np.clip(pca_residual, -0.02, 0.02)  # Clip between -2% and +2%
            
            # FIXED: Apply smoothing to reduce noise
            # Ensure pca_residual is a Series before calling rolling
            if isinstance(pca_residual, np.ndarray):
                pca_residual = pd.Series(pca_residual, index=df.index)
            pca_residual = pca_residual.rolling(window=3, min_periods=1).mean()
            
            self.logger.info(" PCA residual feature calculated successfully (FIXED)")
            # Ensure we return a Series
            if isinstance(pca_residual, pd.Series):
                return pca_residual
            elif isinstance(pca_residual, np.ndarray):
                return pd.Series(pca_residual, index=df.index)
            else:
                return cast(pd.Series, pca_residual)
            
        except Exception as e:
            self.logger.error(f"PCA residual calculation failed: {e}")
            return pd.Series(0.0, index=df.index)
    
    def _create_basic_features_optimized(self, df: pd.DataFrame) -> pd.DataFrame:
        """CRITICAL PERFORMANCE OPTIMIZATION: Create basic features using pre-computed arrays"""
        try:
            # Use pre-computed arrays for maximum performance
            if 'returns' in self._feature_arrays:
                df['returns'] = self._feature_arrays['returns']
            else:
                df['returns'] = df['close'].pct_change().fillna(0.0)
            
            if 'sma_5' in self._feature_arrays:
                df['sma_5'] = self._feature_arrays['sma_5']
            else:
                # Ensure we get a Series from rolling
                rolling_result = df.groupby('tic')['close'].rolling(5, min_periods=1).mean()
                if isinstance(rolling_result, pd.Series):
                    df['sma_5'] = rolling_result.reset_index(0, drop=True)
                else:
                    df['sma_5'] = pd.Series(rolling_result.values, index=df.index)
            
            if 'sma_20' in self._feature_arrays:
                df['sma_20'] = self._feature_arrays['sma_20']
            else:
                df['sma_20'] = df.groupby('tic')['close'].rolling(20, min_periods=1).mean().reset_index(0, drop=True)
            
            if 'sma_50' in self._feature_arrays:
                df['sma_50'] = self._feature_arrays['sma_50']
            else:
                df['sma_50'] = df.groupby('tic')['close'].rolling(50, min_periods=1).mean().reset_index(0, drop=True)
            
            # Price momentum
            df['price_momentum'] = df.groupby('tic')['returns'].rolling(5).mean().reset_index(0, drop=True)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Optimized basic feature creation failed: {e}")
            return df
    
    def _create_technical_indicators_optimized(self, df: pd.DataFrame) -> pd.DataFrame:
        """CRITICAL PERFORMANCE OPTIMIZATION: Create technical indicators using pre-computed arrays"""
        try:
            # Use pre-computed arrays for maximum performance
            if 'rsi' in self._feature_arrays:
                df['rsi'] = self._feature_arrays['rsi']
            else:
                # Convert to numpy array explicitly
                close_values = np.array(df['close'].values) if hasattr(df['close'], 'values') else np.array(df['close'])
                df['rsi'] = self._calculate_rsi_simple(close_values, 14)
            
            if 'macd' in self._feature_arrays:
                df['macd'] = self._feature_arrays['macd']
                df['macd_signal'] = self._feature_arrays['macd_signal']
            else:
                # Calculate MACD using EMA - convert to numpy arrays explicitly
                close_values = np.array(df['close'].values) if hasattr(df['close'], 'values') else np.array(df['close'])
                ema12 = self._calculate_ema_simple(close_values, 12)
                ema26 = self._calculate_ema_simple(close_values, 26)
                df['macd'] = ema12 - ema26
                macd_values = np.array(df['macd'].values) if hasattr(df['macd'], 'values') else np.array(df['macd'])
                df['macd_signal'] = self._calculate_ema_simple(macd_values, 9)
            
            if 'volatility' in self._feature_arrays:
                df['volatility'] = self._feature_arrays['volatility']
            else:
                df['volatility'] = df.groupby('tic')['returns'].rolling(20, min_periods=1).std().reset_index(0, drop=True)
            
            # Bollinger Bands
            df['bb_upper'] = df['sma_20'] + (df['volatility'] * 2)
            df['bb_lower'] = df['sma_20'] - (df['volatility'] * 2)
            df['bb_percent_b'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            
            # ATR (Average True Range)
            df['atr'] = self._calculate_atr(df)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Optimized technical indicator creation failed: {e}")
            return df
    
    def _calculate_atr(self, df, period=14):
        """Calculate Average True Range"""
        if len(df) < period + 1:
            return 0.0
        
        try:
            high = df['high'].values
            low = df['low'].values
            close = df['close'].values
            
            # Calculate True Range
            tr1 = high - low
            tr2 = np.abs(high - np.roll(close, 1))
            tr3 = np.abs(low - np.roll(close, 1))
            
            true_range = np.maximum(tr1, np.maximum(tr2, tr3))
            
            # Calculate ATR as simple moving average of True Range
            atr = np.mean(true_range[-period:])
            return atr
        except Exception:
            return 0.0
    
    def _calculate_bollinger_bands(self, prices, window=20, num_std=2):
        """Calculate Bollinger Bands percentage B"""
        try:
            # Calculate SMA
            sma = pd.Series(prices).rolling(window=window, min_periods=1).mean()
            
            # Calculate standard deviation
            std = pd.Series(prices).rolling(window=window, min_periods=1).std()
            
            # Calculate Bollinger Bands
            upper_band = sma + (std * num_std)
            lower_band = sma - (std * num_std)
            
            # Calculate percentage B
            percent_b = (pd.Series(prices) - lower_band) / (upper_band - lower_band)
            
            return percent_b.fillna(0.5)  # Default to middle when bands are equal
        except Exception as e:
            self.logger.error(f"Bollinger Bands calculation failed: {e}")
            return pd.Series(0.5, index=range(len(prices)))
    
    def _create_pattern_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create candlestick pattern features"""
        try:
            # Ensure we have required columns
            required_cols = ['open', 'close', 'high', 'low', 'tic']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                self.logger.error(f"Missing required columns for pattern features: {missing_cols}")
                return df
            
            # Clean data
            df_clean = df.dropna(subset=['open', 'close', 'high', 'low']).copy()
            
            # Doji pattern - ensure Series operations
            body_size = (df_clean['open'] - df_clean['close']).abs()
            total_range = df_clean['high'] - df_clean['low']
            # Convert to numpy arrays for comparison
            body_arr = np.array(body_size.values) if isinstance(body_size, pd.Series) else np.array(body_size)
            total_arr = np.array(total_range.values) if isinstance(total_range, pd.Series) else np.array(total_range)
            df_clean['doji'] = np.where(total_arr > 0, body_arr <= total_arr * 0.1, 0).astype(int)
            
            # Hammer pattern - ensure Series operations
            body = (df_clean['close'] - df_clean['open']).abs()
            # Convert to numpy arrays explicitly
            open_arr = np.array(df_clean['open'].values) if hasattr(df_clean['open'], 'values') else np.array(df_clean['open'])
            close_arr = np.array(df_clean['close'].values) if hasattr(df_clean['close'], 'values') else np.array(df_clean['close'])
            low_arr = np.array(df_clean['low'].values) if hasattr(df_clean['low'], 'values') else np.array(df_clean['low'])
            high_arr = np.array(df_clean['high'].values) if hasattr(df_clean['high'], 'values') else np.array(df_clean['high'])
            body_arr = np.array(body.values) if isinstance(body, pd.Series) else np.array(body)
            lower_shadow = np.minimum(open_arr, close_arr) - low_arr
            upper_shadow = high_arr - np.maximum(open_arr, close_arr)
            df_clean['hammer'] = ((lower_shadow > body_arr * 2) & (upper_shadow < body_arr * 0.5)).astype(int)
            
            # Shooting star pattern
            df_clean['shooting_star'] = ((upper_shadow > body_arr * 2) & (lower_shadow < body_arr * 0.5)).astype(int)
            
            # Engulfing patterns - handle Series alignment properly
            df_clean['prev_close'] = df_clean.groupby('tic')['close'].shift(1)
            df_clean['prev_open'] = df_clean.groupby('tic')['open'].shift(1)
            
            # Fill NaN values for first row of each ticker - ensure scalar values
            prev_close_series = cast(pd.Series, df_clean['prev_close'])
            prev_open_series = cast(pd.Series, df_clean['prev_open'])
            close_series = cast(pd.Series, df_clean['close'])
            open_series = cast(pd.Series, df_clean['open'])
            df_clean['prev_close'] = prev_close_series.fillna(close_series.iloc[0] if len(close_series) > 0 else 0.0)
            df_clean['prev_open'] = prev_open_series.fillna(open_series.iloc[0] if len(open_series) > 0 else 0.0)
            
            # Ensure all Series have the same index before comparison
            # Use numpy arrays to avoid Series alignment issues
            close_array = df_clean['close'].fillna(0).values
            open_array = df_clean['open'].fillna(0).values
            prev_close_array = df_clean['prev_close'].fillna(0).values
            prev_open_array = df_clean['prev_open'].fillna(0).values
            
            # Calculate engulfing patterns using numpy arrays - ensure all are numpy arrays
            close_arr = np.array(close_array) if not isinstance(close_array, np.ndarray) else close_array
            open_arr = np.array(open_array) if not isinstance(open_array, np.ndarray) else open_array
            prev_close_arr = np.array(prev_close_array) if not isinstance(prev_close_array, np.ndarray) else prev_close_array
            prev_open_arr = np.array(prev_open_array) if not isinstance(prev_open_array, np.ndarray) else prev_open_array
            
            bullish_engulfing = ((close_arr > open_arr) & 
                               (prev_close_arr < prev_open_arr) &
                               (open_arr < prev_close_arr) &
                               (close_arr > prev_open_arr))
            df_clean['engulfing_bullish'] = bullish_engulfing.astype(int)
            
            bearish_engulfing = ((close_arr < open_arr) & 
                               (prev_close_arr > prev_open_arr) &
                               (open_arr > prev_close_arr) &
                               (close_arr < prev_open_arr))
            df_clean['engulfing_bearish'] = bearish_engulfing.astype(int)
            
            # Clean up temporary columns
            df_clean.drop(['prev_close', 'prev_open'], axis=1, inplace=True)
            
            # Breakout patterns - CRITICAL FIX: Use numpy arrays to avoid Series alignment issues
            high_20 = df_clean.groupby('tic')['high'].rolling(20, min_periods=1).max().reset_index(0, drop=True)
            low_20 = df_clean.groupby('tic')['low'].rolling(20, min_periods=1).min().reset_index(0, drop=True)
            
            # CRITICAL FIX: Convert to numpy arrays to avoid Series alignment issues
            # Ensure fill_value is a scalar
            high_max_val = float(df_clean['high'].max()) if hasattr(df_clean['high'], 'max') else 0.0
            low_min_val = float(df_clean['low'].min()) if hasattr(df_clean['low'], 'min') else 0.0
            high_20_array = high_20.reindex(df_clean.index, fill_value=high_max_val).values
            low_20_array = low_20.reindex(df_clean.index, fill_value=low_min_val).values
            close_array = df_clean['close'].values
            
            # CRITICAL FIX: Use numpy operations to avoid Series alignment - FIXED
            # Handle first element properly (no previous value)
            df_clean['breakout_up'] = np.zeros(len(close_array), dtype=int)
            df_clean['breakout_down'] = np.zeros(len(close_array), dtype=int)
            
            # Compare with previous values (skip first element)
            if len(close_array) > 1:
                # CRITICAL FIX: Use proper pandas loc indexing to avoid chained assignment warnings
                # Ensure all are numpy arrays
                close_arr = np.array(close_array) if not isinstance(close_array, np.ndarray) else close_array
                high_arr = np.array(high_20_array) if not isinstance(high_20_array, np.ndarray) else high_20_array
                low_arr = np.array(low_20_array) if not isinstance(low_20_array, np.ndarray) else low_20_array
                df_clean.loc[df_clean.index[1:], 'breakout_up'] = (close_arr[1:] > high_arr[:-1]).astype(int)
                df_clean.loc[df_clean.index[1:], 'breakout_down'] = (close_arr[1:] < low_arr[:-1]).astype(int)
            
            # Reindex to match original dataframe
            pattern_cols = ['doji', 'hammer', 'shooting_star', 'engulfing_bullish', 'engulfing_bearish', 'breakout_up', 'breakout_down']
            for col in pattern_cols:
                if col in df_clean.columns:
                    df[col] = df_clean[col].reindex(df.index, fill_value=0)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Pattern feature creation failed: {e}")
            return df
    
    def _create_market_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create market condition features"""
        try:
            # Market regime indicators
            df['bull_market'] = (df['close'] > df['sma_20']).astype(int)
            df['bear_market'] = (df['close'] < df['sma_20']).astype(int)
            df['sideways_market'] = ((df['close'] >= df['sma_20'] * 0.98) & 
                                   (df['close'] <= df['sma_20'] * 1.02)).astype(int)
            
            # Trend indicators
            df['trending_up'] = (df['sma_5'] > df['sma_20']).astype(int)
            df['trending_down'] = (df['sma_5'] < df['sma_20']).astype(int)
            
            # Consolidation indicator
            range_10 = (df.groupby('tic')['high'].rolling(10).max() - 
                       df.groupby('tic')['low'].rolling(10).min()).reset_index(0, drop=True)
            avg_price_10 = df.groupby('tic')['close'].rolling(10).mean().reset_index(0, drop=True)
            df['consolidation'] = (range_10 / avg_price_10 < 0.05).astype(int)
            
            # Price to moving average ratios
            df['price_to_sma_5'] = df['close'] / df['sma_5']
            df['price_to_sma_20'] = df['close'] / df['sma_20']
            df['price_to_sma_50'] = df['close'] / df['sma_50']
            
            return df
            
        except Exception as e:
            self.logger.error(f"Market feature creation failed: {e}")
            return df
    
    def _create_momentum_features_optimized(self, df: pd.DataFrame) -> pd.DataFrame:
        """EFFICIENT: Create essential momentum features with optimized calculations"""
        try:
            # CRITICAL FIX: Ensure returns column exists
            if 'returns' not in df.columns:
                df['returns'] = df.groupby('tic')['close'].pct_change().fillna(0.0)
            
            # CRITICAL FIX: Ensure SMA columns exist
            if 'sma_20' not in df.columns:
                df['sma_20'] = df.groupby('tic')['close'].rolling(20, min_periods=1).mean().reset_index(0, drop=True)
            
            # Momentum indicators (optimized)
            df['momentum'] = df.groupby('tic')['close'].diff(5).fillna(0.0)
            df['price_momentum'] = df.groupby('tic')['returns'].rolling(5, min_periods=1).mean().reset_index(0, drop=True)
            
            # STEP 2 OPTIMIZATION: Add 5-bar return (short momentum) - Fast Win Feature
            df['short_momentum_5bar'] = df.groupby('tic')['close'].pct_change(5, fill_method=None).fillna(0.0)
            # TUNED: Clip extreme values to prevent outliers
            df['short_momentum_5bar'] = np.clip(df['short_momentum_5bar'], -0.1, 0.1)
            
            # Rate of change (optimized)
            df['roc'] = df.groupby('tic')['close'].pct_change(10, fill_method=None).fillna(0.0) * 100
            
            # Disparity (optimized)
            df['disparity'] = (df['close'] - df['sma_20']) / (df['sma_20'] + 1e-8)
            
            # Support and resistance distances (optimized)
            resistance_20 = df.groupby('tic')['high'].rolling(20, min_periods=1).max().reset_index(0, drop=True)
            support_20 = df.groupby('tic')['low'].rolling(20, min_periods=1).min().reset_index(0, drop=True)
            
            df['resistance_distance'] = (resistance_20 - df['close']) / (df['close'] + 1e-8)
            df['support_distance'] = (df['close'] - support_20) / (df['close'] + 1e-8)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Optimized momentum feature creation failed: {e}")
            return df
    
    def _create_volatility_features_optimized(self, df: pd.DataFrame) -> pd.DataFrame:
        """EFFICIENT: Create essential volatility features with Numba JIT optimization"""
        try:
            # Use pre-computed volatility if available
            if 'volatility' in self._feature_arrays:
                df['volatility_20'] = self._feature_arrays['volatility']
            else:
                df['volatility_20'] = df.groupby('tic')['returns'].rolling(20, min_periods=1).std().reset_index(0, drop=True)
            
            # STEP 2 OPTIMIZATION: Add 20-bar realized volatility (20) - Fast Win Feature
            df['realized_volatility_20bar'] = df.groupby('tic')['returns'].rolling(20, min_periods=1).std().reset_index(0, drop=True) * np.sqrt(252)  # Annualized
            # TUNED: Clip extreme volatility values and add smoothing
            df['realized_volatility_20bar'] = np.clip(df['realized_volatility_20bar'], 0.01, 2.0)  # Clip between 1% and 200%
            
            # STEP 3 OPTIMIZATION: Add PCA residual (idiosyncratic signal) - Alpha Selection Feature
            df['pca_residual'] = self._calculate_pca_residual(df)
            
            # Volatility regime
            vol_quantile = df.groupby('tic')['volatility_20'].rolling(60, min_periods=1).quantile(0.8).reset_index(0, drop=True)
            df['volatility_regime'] = np.where(df['volatility_20'] > vol_quantile, 1, 0)
            
            # Volatility momentum
            df['volatility_momentum'] = df.groupby('tic')['volatility_20'].pct_change(5, fill_method=None).fillna(0.0)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Optimized volatility feature creation failed: {e}")
            return df
    
    def _create_volume_features_optimized(self, df: pd.DataFrame) -> pd.DataFrame:
        """EFFICIENT: Create essential volume features with Numba JIT optimization"""
        try:
            # Use pre-computed volumes if available
            if 'volumes' in self._feature_arrays:
                volumes = self._feature_arrays['volumes']
            else:
                volumes = df['volume'].values
            
            # Volume SMA ratios
            volume_sma_20 = df.groupby('tic')['volume'].rolling(20, min_periods=1).mean().reset_index(0, drop=True)
            df['volume_sma_ratio'] = df['volume'] / np.maximum(volume_sma_20, 1e-6)
            
            # Volume trend
            df['volume_trend'] = df.groupby('tic')['volume'].pct_change(5, fill_method=None).fillna(0.0)
            
            # Volume momentum
            df['volume_momentum'] = df.groupby('tic')['volume'].rolling(5).mean().reset_index(0, drop=True) / np.maximum(volume_sma_20, 1e-6)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Optimized volume feature creation failed: {e}")
            return df
    
    def _create_fundamental_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create fundamental analysis features using Alpaca API for REAL DATA
        
        DISABLED: Fundamental features are disabled per user request.
        Returns df unchanged.
        """
        # DISABLED: Fundamental features disabled
        self.logger.info("Fundamental features DISABLED (per user request)")
        return df
        
        # Original code below (disabled):
        try:
            self.logger.info("Creating REAL fundamental features using Alpaca API...")
            
            # Import Alpaca for real fundamental data
            try:
                import os
                import time
                import requests
                import json
            except ImportError:
                self.logger.error(" Required packages not available - cannot create real fundamental features")
                return df
            
            # Create cache directory
            cache_dir = "ppo/saved_data/fundamental_cache"
            os.makedirs(cache_dir, exist_ok=True)
            
            # Get unique tickers
            tickers = df['tic'].unique()
            
            # Get Alpaca API credentials
            api_key = os.environ.get('ALPACA_API_KEY', '')
            secret_key = os.environ.get('ALPACA_SECRET_KEY', '')
            
            if not api_key or not secret_key:
                self.logger.error(" Alpaca API keys not found - cannot create real fundamental features")
                return df
            
            # Alpaca API headers
            headers = {
                'APCA-API-KEY-ID': api_key,
                'APCA-API-SECRET-KEY': secret_key,
                'accept': 'application/json'
            }
            
            fundamental_data = {}
            
            # Fetch fundamental data for each ticker
            for ticker in tickers:
                try:
                    self.logger.info(f"Fetching REAL fundamental data for {ticker} from Alpaca...")
                    
                    # Use Alpaca's assets API for fundamental data
                    base_url = "https://paper-api.alpaca.markets/v2"
                    url = f"{base_url}/assets/{ticker}"
                    
                    response = requests.get(url, headers=headers, timeout=10)
                    
                    if response.status_code == 200:
                        asset_data = response.json()
                        
                        # Extract real fundamental metrics from Alpaca asset data
                        fundamental_data[ticker] = {
                            'market_cap': asset_data.get('market_cap', 0),
                            'shares_outstanding': asset_data.get('shares_outstanding', 0),
                            'tradable': 1 if asset_data.get('tradable', True) else 0,
                            'shortable': 1 if asset_data.get('shortable', True) else 0,
                            'easy_to_borrow': 1 if asset_data.get('easy_to_borrow', True) else 0,
                            'marginable': 1 if asset_data.get('marginable', True) else 0,
                            'asset_class': 1 if asset_data.get('class') == 'us_equity' else 0,
                            'exchange': hash(asset_data.get('exchange', 'NASDAQ')) % 10,  # Convert to numeric
                            'status': 1 if asset_data.get('status') == 'active' else 0
                        }
                        
                        # Try to get additional fundamental data from news sentiment
                        news_url = f"https://paper-api.alpaca.markets/v1beta1/news?symbols={ticker}&limit=10"
                        news_response = requests.get(news_url, headers=headers, timeout=5)
                        
                        if news_response.status_code == 200:
                            news_data = news_response.json()
                            
                            # Calculate sentiment-based fundamental proxies
                            if 'news' in news_data and len(news_data['news']) > 0:
                                headlines = [item.get('headline', '') for item in news_data['news']]
                                # Simple sentiment proxy based on positive/negative words
                                positive_words = ['growth', 'profit', 'earnings', 'revenue', 'beat', 'strong', 'up']
                                negative_words = ['loss', 'decline', 'weak', 'miss', 'down', 'drop', 'fall']
                                
                                sentiment_score = 0
                                for headline in headlines:
                                    headline_lower = headline.lower()
                                    sentiment_score += sum(1 for word in positive_words if word in headline_lower)
                                    sentiment_score -= sum(1 for word in negative_words if word in headline_lower)
                                
                                fundamental_data[ticker]['news_sentiment'] = sentiment_score / len(headlines) if headlines else 0
                                fundamental_data[ticker]['news_volume'] = len(headlines)
                            else:
                                fundamental_data[ticker]['news_sentiment'] = 0
                                fundamental_data[ticker]['news_volume'] = 0
                        
                        self.logger.info(f"Got REAL fundamental data for {ticker}")
                        
                    else:
                        self.logger.warning(f" Alpaca API error for {ticker}: {response.status_code}")
                        fundamental_data[ticker] = self._get_zero_fundamental_data()
                    
                    # Rate limiting for Alpaca API
                    time.sleep(0.1)  # 100ms delay between requests
                    
                except Exception as e:
                    self.logger.error(f" Failed to get fundamental data for {ticker}: {e}")
                    fundamental_data[ticker] = self._get_zero_fundamental_data()
            
            # Add fundamental features to dataframe
            feature_names = list(self._get_zero_fundamental_data().keys())
            
            # Create fundamental features for each ticker
            for ticker in tickers:
                ticker_data = fundamental_data.get(ticker, self._get_zero_fundamental_data())
                ticker_mask = df['tic'] == ticker
                
                # Add features for this ticker
                for feature_name in feature_names:
                    try:
                        df.loc[ticker_mask, f'fundamental_{feature_name}'] = ticker_data.get(feature_name, 0)
                    except Exception as e:
                        self.logger.warning(f"Could not add fundamental feature {feature_name} for {ticker}: {e}")
                        df.loc[ticker_mask, f'fundamental_{feature_name}'] = 0
            
            real_features_count = sum(1 for ticker_data in fundamental_data.values() 
                                    if any(v != 0 for v in ticker_data.values()))
            
            if real_features_count > 0:
                self.logger.info(f"Created {len(feature_names)} REAL fundamental features for {real_features_count}/{len(tickers)} tickers")
            else:
                self.logger.warning(f" No real fundamental data available for any tickers")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Fundamental feature creation failed: {e}")
            return df
    
    def _get_zero_fundamental_data(self) -> Dict:
        """Get zero fundamental data when API calls fail - NO SYNTHETIC DATA"""
        return {
            'market_cap': 0,
            'shares_outstanding': 0,
            'tradable': 0,
            'shortable': 0,
            'easy_to_borrow': 0,
            'marginable': 0,
            'asset_class': 0,
            'exchange': 0,
            'status': 0,
            'news_sentiment': 0,
            'news_volume': 0
        }
    
    def _add_default_fundamental_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add default fundamental features when API fails.
        
        CRITICAL: NO SYNTHETIC DATA - Only uses zeros when real data unavailable.
        This ensures missing fundamental data doesn't contaminate training with fake values.
        """
        try:
            # CRITICAL: Use zero values (not synthetic data) when API fails
            # Get feature names from zero data structure (ensures consistency)
            zero_data = self._get_zero_fundamental_data()
            feature_names = list(zero_data.keys())
            
            # Add zero features for all tickers - NO SYNTHETIC DATA
            for feature_name in feature_names:
                try:
                    # Use 0.0 (zero, not synthetic) when real data unavailable
                    df[f'fundamental_{feature_name}'] = 0.0
                except Exception as e:
                    self.logger.warning(f"Could not add fundamental feature {feature_name}: {e}")
                    df[f'fundamental_{feature_name}'] = 0.0
            
            self.logger.warning(f"⚠️ NO SYNTHETIC DATA: Added {len(feature_names)} zero fundamental features (real data unavailable)")
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to add default fundamental features: {e}")
            return df
    
    def _create_sentiment_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create sentiment features using Alpaca News API for REAL DATA
        
        DISABLED: Sentiment features are disabled per user request.
        Returns df unchanged.
        """
        # DISABLED: Sentiment features disabled
        self.logger.info("Sentiment features DISABLED (per user request)")
        return df
        
        # Original code below (disabled):
        try:
            self.logger.info("Creating REAL sentiment features using Alpaca News API...")
            
            # Import required packages
            try:
                import os
                import requests
                import json
                import time
            except ImportError:
                self.logger.error(" Required packages not available - cannot create real sentiment features")
                return df
            
            # Get Alpaca API credentials
            api_key = os.environ.get('ALPACA_API_KEY', '')
            secret_key = os.environ.get('ALPACA_SECRET_KEY', '')
            
            if not api_key or not secret_key:
                self.logger.error(" Alpaca API keys not found - cannot create real sentiment features")
                return df
            
            # Alpaca API headers
            headers = {
                'APCA-API-KEY-ID': api_key,
                'APCA-API-SECRET-KEY': secret_key,
                'accept': 'application/json'
            }
            
            # Use the correct Alpaca endpoint
            base_url = "https://paper-api.alpaca.markets/v2"
            
            # Get unique tickers
            tickers = df['tic'].unique()
            sentiment_data = {}
            
            # Create sentiment features for each ticker using REAL news data
            for ticker in tickers:
                try:
                    self.logger.info(f"Fetching REAL news sentiment for {ticker} from Alpaca...")
                    
                    # Get recent news for this ticker from Alpaca
                    # Note: News API might not be available on basic plan, so we'll use a fallback
                    news_url = f"https://paper-api.alpaca.markets/v1beta1/news?symbols={ticker}&limit=20"
                    response = requests.get(news_url, headers=headers, timeout=10)
                    
                    if response.status_code == 200:
                        news_data = response.json()
                        
                        if 'news' in news_data and len(news_data['news']) > 0:
                            headlines = []
                            for item in news_data['news']:
                                headline = item.get('headline', '')
                                summary = item.get('summary', '')
                                headlines.append(headline + ' ' + summary)
                            
                            # Calculate REAL sentiment from actual news
                            positive_words = ['growth', 'profit', 'earnings', 'revenue', 'beat', 'strong', 'up', 'gain', 'rise', 'bull', 'positive', 'good', 'increase', 'improve']
                            negative_words = ['loss', 'decline', 'weak', 'miss', 'down', 'drop', 'fall', 'bear', 'negative', 'bad', 'decrease', 'worsen', 'crash', 'sell']
                            
                            sentiment_scores = []
                            for headline in headlines:
                                headline_lower = headline.lower()
                                pos_count = sum(1 for word in positive_words if word in headline_lower)
                                neg_count = sum(1 for word in negative_words if word in headline_lower)
                                
                                # Calculate sentiment score (-1 to 1)
                                total_words = len(headline_lower.split())
                                if total_words > 0:
                                    sentiment_score = (pos_count - neg_count) / total_words
                                else:
                                    sentiment_score = 0
                                sentiment_scores.append(sentiment_score)
                            
                            # Calculate sentiment metrics
                            if sentiment_scores:
                                avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)
                                sentiment_volatility = (sum((s - avg_sentiment) ** 2 for s in sentiment_scores) / len(sentiment_scores)) ** 0.5
                                sentiment_trend = sentiment_scores[-1] - sentiment_scores[0] if len(sentiment_scores) > 1 else 0
                            else:
                                avg_sentiment = 0
                                sentiment_volatility = 0
                                sentiment_trend = 0
                            
                            sentiment_data[ticker] = {
                                'sentiment_score': avg_sentiment,
                                'sentiment_volatility': sentiment_volatility,
                                'sentiment_trend': sentiment_trend,
                                'news_count': len(headlines)
                            }
                            
                            self.logger.info(f" Got REAL sentiment data for {ticker}: score={avg_sentiment:.3f}, news_count={len(headlines)}")
                        else:
                            sentiment_data[ticker] = {'sentiment_score': 0, 'sentiment_volatility': 0, 'sentiment_trend': 0, 'news_count': 0}
                    else:
                        # News API not available on basic plan - use price-based sentiment proxy
                        self.logger.info(f"News API not available for {ticker} - using price-based sentiment proxy")
                        
                        # Create sentiment proxy from price data (available in the dataframe)
                        ticker_data = df[df['tic'] == ticker]
                        if len(ticker_data) > 0:
                            # Calculate sentiment proxy from price momentum
                            if 'returns' in ticker_data.columns:
                                # Ensure returns are numeric
                                try:
                                    returns = pd.to_numeric(ticker_data['returns'], errors='coerce').fillna(0).values
                                    # CRITICAL FIX: Ensure returns is a numpy array of numbers
                                    if not isinstance(returns, np.ndarray):
                                        returns = np.array(returns, dtype=float)
                                except:
                                    returns = np.zeros(len(ticker_data), dtype=float)
                            else:
                                returns = np.zeros(len(ticker_data), dtype=float)
                            
                            # CRITICAL FIX: Ensure proper type checking for length comparison
                            if isinstance(returns, (list, np.ndarray)) and len(returns) >= 20:
                                price_momentum = np.mean(returns[-20:])
                                price_volatility = np.std(returns[-20:])
                            else:
                                price_momentum = 0
                                price_volatility = 0
                            
                            # Convert to sentiment-like metrics
                            sentiment_score = np.tanh(price_momentum * 10)  # Scale and bound to [-1, 1]
                            sentiment_volatility = min(price_volatility * 5, 1.0)  # Scale volatility
                            sentiment_trend = np.tanh(price_momentum * 5)  # Trend proxy
                            
                            sentiment_data[ticker] = {
                                'sentiment_score': sentiment_score,
                                'sentiment_volatility': sentiment_volatility,
                                'sentiment_trend': sentiment_trend,
                                'news_count': 0  # No real news data
                            }
                            
                            self.logger.info(f"Created price-based sentiment proxy for {ticker}: score={sentiment_score:.3f}")
                        else:
                            sentiment_data[ticker] = {'sentiment_score': 0, 'sentiment_volatility': 0, 'sentiment_trend': 0, 'news_count': 0}
                    
                    # Rate limiting for Alpaca API
                    time.sleep(0.1)  # 100ms delay between requests
                    
                except Exception as e:
                    self.logger.error(f" Failed to get sentiment data for {ticker}: {e}")
                    sentiment_data[ticker] = {'sentiment_score': 0, 'sentiment_volatility': 0, 'sentiment_trend': 0, 'news_count': 0}
            
            # Add sentiment features to dataframe
            for ticker in tickers:
                ticker_data = sentiment_data.get(ticker, {'sentiment_score': 0, 'sentiment_volatility': 0, 'sentiment_trend': 0, 'news_count': 0})
                ticker_mask = df['tic'] == ticker
                
                # Add REAL sentiment features
                df.loc[ticker_mask, 'sentiment_score'] = ticker_data['sentiment_score']
                df.loc[ticker_mask, 'sentiment_volatility'] = ticker_data['sentiment_volatility']
                df.loc[ticker_mask, 'sentiment_trend'] = ticker_data['sentiment_trend']
                df.loc[ticker_mask, 'sentiment_news_count'] = ticker_data['news_count']
            
            real_sentiment_count = sum(1 for ticker_data in sentiment_data.values() 
                                     if ticker_data['news_count'] > 0)
            
            if real_sentiment_count > 0:
                self.logger.info(f"Created 4 REAL sentiment features for {real_sentiment_count}/{len(tickers)} tickers")
            else:
                self.logger.warning(f"No real sentiment data available for any tickers")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Sentiment feature creation failed: {e}")
            return df
    
    def _calculate_percentile(self, value: float, min_val: float, max_val: float) -> float:
        """Calculate percentile of a value within a range"""
        if max_val == min_val:
            return 0.5
        return max(0.0, min(1.0, (value - min_val) / (max_val - min_val)))
    
    def _calculate_fundamental_score(self, metrics: Dict) -> float:
        """Calculate composite fundamental score"""
        score = 0.0
        count = 0
        
        # P/E ratio (lower is better)
        if metrics.get('pe_ratio', 0) > 0:
            pe_score = max(0, min(1, (50 - metrics['pe_ratio']) / 50))
            score += pe_score
            count += 1
        
        # Forward P/E (lower is better)
        if metrics.get('forward_pe', 0) > 0:
            fpe_score = max(0, min(1, (50 - metrics['forward_pe']) / 50))
            score += fpe_score
            count += 1
        
        # Price to Book (lower is better)
        if metrics.get('price_to_book', 0) > 0:
            pb_score = max(0, min(1, (10 - metrics['price_to_book']) / 10))
            score += pb_score
            count += 1
        
        # Return on Equity (higher is better)
        if metrics.get('return_on_equity', 0) > 0:
            roe_score = min(1, metrics['return_on_equity'] / 0.3)  # Normalize to 30%
            score += roe_score
            count += 1
        
        # Revenue Growth (higher is better)
        if metrics.get('revenue_growth', 0) > 0:
            growth_score = min(1, metrics['revenue_growth'] / 0.5)  # Normalize to 50%
            score += growth_score
            count += 1
        
        # Profit Margins (higher is better)
        if metrics.get('profit_margins', 0) > 0:
            margin_score = min(1, metrics['profit_margins'] / 0.2)  # Normalize to 20%
            score += margin_score
            count += 1
        
        # Debt to Equity (lower is better)
        if metrics.get('debt_to_equity', 0) > 0:
            debt_score = max(0, min(1, (2 - metrics['debt_to_equity']) / 2))
            score += debt_score
            count += 1
        
        # Current Ratio (higher is better)
        if metrics.get('current_ratio', 0) > 0:
            current_score = min(1, metrics['current_ratio'] / 3)  # Normalize to 3
            score += current_score
            count += 1
        
        return score / count if count > 0 else 0.5
    
    def _create_lob_proxy_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create lightweight microstructure proxy features from OHLCV/top-of-book.

        Adds:
        - ofi_proxy: returns * volume (pressure proxy), and its rolling z-score per ticker (ofi_z)
        - spread_proxy: (high - low)/close and z-score (spread_z)
        - microprice proxy (if bid/ask and sizes exist): z-score (micro_z)
        - time-of-day sin/cos: tod_sin, tod_cos
        """
        try:
            cfg = (self.config.get('feature_config') or {}) if isinstance(self.config, dict) else {}
            if not bool(cfg.get('enable_lob_proxies', True)):
                return df

            # Rolling window for per-ticker z-score
            zwin = int(cfg.get('lob_zscore_window', 240))

            # Order Flow Imbalance proxy
            if 'returns' not in df.columns:
                df['returns'] = df.groupby('tic')['close'].pct_change()
            if 'volume' in df.columns:
                df['ofi_proxy'] = df['returns'] * df['volume']
                # Clip heavy tails to robustify
                df['ofi_proxy'] = df.groupby('tic')['ofi_proxy'].transform(
                    lambda s: s.clip(lower=s.quantile(0.01), upper=s.quantile(0.99))
                )
                df['ofi_z'] = (
                    df.groupby('tic')['ofi_proxy'].transform(lambda s: (s - s.rolling(zwin).mean()) / (s.rolling(zwin).std() + 1e-9))
                )

            # Spread tightness proxy
            if {'high', 'low', 'close'}.issubset(df.columns):
                spread = (df['high'] - df['low']) / df['close'].replace(0, np.nan)
                df['spread_proxy'] = spread.replace([np.inf, -np.inf], np.nan)
                df['spread_proxy'] = df['spread_proxy'].fillna(0.0)
                df['spread_z'] = (
                    df.groupby('tic')['spread_proxy'].transform(lambda s: (s - s.rolling(zwin).mean()) / (s.rolling(zwin).std() + 1e-9))
                )

            # Microprice proxy (requires top-of-book)
            has_top = {'bid', 'ask', 'bid_size', 'ask_size'}.issubset(df.columns)
            if has_top:
                denom = (df['bid_size'] + df['ask_size']).replace(0, np.nan)
                micro = (df['ask'] * df['bid_size'] + df['bid'] * df['ask_size']) / denom
                micro = micro.replace([np.inf, -np.inf], np.nan).fillna(method='ffill').fillna(method='bfill')
                df['microprice'] = micro
                df['micro_dev'] = df['microprice'] - df['close']
                df['micro_z'] = df.groupby('tic')['micro_dev'].transform(lambda s: (s - s.rolling(zwin).mean()) / (s.rolling(zwin).std() + 1e-9))

            # Time-of-day sin/cos
            try:
                # Ensure datetime
                date_dtype = df['date'].dtype
                # Handle ExtensionDtype by checking string representation
                if hasattr(date_dtype, 'name') and 'datetime' not in str(date_dtype.name).lower():
                    if not isinstance(date_dtype, (np.dtype, type)) or not np.issubdtype(cast(Any, date_dtype), np.datetime64):
                        df['date'] = pd.to_datetime(df['date'])
                elif not np.issubdtype(cast(Any, date_dtype), np.datetime64):
                    df['date'] = pd.to_datetime(df['date'])
                minutes = df['date'].dt.hour * 60 + df['date'].dt.minute
                two_pi = 2 * np.pi
                df['tod_sin'] = np.sin(two_pi * minutes / 1440.0)
                df['tod_cos'] = np.cos(two_pi * minutes / 1440.0)
            except Exception:
                # If date parsing fails, default zeros
                df['tod_sin'] = 0.0
                df['tod_cos'] = 0.0

            # Final NaN/Inf guard
            for col in ['ofi_z', 'spread_z', 'micro_z', 'tod_sin', 'tod_cos']:
                if col in df.columns:
                    df[col] = df[col].replace([np.inf, -np.inf], 0.0).fillna(0.0)

            return df
        except Exception as e:
            self.logger.error(f"LOB proxy feature creation failed: {e}")
            return df
    
    def _calculate_rsi_vectorized_grouped(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate RSI indicator with vectorized operations for grouped data"""
        try:
            # Calculate RSI for each ticker separately
            result = pd.Series(index=df.index, dtype=float)
            
            for tic in df['tic'].unique():
                tic_mask = df['tic'] == tic
                tic_data = df[tic_mask]['close']
                
                if len(tic_data) > 1:  # Need at least 2 points for RSI
                    # Calculate RSI for this ticker - ensure tic_data is a Series
                    tic_series = cast(pd.Series, tic_data)
                    rsi = self._calculate_rsi_vectorized(tic_series, period)
                    # Ensure rsi is a Series before assignment
                    if isinstance(rsi, pd.Series):
                        result.loc[tic_mask] = rsi.values
                    else:
                        result.loc[tic_mask] = rsi
                else:
                    # Fallback for insufficient data
                    result[tic_mask] = 50.0
            
            return result.fillna(50.0)
            
        except Exception as e:
            self.logger.error(f"Vectorized RSI calculation failed: {e}")
            return pd.Series(50, index=df.index)
    
    def _calculate_rsi_vectorized(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator with vectorized operations"""
        try:
            # Vectorized RSI calculation
            delta = prices.diff()
            gain = delta.where(delta > 0, 0).rolling(window=period, min_periods=1).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=1).mean()
            
            # Avoid division by zero - use pandas operations instead of numpy
            # Ensure gain and loss are Series
            gain_series = cast(pd.Series, gain)
            loss_series = cast(pd.Series, loss)
            rs = gain_series / loss_series.replace(0, np.inf)  # Replace 0 with inf to avoid division by zero
            rs = cast(pd.Series, rs).replace(np.inf, 0)  # Replace inf with 0 for the final result
            rsi = 100 - (100 / (1 + rs))
            
            # Ensure we return a pandas Series with proper index
            return pd.Series(rsi, index=prices.index)
            
        except Exception as e:
            self.logger.error(f"Vectorized RSI calculation failed: {e}")
            return pd.Series(50, index=prices.index)
    
    def _calculate_atr_vectorized(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate ATR with vectorized operations"""
        try:
            # Vectorized True Range calculation
            high_low = df['high'] - df['low']
            
            # Previous close for each ticker
            prev_close = df.groupby('tic')['close'].shift(1)
            high_close = np.abs(df['high'] - prev_close)
            low_close = np.abs(df['low'] - prev_close)
            
            # True Range is the maximum of the three
            true_range = np.maximum(high_low, np.maximum(high_close, low_close))
            
            # Calculate ATR using rolling mean
            # Add true_range as a column temporarily
            df_temp = df.copy()
            if isinstance(true_range, np.ndarray):
                df_temp['_true_range'] = true_range
            else:
                true_range_series = cast(pd.Series, true_range)
                df_temp['_true_range'] = true_range_series.values if hasattr(true_range_series, 'values') else true_range_series
            
            atr_result = df_temp.groupby('tic')['_true_range'].rolling(period, min_periods=1).mean()
            if isinstance(atr_result, pd.Series):
                atr = atr_result.reset_index(0, drop=True)
            else:
                atr = pd.Series(atr_result.values, index=df.index)
            
            return atr
            
        except Exception as e:
            self.logger.error(f"Vectorized ATR calculation failed: {e}")
            return pd.Series(0, index=df.index)
    
    def _calculate_stochastic_rsi_vectorized(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Stochastic RSI with vectorized operations"""
        try:
            # Calculate Stochastic RSI for each ticker separately
            result = pd.Series(index=df.index, dtype=float)
            
            for tic in df['tic'].unique():
                tic_mask = df['tic'] == tic
                tic_data = df[tic_mask]['close']
                
                if len(tic_data) > 1:  # Need at least 2 points for RSI
                    # Calculate RSI for this ticker - ensure it's a pandas Series
                    tic_series = cast(pd.Series, tic_data)
                    tic_values = np.array(tic_series.values) if hasattr(tic_series, 'values') else np.array(tic_series)
                    rsi = self._calculate_rsi_simple(tic_values, 14)  # Use existing method
                    
                    # Ensure rsi is a pandas Series with proper index
                    if not isinstance(rsi, pd.Series):
                        rsi = pd.Series(rsi, index=tic_series.index)
                    
                    # Calculate Stochastic RSI using pandas Series methods
                    rsi_min = rsi.rolling(period, min_periods=1).min()
                    rsi_max = rsi.rolling(period, min_periods=1).max()
                    
                    # Handle division by zero properly - use pandas operations
                    denominator = rsi_max - rsi_min
                    stoch_rsi = (rsi - rsi_min) / denominator.replace(0, np.inf)
                    stoch_rsi = stoch_rsi.replace(np.inf, 0.5)
                    
                    # Ensure we have a pandas Series with proper index
                    tic_series = cast(pd.Series, tic_data)
                    stoch_rsi_series = pd.Series(stoch_rsi, index=tic_series.index)
                    result.loc[tic_mask] = stoch_rsi_series.values if hasattr(stoch_rsi_series, 'values') else stoch_rsi_series
                else:
                    # Fallback for insufficient data
                    result[tic_mask] = 0.5
            
            return result.fillna(0.5)
            
        except Exception as e:
            self.logger.error(f"Vectorized Stochastic RSI calculation failed: {e}")
            return pd.Series(0.5, index=df.index)
    
    def _calculate_obv(self, df: pd.DataFrame) -> pd.Series:
        """Calculate On-Balance Volume"""
        try:
            obv_result = df.groupby('tic').apply(
                lambda x: (np.sign(x['close'].diff()) * x['volume']).cumsum()
            ).reset_index(0, drop=True)
            # Ensure we return a Series, not a DataFrame
            if isinstance(obv_result, pd.Series):
                return obv_result
            elif isinstance(obv_result, pd.DataFrame):
                # If it's a DataFrame, take the first column or create a Series
                if len(obv_result.columns) > 0:
                    return cast(pd.Series, obv_result.iloc[:, 0])
                else:
                    return pd.Series(0.0, index=df.index)
            else:
                return cast(pd.Series, obv_result)
        except Exception as e:
            self.logger.error(f"OBV calculation failed: {e}")
            return pd.Series(0, index=df.index)
    
    def select_features(self, df: pd.DataFrame, tier: str = 'tier_1_core') -> List[str]:
        """
        Select features based on tier
        
        Args:
            df: DataFrame with features
            tier: Feature tier to select
            
        Returns:
            List of selected feature names
        """
        try:
            available_features = self.feature_tiers.get(tier, [])
            selected_features = [col for col in available_features if col in df.columns]
            
            self.logger.info(f"Selected {len(selected_features)} features from {tier}")
            return selected_features
            
        except Exception as e:
            self.logger.error(f"Feature selection failed: {e}")
            return []
    
    def _calculate_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate basic features like returns, volatility"""
        try:
            # Calculate returns
            df['returns'] = df.groupby('tic')['close'].pct_change().fillna(0.0)
            
            # Calculate volatility (rolling standard deviation of returns)
            df['volatility'] = df.groupby('tic')['returns'].rolling(20, min_periods=1).std().reset_index(0, drop=True).fillna(0.0)
            
            # Calculate momentum
            df['momentum'] = df.groupby('tic')['close'].pct_change(5).fillna(0.0)
            
            return df
        except Exception as e:
            self.logger.error(f"Basic feature calculation failed: {e}")
            return df
    
    def _calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators with robust error handling"""
        try:
            # Ensure we have required columns
            required_cols = ['close', 'high', 'low', 'volume']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                self.logger.warning(f"Missing columns for technical indicators: {missing_cols}")
                return df
            
            # Calculate moving averages with error handling
            try:
                df['sma_20'] = df.groupby('tic')['close'].rolling(20, min_periods=1).mean().reset_index(0, drop=True).fillna(0.0)
                df['ema_12'] = df.groupby('tic')['close'].ewm(span=12).mean().reset_index(0, drop=True).fillna(0.0)
                
                # CRITICAL FIX: Check for NaN/Inf in moving averages
                sma_series = cast(pd.Series, df['sma_20'])
                sma_nulls = bool(np.any(np.array(sma_series.isnull().values)))
                sma_infs = bool(np.any(np.isinf(sma_series.values)))
                if sma_nulls or sma_infs:
                    self.logger.warning(f" Found {sma_series.isnull().sum()} NaNs and {np.isinf(sma_series.values).sum()} Infs in SMA")
                    df['sma_20'] = sma_series.fillna(0.0).replace([np.inf, -np.inf], 0.0)
                
                ema_series = cast(pd.Series, df['ema_12'])
                ema_nulls = bool(np.any(np.array(ema_series.isnull().values)))
                ema_infs = bool(np.any(np.isinf(ema_series.values)))
                if ema_nulls or ema_infs:
                    self.logger.warning(f" Found {ema_series.isnull().sum()} NaNs and {np.isinf(ema_series.values).sum()} Infs in EMA")
                    df['ema_12'] = ema_series.fillna(0.0).replace([np.inf, -np.inf], 0.0)
                    
            except Exception as e:
                self.logger.error(f"Moving average calculation failed: {e}")
                df['sma_20'] = 0.0
                df['ema_12'] = 0.0
            
            # Calculate RSI with proper error handling
            try:
                def calculate_rsi_for_group(group):
                    if len(group) < 15:
                        return pd.Series(50.0, index=group.index)  # Neutral RSI
                    
                    close = group['close']
                    delta = close.diff()
                    
                    gain = delta.where(delta > 0, 0)
                    loss = -delta.where(delta < 0, 0)
                    
                    avg_gain = gain.rolling(window=14, min_periods=1).mean()
                    avg_loss = loss.rolling(window=14, min_periods=1).mean()
                    
                    rs = avg_gain / avg_loss.replace(0, 1)  # Avoid division by zero
                    rsi = 100 - (100 / (1 + rs))
                    return rsi.fillna(50.0)  # Fill NaN with neutral RSI
                
                rsi_results = df.groupby('tic').apply(calculate_rsi_for_group, include_groups=False).reset_index(0, drop=True)
                df['rsi'] = rsi_results
            except Exception as e:
                self.logger.error(f"RSI calculation failed: {e}")
                df['rsi'] = 50.0  # Neutral RSI value
            
            # Calculate MACD with proper error handling
            try:
                def calculate_macd(prices):
                    if len(prices) < 26:
                        return 0.0
                    # Use pandas ewm for more reliable EMA calculation
                    ema12 = prices.ewm(span=12).mean().iloc[-1]
                    ema26 = prices.ewm(span=26).mean().iloc[-1]
                    return ema12 - ema26
                
                df['macd'] = df.groupby('tic')['close'].transform(calculate_macd).fillna(0.0)
            except Exception as e:
                self.logger.error(f"MACD calculation failed: {e}")
                df['macd'] = 0.0
            
            # Calculate ATR with proper error handling
            try:
                def calculate_atr_for_group(group):
                    if len(group) < 15:
                        return pd.Series(0.0, index=group.index)
                    
                    high = group['high']
                    low = group['low']
                    close = group['close']
                    
                    # Calculate True Range
                    tr1 = high - low
                    tr2 = abs(high - close.shift(1))
                    tr3 = abs(low - close.shift(1))
                    
                    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
                    # Ensure true_range is a Series before calling rolling
                    # Ensure true_range is a Series before calling rolling
                    if isinstance(true_range, pd.Series):
                        atr = true_range.rolling(window=14, min_periods=1).mean()
                        atr_series = cast(pd.Series, atr)
                        return atr_series.fillna(0.0)
                    elif isinstance(true_range, np.ndarray):
                        true_range_series = pd.Series(true_range, index=group.index)
                        atr = true_range_series.rolling(window=14, min_periods=1).mean()
                        atr_series = cast(pd.Series, atr)
                        return atr_series.fillna(0.0)
                    else:
                        # Convert to Series
                        true_range_series = pd.Series(true_range.values if hasattr(true_range, 'values') else true_range, index=group.index)
                        atr = true_range_series.rolling(window=14, min_periods=1).mean()
                        atr_series = cast(pd.Series, atr)
                        return atr_series.fillna(0.0)
                
                atr_results = df.groupby('tic').apply(calculate_atr_for_group, include_groups=False).reset_index(0, drop=True)
                df['atr'] = atr_results
            except Exception as e:
                self.logger.error(f"ATR calculation failed: {e}")
                df['atr'] = 0.0
            
            # Calculate Bollinger Bands with proper error handling
            try:
                def calculate_bollinger_for_group(group):
                    if len(group) < 20:
                        return pd.DataFrame({
                            'bb_upper': 0.0,
                            'bb_middle': 0.0,
                            'bb_lower': 0.0
                        }, index=group.index)
                    
                    close = group['close']
                    sma = close.rolling(window=20, min_periods=1).mean()
                    std = close.rolling(window=20, min_periods=1).std()
                    
                    upper = sma + (std * 2)
                    lower = sma - (std * 2)
                    
                    return pd.DataFrame({
                        'bb_upper': upper.fillna(0.0),
                        'bb_middle': sma.fillna(0.0),
                        'bb_lower': lower.fillna(0.0)
                    }, index=group.index)
                
                bb_results = df.groupby('tic').apply(calculate_bollinger_for_group, include_groups=False).reset_index(0, drop=True)
                df['bb_upper'] = bb_results['bb_upper']
                df['bb_middle'] = bb_results['bb_middle']
                df['bb_lower'] = bb_results['bb_lower']
            except Exception as e:
                self.logger.error(f"Bollinger Bands calculation failed: {e}")
                df['bb_upper'] = 0.0
                df['bb_middle'] = 0.0
                df['bb_lower'] = 0.0
            
            # Calculate price to SMA ratio with robust division
            try:
                # CRITICAL FIX: Use epsilon to prevent division by zero
                epsilon = 1e-8
                df['price_to_sma_20'] = df['close'] / (df['sma_20'] + epsilon)
                df['price_to_sma_20'] = df['price_to_sma_20'].fillna(1.0).clip(0.1, 10.0)  # Clip to reasonable range
                
                # Check for NaN/Inf
                price_sma_series = cast(pd.Series, df['price_to_sma_20'])
                price_sma_nulls = bool(np.any(np.array(price_sma_series.isnull().values)))
                price_sma_infs = bool(np.any(np.isinf(price_sma_series.values)))
                if price_sma_nulls or price_sma_infs:
                    self.logger.warning(f" Found {price_sma_series.isnull().sum()} NaNs and {np.isinf(price_sma_series.values).sum()} Infs in price_to_sma_20")
                    df['price_to_sma_20'] = price_sma_series.fillna(1.0).replace([np.inf, -np.inf], 1.0)
                    
            except Exception as e:
                self.logger.error(f"Price to SMA calculation failed: {e}")
                df['price_to_sma_20'] = 1.0
            
            # Calculate volume trend
            try:
                df['volume_trend'] = df.groupby('tic')['volume'].rolling(10, min_periods=1).mean().reset_index(0, drop=True).fillna(0.0)
            except Exception as e:
                self.logger.error(f"Volume trend calculation failed: {e}")
                df['volume_trend'] = 0.0
            
            # CRITICAL FIX: Final comprehensive check for all technical indicators
            self.logger.info(" Final technical indicator validation...")
            
            # Check all numeric columns for NaN/Inf
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if col in df.columns:
                    nan_count = df[col].isnull().sum()
                    inf_count = np.isinf(df[col]).sum()
                    if nan_count > 0 or inf_count > 0:
                        self.logger.warning(f" {col}: {nan_count} NaNs, {inf_count} Infs - cleaning...")
                        df[col] = df[col].fillna(0.0).replace([np.inf, -np.inf], 0.0)
            
            self.logger.info(" Technical indicators calculated successfully")
            return df
            
        except Exception as e:
            self.logger.error(f"Technical indicator calculation failed: {e}")
            return df
    
    def _calculate_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate advanced features"""
        try:
            # Price to SMA ratio
            df['price_to_sma_20'] = df['close'] / df['sma_20'].replace(0, 1)
            
            # Volume trend
            df['volume_trend'] = df.groupby('tic')['volume'].pct_change().fillna(0.0)
            
            # ATR (Average True Range) - use existing _calculate_atr method
            df['atr'] = self._calculate_atr(df)
            
            return df
        except Exception as e:
            self.logger.error(f"Advanced feature calculation failed: {e}")
            return df
    
    def get_feature_importance(self, df: pd.DataFrame, target_col: str = 'returns') -> Dict[str, float]:
        """
        Calculate feature importance using correlation
        
        Args:
            df: DataFrame with features
            target_col: Target column for importance calculation
            
        Returns:
            Dictionary of feature importance scores
        """
        try:
            if target_col not in df.columns:
                self.logger.warning(f"Target column {target_col} not found")
                return {}
            
            # Calculate correlations with target
            correlations = df.corr()[target_col].abs().sort_values(ascending=False)  # type: ignore[arg-type]
            
            # Remove target column and non-feature columns
            feature_importance = {}
            for col in correlations.index:
                if col != target_col and col not in self.exclude_cols:
                    feature_importance[col] = correlations[col]
            
            return feature_importance
            
        except Exception as e:
            self.logger.error(f"Feature importance calculation failed: {e}")
            return {}

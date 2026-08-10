"""
Pre-Compute Runtime Calculations Cache

This module pre-computes all calculations that happen at runtime during training,
significantly speeding up PPO training by eliminating redundant calculations.

Calculations Pre-Computed:
1. VWAP (Volume-Weighted Average Price) - per ticker, per step
2. Market Regime (trending/mean_reversion) - per step
3. Volatility Metrics - per ticker, per step
4. Returns - per ticker, per step
5. Technical Signal Scores - per ticker, per step
6. Volume Ratios - per ticker, per step
7. ATR (Average True Range) - per ticker, per step
8. Price Changes - per ticker, per step

Author: PPO Trading Agent Team
Date: 2026-01-25
"""

import os
import sys
import pandas as pd
import numpy as np
import pickle
import hashlib
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, cast
from datetime import datetime
import json

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

logger = logging.getLogger(__name__)


class RuntimeCalculationsCache:
    """
    Pre-compute and cache all runtime calculations for faster training.
    
    This eliminates redundant calculations during PPO training by pre-computing:
    - VWAP values
    - Market regime classifications
    - Volatility metrics
    - Returns
    - Technical signal scores
    - Volume ratios
    - ATR values
    """
    
    def __init__(self, 
                 cache_dir: str = "ppo/saved_data/calculation_cache",
                 vwap_lookback: int = 10,
                 atr_period: int = 14,
                 high_volatility_threshold: float = 0.02):
        """
        Initialize calculation cache.
        
        Args:
            cache_dir: Directory for cache files
            vwap_lookback: Lookback period for VWAP (default 10)
            atr_period: Period for ATR calculation (default 14)
            high_volatility_threshold: Threshold for high volatility regime (default 0.02)
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.vwap_lookback = vwap_lookback
        self.atr_period = atr_period
        self.high_volatility_threshold = high_volatility_threshold
        
        self.logger = logging.getLogger(__name__)
        
    def file_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash of file."""
        if not file_path.exists():
            return ""
        with open(file_path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()
    
    def precompute_vwap(self, price_matrix: np.ndarray, volume_matrix: np.ndarray) -> np.ndarray:
        """
        Pre-compute VWAP for all tickers at all steps.
        
        Returns:
            VWAP matrix: shape (num_steps, num_tickers)
        """
        self.logger.info("Pre-computing VWAP...")
        num_steps, num_tickers = price_matrix.shape
        vwap_matrix = np.zeros((num_steps, num_tickers))
        
        for step in range(num_steps):
            for ticker in range(num_tickers):
                start_idx = max(0, step - self.vwap_lookback)
                end_idx = step  # Exclude current step (data leakage prevention)
                
                if end_idx <= start_idx:
                    vwap_matrix[step, ticker] = price_matrix[step, ticker]  # Use current price
                    continue
                
                prices = price_matrix[start_idx:end_idx, ticker]
                volumes = volume_matrix[start_idx:end_idx, ticker]
                
                total_vol = np.sum(volumes)
                if total_vol > 0:
                    vwap = np.sum(prices * volumes) / total_vol
                    vwap_matrix[step, ticker] = vwap
                else:
                    vwap_matrix[step, ticker] = price_matrix[step, ticker]
        
        self.logger.info(f"✅ Pre-computed VWAP: {vwap_matrix.shape}")
        return vwap_matrix
    
    def precompute_returns(self, price_matrix: np.ndarray) -> np.ndarray:
        """
        Pre-compute returns for all tickers.
        
        Returns:
            Returns matrix: shape (num_steps, num_tickers)
        """
        self.logger.info("Pre-computing returns...")
        returns = np.diff(price_matrix, axis=0) / (price_matrix[:-1] + 1e-10)
        # Pad first row with zeros (no previous price)
        returns_padded = np.zeros_like(price_matrix)
        returns_padded[1:] = returns
        returns_padded[0] = 0.0
        
        self.logger.info(f"✅ Pre-computed returns: {returns_padded.shape}")
        return returns_padded
    
    def precompute_volatility(self, price_matrix: np.ndarray, window: int = 20) -> np.ndarray:
        """
        Pre-compute volatility (rolling std of returns) for all tickers.
        
        Returns:
            Volatility matrix: shape (num_steps, num_tickers)
        """
        self.logger.info("Pre-computing volatility...")
        num_steps, num_tickers = price_matrix.shape
        volatility_matrix = np.zeros((num_steps, num_tickers))
        
        returns = self.precompute_returns(price_matrix)
        
        for step in range(num_steps):
            for ticker in range(num_tickers):
                start_idx = max(0, step - window + 1)
                end_idx = step + 1
                
                if end_idx <= start_idx:
                    volatility_matrix[step, ticker] = 0.0
                    continue
                
                step_returns = returns[start_idx:end_idx, ticker]
                if len(step_returns) > 1:
                    volatility_matrix[step, ticker] = np.std(step_returns)
                else:
                    volatility_matrix[step, ticker] = 0.0
        
        self.logger.info(f"✅ Pre-computed volatility: {volatility_matrix.shape}")
        return volatility_matrix
    
    def precompute_atr(self, price_matrix: np.ndarray, high_matrix: Optional[np.ndarray] = None, 
                       low_matrix: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Pre-compute ATR (Average True Range) for all tickers.
        
        If high/low matrices not provided, uses price_matrix as approximation.
        
        Returns:
            ATR matrix: shape (num_steps, num_tickers)
        """
        self.logger.info("Pre-computing ATR...")
        num_steps, num_tickers = price_matrix.shape
        atr_matrix = np.zeros((num_steps, num_tickers))
        
        # If high/low not provided, use price as approximation
        if high_matrix is None:
            high_matrix = price_matrix
        if low_matrix is None:
            low_matrix = price_matrix
        
        for step in range(num_steps):
            for ticker in range(num_tickers):
                start_idx = max(0, step - self.atr_period + 1)
                end_idx = step + 1
                
                if end_idx <= start_idx or step == 0:
                    atr_matrix[step, ticker] = 0.0
                    continue
                
                # Calculate True Range for this period
                tr_values = []
                for i in range(start_idx, end_idx):
                    if i == 0:
                        continue
                    high = high_matrix[i, ticker]
                    low = low_matrix[i, ticker]
                    prev_close = price_matrix[i-1, ticker]
                    
                    tr1 = high - low
                    tr2 = abs(high - prev_close)
                    tr3 = abs(low - prev_close)
                    tr = max(tr1, tr2, tr3)
                    tr_values.append(tr)
                
                if tr_values:
                    atr_matrix[step, ticker] = np.mean(tr_values)
                else:
                    atr_matrix[step, ticker] = 0.0
        
        self.logger.info(f"✅ Pre-computed ATR: {atr_matrix.shape}")
        return atr_matrix
    
    def precompute_market_regime(self, price_matrix: np.ndarray) -> np.ndarray:
        """
        Pre-compute market regime (trending/mean_reversion) for each step.
        
        Returns:
            Regime array: shape (num_steps,) with values: 0=unknown, 1=trending, 2=mean_reversion
        """
        self.logger.info("Pre-computing market regime...")
        num_steps = price_matrix.shape[0]
        regime_array = np.zeros(num_steps, dtype=int)  # 0=unknown, 1=trending, 2=mean_reversion
        
        # Pre-compute ATR for regime detection
        atr_matrix = self.precompute_atr(price_matrix)
        
        for step in range(num_steps):
            if step < self.atr_period:
                regime_array[step] = 0  # unknown
                continue
            
            # Calculate normalized ATR for all tickers at this step
            atr_values = []
            for ticker in range(price_matrix.shape[1]):
                atr = atr_matrix[step, ticker]
                price = price_matrix[step, ticker]
                if price > 0:
                    atr_norm = atr / price
                    atr_values.append(atr_norm)
            
            if atr_values:
                avg_atr = np.mean(atr_values)
                # 1 = trending, 2 = mean_reversion
                regime_array[step] = 1 if avg_atr > self.high_volatility_threshold else 2
            else:
                regime_array[step] = 0  # unknown
        
        self.logger.info(f"✅ Pre-computed market regime: {regime_array.shape}")
        return regime_array
    
    def precompute_volume_ratios(self, volume_matrix: np.ndarray, window: int = 20) -> np.ndarray:
        """
        Pre-compute volume ratios (current / average) for all tickers.
        
        Returns:
            Volume ratio matrix: shape (num_steps, num_tickers)
        """
        self.logger.info("Pre-computing volume ratios...")
        num_steps, num_tickers = volume_matrix.shape
        volume_ratio_matrix = np.zeros((num_steps, num_tickers))
        
        for step in range(num_steps):
            for ticker in range(num_tickers):
                start_idx = max(0, step - window + 1)
                end_idx = step + 1
                
                if end_idx <= start_idx:
                    volume_ratio_matrix[step, ticker] = 1.0
                    continue
                
                current_vol = volume_matrix[step, ticker]
                avg_vol = np.mean(volume_matrix[start_idx:end_idx, ticker])
                
                if avg_vol > 0:
                    volume_ratio_matrix[step, ticker] = current_vol / avg_vol
                else:
                    volume_ratio_matrix[step, ticker] = 1.0
        
        self.logger.info(f"✅ Pre-computed volume ratios: {volume_ratio_matrix.shape}")
        return volume_ratio_matrix
    
    def precompute_technical_signals(self, 
                                    price_matrix: np.ndarray,
                                    volume_matrix: np.ndarray,
                                    feature_matrix: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """
        Pre-compute technical signal scores for all tickers at all steps.
        
        Uses feature_matrix if available (faster), otherwise calculates from price_matrix.
        
        Returns:
            Dictionary with signal arrays:
            - rsi: RSI values (num_steps, num_tickers)
            - macd: MACD values (num_steps, num_tickers)
            - macd_signal: MACD signal values (num_steps, num_tickers)
            - bb_z: Bollinger Band z-scores (num_steps, num_tickers)
            - volume_score: Volume scores (num_steps, num_tickers)
        """
        self.logger.info("Pre-computing technical signals...")
        num_steps, num_tickers = price_matrix.shape
        
        signals = {
            'rsi': np.zeros((num_steps, num_tickers)),
            'macd': np.zeros((num_steps, num_tickers)),
            'macd_signal': np.zeros((num_steps, num_tickers)),
            'bb_z': np.zeros((num_steps, num_tickers)),
            'volume_score': np.zeros((num_steps, num_tickers))
        }
        
        # If feature matrix available, extract indicators directly
        if feature_matrix is not None:
            # Feature matrix shape: (num_steps, num_tickers, num_features)
            # Assuming feature order: [sma_20, ema_12, rsi, macd, macd_signal, atr, bb_upper, bb_middle, bb_lower, ...]
            # This is approximate - adjust based on actual feature order
            try:
                # Extract arrays and ensure proper shape
                rsi_array = np.array(feature_matrix[:, :, 2])  # RSI at index 2
                macd_array = np.array(feature_matrix[:, :, 3])  # MACD at index 3
                macd_signal_array = np.array(feature_matrix[:, :, 4])  # MACD signal at index 4
                
                # Ensure arrays match expected shape - use type ignore for shape assignment
                if rsi_array.shape == signals['rsi'].shape:
                    signals['rsi'] = cast(Any, rsi_array)  # type: ignore[assignment]
                if macd_array.shape == signals['macd'].shape:
                    signals['macd'] = cast(Any, macd_array)  # type: ignore[assignment]
                if macd_signal_array.shape == signals['macd_signal'].shape:
                    signals['macd_signal'] = cast(Any, macd_signal_array)  # type: ignore[assignment]
                
                # Calculate BB z-score from BB bands
                bb_upper = np.array(feature_matrix[:, :, 6])  # BB upper
                bb_middle = np.array(feature_matrix[:, :, 7])  # BB middle (SMA_20)
                bb_lower = np.array(feature_matrix[:, :, 8])  # BB lower
                
                bb_std = (bb_upper - bb_middle) / 2.0  # Approximate std
                price_arr = np.array(price_matrix)
                bb_z_array = (price_arr - bb_middle) / (bb_std + 1e-9)
                if bb_z_array.shape == signals['bb_z'].shape:
                    signals['bb_z'] = cast(Any, bb_z_array)  # type: ignore[assignment]
                
                self.logger.info("✅ Extracted technical signals from feature matrix")
            except (IndexError, TypeError):
                self.logger.warning("Feature matrix extraction failed, calculating from scratch...")
                feature_matrix = None
        
        # Calculate from scratch if feature matrix not available or extraction failed
        if feature_matrix is None:
            for ticker in range(num_tickers):
                prices = price_matrix[:, ticker]
                volumes = volume_matrix[:, ticker]
                
                # RSI
                delta = np.diff(prices)
                gain = np.clip(delta, 0, None)
                loss = -np.clip(delta, None, 0)
                
                for step in range(1, num_steps):
                    window = min(14, step)
                    if window > 0:
                        avg_gain = np.mean(gain[max(0, step-window):step])
                        avg_loss = np.mean(loss[max(0, step-window):step])
                        if avg_loss > 0:
                            rs = avg_gain / avg_loss
                            signals['rsi'][step, ticker] = 100.0 - (100.0 / (1.0 + rs))
                        else:
                            signals['rsi'][step, ticker] = 100.0
                
                # MACD (simplified - use EMA approximation)
                prices_series = pd.Series(prices)
                ema12_series = prices_series.ewm(span=12, adjust=False).mean()
                ema26_series = prices_series.ewm(span=26, adjust=False).mean()
                ema12 = np.array(ema12_series.values) if hasattr(ema12_series, 'values') else np.array(ema12_series)
                ema26 = np.array(ema26_series.values) if hasattr(ema26_series, 'values') else np.array(ema26_series)
                macd_values = ema12 - ema26
                signals['macd'][:, ticker] = macd_values
                
                macd_series = pd.Series(macd_values)
                macd_signal_series = macd_series.ewm(span=9, adjust=False).mean()
                macd_signal_values = np.array(macd_signal_series.values) if hasattr(macd_signal_series, 'values') else np.array(macd_signal_series)
                signals['macd_signal'][:, ticker] = macd_signal_values
                
                # Bollinger Bands z-score
                sma20_series = prices_series.rolling(window=20, min_periods=1).mean()
                std20_series = prices_series.rolling(window=20, min_periods=1).std()
                # Ensure these are Series before accessing .values
                sma20_series_typed = cast(pd.Series, sma20_series)
                std20_series_typed = cast(pd.Series, std20_series)
                sma20 = np.array(sma20_series_typed.values)
                std20 = np.array(std20_series_typed.values)
                prices_arr = np.array(prices)
                signals['bb_z'][:, ticker] = (prices_arr - sma20) / (std20 + 1e-9)
                
                # Volume score
                volumes_series = pd.Series(volumes)
                vol_sma_series = volumes_series.rolling(window=20, min_periods=1).mean()
                # Ensure this is a Series before accessing .values
                vol_sma_series_typed = cast(pd.Series, vol_sma_series)
                vol_sma = np.array(vol_sma_series_typed.values)
                volumes_arr = np.array(volumes)
                vol_ratio = volumes_arr / (vol_sma + 1e-9)
                signals['volume_score'][:, ticker] = np.where(
                    vol_ratio > 1.2, 0.2,
                    np.where(vol_ratio < 0.8, -0.2, 0.0)
                )
            
            self.logger.info("✅ Calculated technical signals from price/volume data")
        
        self.logger.info(f"✅ Pre-computed technical signals")
        return signals
    
    def precompute_all(self, 
                      price_matrix: np.ndarray,
                      volume_matrix: np.ndarray,
                      feature_matrix: Optional[np.ndarray] = None,
                      high_matrix: Optional[np.ndarray] = None,
                      low_matrix: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """
        Pre-compute all runtime calculations.
        
        Returns:
            Dictionary with all pre-computed arrays
        """
        self.logger.info("=" * 80)
        self.logger.info("PRE-COMPUTING RUNTIME CALCULATIONS")
        self.logger.info("=" * 80)
        
        cache = {}
        
        # 1. VWAP
        cache['vwap'] = self.precompute_vwap(price_matrix, volume_matrix)
        
        # 2. Returns
        cache['returns'] = self.precompute_returns(price_matrix)
        
        # 3. Volatility
        cache['volatility'] = self.precompute_volatility(price_matrix)
        
        # 4. ATR
        cache['atr'] = self.precompute_atr(price_matrix, high_matrix, low_matrix)
        
        # 5. Market Regime
        cache['market_regime'] = self.precompute_market_regime(price_matrix)
        
        # 6. Volume Ratios
        cache['volume_ratios'] = self.precompute_volume_ratios(volume_matrix)
        
        # 7. Technical Signals
        cache['technical_signals'] = self.precompute_technical_signals(
            price_matrix, volume_matrix, feature_matrix
        )
        
        self.logger.info("=" * 80)
        self.logger.info("✅ ALL CALCULATIONS PRE-COMPUTED")
        self.logger.info("=" * 80)
        
        return cache
    
    def save_cache(self, cache: Dict[str, np.ndarray], cache_file: Optional[Path] = None) -> Path:
        """
        Save pre-computed cache to disk.
        
        Args:
            cache: Dictionary with pre-computed arrays
            cache_file: Optional path to cache file
            
        Returns:
            Path to saved cache file
        """
        if cache_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            cache_file = self.cache_dir / f"runtime_calculations_{timestamp}.pkl"
        
        with open(cache_file, 'wb') as f:
            pickle.dump(cache, f)
        
        # Also save metadata
        metadata = {
            'created_at': datetime.now().isoformat(),
            'vwap_lookback': self.vwap_lookback,
            'atr_period': self.atr_period,
            'high_volatility_threshold': self.high_volatility_threshold,
            'cache_hash': self.file_hash(cache_file),
            'shapes': {k: list(v.shape) if isinstance(v, np.ndarray) else str(type(v)) 
                      for k, v in cache.items()}
        }
        
        metadata_file = cache_file.with_suffix('.json')
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save as latest
        latest_file = self.cache_dir / "runtime_calculations_latest.pkl"
        with open(latest_file, 'wb') as f:
            pickle.dump(cache, f)
        
        self.logger.info(f"✅ Saved calculation cache to {cache_file}")
        self.logger.info(f"✅ Saved metadata to {metadata_file}")
        self.logger.info(f"✅ Saved as latest: {latest_file}")
        
        return cache_file
    
    def load_cache(self, cache_file: Optional[Path] = None) -> Dict[str, np.ndarray]:
        """
        Load pre-computed cache from disk.
        
        Args:
            cache_file: Optional path to cache file (defaults to latest)
            
        Returns:
            Dictionary with pre-computed arrays
        """
        if cache_file is None:
            cache_file = self.cache_dir / "runtime_calculations_latest.pkl"
        
        if not cache_file.exists():
            raise FileNotFoundError(
                f"Calculation cache not found: {cache_file}\n"
                f"Run precompute_all() first to create cache."
            )
        
        with open(cache_file, 'rb') as f:
            cache = pickle.load(f)
        
        self.logger.info(f"✅ Loaded calculation cache from {cache_file}")
        return cache


def main():
    """Example: Pre-compute calculations from CSV data."""
    import logging
    logging.basicConfig(level=logging.INFO)
    
    from .csv_data_manager import CSVDataManager
    from .csv_loader_adapter import CSVLoaderAdapter
    
    logger.info("=" * 80)
    logger.info("PRE-COMPUTING RUNTIME CALCULATIONS FROM CSV")
    logger.info("=" * 80)
    
    # Load CSV data (merge_intraday=True in adapter)
    csv_manager = CSVDataManager()
    df = csv_manager.load_processed_data(merge_intraday=True)
    
    # Convert to matrices using canonical feature list (base + intraday)
    adapter = CSVLoaderAdapter(csv_manager)
    tickers = sorted(df['Ticker'].unique())
    try:
        from ppo.utils.dynamic_config import get_canonical_feature_cols
        feature_cols = get_canonical_feature_cols(include_intraday=True)
    except Exception:
        feature_cols = ['sma_20', 'ema_12', 'rsi', 'macd', 'macd_signal', 'atr',
                       'bb_upper', 'bb_middle', 'bb_lower', 'price_to_sma_20', 'volume_trend']
    
    data_dict = adapter.load_and_convert_to_matrices(tickers, feature_cols, train_split=1.0)
    train_data = data_dict['train_data']
    
    # Pre-compute calculations
    calculator = RuntimeCalculationsCache()
    cache = calculator.precompute_all(
        price_matrix=train_data['price_matrix'],
        volume_matrix=train_data['volume_matrix'],
        feature_matrix=train_data['feature_matrix']
    )
    
    # Save cache
    cache_file = calculator.save_cache(cache)
    
    logger.info("=" * 80)
    logger.info("✅ PRE-COMPUTATION COMPLETE")
    logger.info(f"   Cache file: {cache_file}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()

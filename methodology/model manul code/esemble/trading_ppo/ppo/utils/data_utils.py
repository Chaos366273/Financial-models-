"""
Data utility functions for trading system
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from pathlib import Path
import pickle
import json

logger = logging.getLogger(__name__)

class DataUtils:
    """Utility functions for data processing and management"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def validate_data_quality(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Validate data quality and return quality metrics"""
        quality_report = {}
        
        try:
            # Basic statistics
            quality_report['shape'] = data.shape
            quality_report['memory_usage'] = data.memory_usage(deep=True).sum()
            
            # Missing values
            missing_counts = data.isnull().sum()
            quality_report['missing_values'] = missing_counts.to_dict()
            quality_report['total_missing'] = missing_counts.sum()
            
            # Duplicate rows
            quality_report['duplicate_rows'] = data.duplicated().sum()
            
            # Data types
            quality_report['dtypes'] = data.dtypes.to_dict()
            
            # Numeric columns statistics
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                quality_report['numeric_stats'] = {
                    col: {
                        'mean': data[col].mean(),
                        'std': data[col].std(),
                        'min': data[col].min(),
                        'max': data[col].max(),
                        'null_count': data[col].isnull().sum()
                    }
                    for col in numeric_cols
                }
            
            # Date range (if date column exists)
            date_cols = data.select_dtypes(include=['datetime64']).columns
            if len(date_cols) > 0:
                for col in date_cols:
                    quality_report[f'{col}_range'] = {
                        'start': data[col].min(),
                        'end': data[col].max(),
                        'unique_dates': data[col].nunique()
                    }
            
            self.logger.info(f" Data quality validation completed: {data.shape}")
            
        except Exception as e:
            self.logger.error(f" Data quality validation failed: {e}")
            quality_report['error'] = str(e)
        
        return quality_report
    
    def save_data(self, data: pd.DataFrame, filepath: str, format: str = 'pickle') -> bool:
        """Save data to file with specified format"""
        try:
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            
            if format == 'pickle':
                data.to_pickle(filepath)
            elif format == 'csv':
                data.to_csv(filepath, index=False)
            elif format == 'parquet':
                data.to_parquet(filepath, index=False)
            elif format == 'json':
                data.to_json(filepath, orient='records')
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            self.logger.info(f" Data saved to {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f" Failed to save data: {e}")
            return False
    
    def load_data(self, filepath: str, format: str = 'auto') -> Optional[pd.DataFrame]:
        """Load data from file with automatic format detection"""
        try:
            if format == 'auto':
                if filepath.endswith('.pkl') or filepath.endswith('.pickle'):
                    format = 'pickle'
                elif filepath.endswith('.csv'):
                    format = 'csv'
                elif filepath.endswith('.parquet'):
                    format = 'parquet'
                elif filepath.endswith('.json'):
                    format = 'json'
                else:
                    format = 'pickle'  # Default
            
            if format == 'pickle':
                data = pd.read_pickle(filepath)
            elif format == 'csv':
                data = pd.read_csv(filepath)
            elif format == 'parquet':
                data = pd.read_parquet(filepath)
            elif format == 'json':
                data = pd.read_json(filepath, orient='records')
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            if isinstance(data, pd.Series):
                data = data.to_frame()
            self.logger.info(f" Data loaded from {filepath}: {data.shape}")
            return data
            
        except Exception as e:
            self.logger.error(f" Failed to load data: {e}")
            return None
    
    def resample_data(self, data: pd.DataFrame, freq: str, agg_method: str = 'mean') -> pd.DataFrame:
        """Resample time series data to different frequency"""
        try:
            # Ensure datetime index
            if not isinstance(data.index, pd.DatetimeIndex):
                date_col = data.select_dtypes(include=['datetime64']).columns[0]
                data = data.set_index(date_col)
            
            # Resample
            if agg_method == 'mean':
                resampled = data.resample(freq).mean()
            elif agg_method == 'sum':
                resampled = data.resample(freq).sum()
            elif agg_method == 'last':
                resampled = data.resample(freq).last()
            elif agg_method == 'first':
                resampled = data.resample(freq).first()
            else:
                raise ValueError(f"Unsupported aggregation method: {agg_method}")
            
            # Drop NaN rows
            resampled = resampled.dropna()
            
            self.logger.info(f" Data resampled to {freq}: {resampled.shape}")
            return resampled
            
        except Exception as e:
            self.logger.error(f" Data resampling failed: {e}")
            return data
    
    def calculate_returns(self, prices: pd.Series, method: str = 'pct_change') -> pd.Series:
        """Calculate returns from price series"""
        try:
            if method == 'pct_change':
                returns = prices.pct_change()
            elif method == 'log':
                returns = np.log(prices / prices.shift(1))
            else:
                raise ValueError(f"Unsupported return method: {method}")
            
            return returns.fillna(0)
            
        except Exception as e:
            self.logger.error(f" Return calculation failed: {e}")
            return pd.Series([0] * len(prices), index=prices.index)
    
    def calculate_technical_indicators(self, data: pd.DataFrame, 
                                    indicators: Optional[List[str]] = None) -> pd.DataFrame:
        """Calculate technical indicators for the data"""
        if indicators is None:
            indicators = ['sma_20', 'sma_50', 'rsi', 'macd']
        
        result = data.copy()
        
        try:
            close_series = result['close'] if 'close' in result.columns else None
            if close_series is None or not isinstance(close_series, pd.Series):
                raise ValueError("Missing or invalid 'close' series in data")

            for indicator in indicators:
                if indicator == 'sma_20':
                    result['sma_20'] = close_series.rolling(window=20).mean()
                elif indicator == 'sma_50':
                    result['sma_50'] = close_series.rolling(window=50).mean()
                elif indicator == 'rsi':
                    result['rsi'] = self._calculate_rsi(close_series)
                elif indicator == 'macd':
                    macd_data = self._calculate_macd(close_series)
                    result['macd'] = macd_data['macd']
                    result['macd_signal'] = macd_data['signal']
                    result['macd_histogram'] = macd_data['histogram']
            
            self.logger.info(f" Calculated {len(indicators)} technical indicators")
            
        except Exception as e:
            self.logger.error(f" Technical indicator calculation failed: {e}")
        
        return result
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return pd.Series(rsi, index=prices.index)
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict:
        """Calculate MACD indicator"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal).mean()
        histogram = macd - signal_line
        
        return {
            'macd': macd,
            'signal': signal_line,
            'histogram': histogram
        }
    
    def split_data(self, data: pd.DataFrame, train_ratio: float = 0.8, 
                   date_col: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split data into training and testing sets"""
        try:
            if date_col:
                data = data.sort_values(date_col)
                split_idx = int(len(data) * train_ratio)
                train_data = data.iloc[:split_idx]
                test_data = data.iloc[split_idx:]
            else:
                split_idx = int(len(data) * train_ratio)
                train_data = data.iloc[:split_idx]
                test_data = data.iloc[split_idx:]
            
            self.logger.info(f" Data split: train={train_data.shape}, test={test_data.shape}")
            return train_data, test_data
            
        except Exception as e:
            self.logger.error(f" Data splitting failed: {e}")
            return data, data
    
    def normalize_data(self, data: pd.DataFrame, method: str = 'minmax') -> pd.DataFrame:
        """Normalize data using specified method"""
        try:
            result = data.copy()
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            
            for col in numeric_cols:
                if method == 'minmax':
                    result[col] = (data[col] - data[col].min()) / (data[col].max() - data[col].min())
                elif method == 'zscore':
                    result[col] = (data[col] - data[col].mean()) / data[col].std()
                elif method == 'robust':
                    result[col] = (data[col] - data[col].median()) / (data[col].quantile(0.75) - data[col].quantile(0.25))
            
            self.logger.info(f" Data normalized using {method} method")
            return result
            
        except Exception as e:
            self.logger.error(f" Data normalization failed: {e}")
            return data

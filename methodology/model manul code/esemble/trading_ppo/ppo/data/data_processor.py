"""
Data processor for PPO Trading Agent
Handles data cleaning, validation, and preprocessing
"""

import pandas as pd
import numpy as np
import pickle
import os
from typing import List, Dict, Tuple, Optional
import logging
from datetime import datetime, timedelta

from ..core.exceptions import DataException, ValidationException

class DataProcessor:
    """Data processor for cleaning and preprocessing market data"""
    
    def __init__(self, config: Dict):
        """
        Initialize data processor
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Data quality settings
        self.outlier_threshold = config.get('data_quality', {}).get('outlier_threshold', 0.1)
        self.min_data_points = config.get('data_quality', {}).get('min_data_points', 1000)
        self.max_missing_days = config.get('data_quality', {}).get('max_missing_days', 30)
        self.price_range_min = config.get('data_quality', {}).get('price_range_min', 0.01)
        self.price_range_max = config.get('data_quality', {}).get('price_range_max', 10000)
        self.volume_range_min = config.get('data_quality', {}).get('volume_range_min', 1000)
        self.returns_range_max = config.get('data_quality', {}).get('returns_range_max', 0.5)
        
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and validate market data
        
        Args:
            df: Raw market data DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        try:
            self.logger.info(f"Cleaning data with {len(df)} rows")
            
            # Remove duplicates
            df = df.drop_duplicates()
            
            # Sort by date and ticker
            df = df.sort_values(['date', 'tic'])
            
            # Handle missing values
            df = self._handle_missing_values(df)
            
            # Remove outliers
            df = self._remove_outliers(df)
            
            # Validate data quality
            if not self._validate_data_quality(df):
                raise DataException("Data quality validation failed")
            
            # Fill remaining missing values
            df = df.fillna(method='ffill').fillna(method='bfill')
            
            self.logger.info(f"Data cleaning complete. Final shape: {df.shape}")
            return df
            
        except Exception as e:
            self.logger.error(f"Data cleaning failed: {e}")
            raise DataException(f"Data cleaning failed: {e}")
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the data"""
        try:
            # Check for missing values
            missing_counts = df.isnull().sum()
            if missing_counts.sum() > 0:
                self.logger.warning(f"Missing values found: {missing_counts}")
            
            # Remove rows with too many missing values
            max_missing_cols = len(df.columns) * 0.5  # Allow 50% missing columns
            df = df.dropna(thresh=max_missing_cols)
            
            # For price data, use forward fill then backward fill
            price_cols = ['open', 'high', 'low', 'close']
            for col in price_cols:
                if col in df.columns:
                    df[col] = df.groupby('tic')[col].fillna(method='ffill').fillna(method='bfill')
            
            # For volume, fill with median
            if 'volume' in df.columns:
                df['volume'] = df.groupby('tic')['volume'].fillna(df['volume'].median())
            
            return df
            
        except Exception as e:
            self.logger.error(f"Missing value handling failed: {e}")
            return df
    
    def _remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove statistical outliers from the data"""
        try:
            # Calculate returns for outlier detection
            df['returns'] = df.groupby('tic')['close'].pct_change()
            
            # Remove extreme returns
            outlier_mask = (df['returns'].abs() > self.returns_range_max)
            if outlier_mask.sum() > 0:
                self.logger.warning(f"Removing {outlier_mask.sum()} outlier returns")
                df = df[~outlier_mask]
            
            # Remove extreme prices
            for col in ['open', 'high', 'low', 'close']:
                if col in df.columns:
                    # Remove prices outside valid range
                    invalid_prices = (df[col] < self.price_range_min) | (df[col] > self.price_range_max)
                    if invalid_prices.sum() > 0:
                        self.logger.warning(f"Removing {invalid_prices.sum()} invalid {col} prices")
                        df = df[~invalid_prices]
            
            # Remove extreme volumes
            if 'volume' in df.columns:
                invalid_volumes = df['volume'] < self.volume_range_min
                if invalid_volumes.sum() > 0:
                    self.logger.warning(f"Removing {invalid_volumes.sum()} invalid volumes")
                    df = df[~invalid_volumes]
            
            return df
            
        except Exception as e:
            self.logger.error(f"Outlier removal failed: {e}")
            return df
    
    def _validate_data_quality(self, df: pd.DataFrame) -> bool:
        """Validate data quality after cleaning"""
        try:
            # Check minimum data points per ticker
            ticker_counts = df['tic'].value_counts()
            insufficient_data = ticker_counts[ticker_counts < self.min_data_points]
            if len(insufficient_data) > 0:
                self.logger.warning(f"Insufficient data for tickers: {insufficient_data.index.tolist()}")
                return False
            
            # Check for required columns
            required_cols = ['date', 'tic', 'close']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                self.logger.error(f"Missing required columns: {missing_cols}")
                return False
            
            # Check for valid prices
            if 'close' in df.columns:
                invalid_prices = (df['close'] <= 0).sum()
                if invalid_prices > 0:
                    self.logger.error(f"Found {invalid_prices} invalid close prices")
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Data quality validation failed: {e}")
            return False
    
    def split_data_by_date(
        self, 
        df: pd.DataFrame, 
        train_ratio: float = 0.7,
        val_ratio: float = 0.15
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List, List, List]:
        """
        Split data into train/validation/test sets by date
        
        Args:
            df: Full dataset
            train_ratio: Training data ratio
            val_ratio: Validation data ratio
            
        Returns:
            Tuple of (train_df, val_df, test_df, train_dates, val_dates, test_dates)
        """
        try:
            # Get unique dates
            dates = sorted(df['date'].unique())
            total_days = len(dates)
            
            # Calculate split indices
            train_end = int(total_days * train_ratio)
            val_end = int(total_days * (train_ratio + val_ratio))
            
            # Split dates
            train_dates = dates[:train_end]
            val_dates = dates[train_end:val_end]
            test_dates = dates[val_end:]
            
            # Split data
            train_df = df[df['date'].isin(train_dates)]
            val_df = df[df['date'].isin(val_dates)]
            test_df = df[df['date'].isin(test_dates)]
            
            self.logger.info(f"Data split: Train={len(train_dates)} days, Val={len(val_dates)} days, Test={len(test_dates)} days")
            
            return train_df, val_df, test_df, train_dates, val_dates, test_dates
            
        except Exception as e:
            self.logger.error(f"Data splitting failed: {e}")
            raise DataException(f"Data splitting failed: {e}")
    
    def create_price_matrix(self, df: pd.DataFrame, tickers: List[str]) -> np.ndarray:
        """
        Create price matrix for environment
        
        Args:
            df: Market data DataFrame
            tickers: List of ticker symbols
            
        Returns:
            Price matrix with shape (timesteps, num_tickers)
        """
        try:
            # Pivot data to get prices by ticker
            price_df = df.pivot(index='date', columns='tic', values='close')
            
            # Ensure all tickers are present
            for ticker in tickers:
                if ticker not in price_df.columns:
                    self.logger.warning(f"Ticker {ticker} not found in data, filling with NaN")
                    price_df[ticker] = np.nan
            
            # Select only the specified tickers
            price_df = price_df[tickers]
            
            # Fill missing values
            price_df = price_df.fillna(method='ffill').fillna(method='bfill')
            
            # Convert to numpy array
            price_matrix = price_df.values
            
            self.logger.info(f"Created price matrix with shape: {price_matrix.shape}")
            return price_matrix
            
        except Exception as e:
            self.logger.error(f"Price matrix creation failed: {e}")
            raise DataException(f"Price matrix creation failed: {e}")
    
    def save_processed_data(self, df: pd.DataFrame, filepath: str):
        """Save processed data to file"""
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            df.to_csv(filepath, index=False)
            self.logger.info(f"Saved processed data to {filepath}")
        except Exception as e:
            self.logger.error(f"Failed to save processed data: {e}")
            raise DataException(f"Failed to save processed data: {e}")
    
    def load_processed_data(self, filepath: str) -> pd.DataFrame:
        """Load processed data from file"""
        try:
            if not os.path.exists(filepath):
                raise DataException(f"Processed data file not found: {filepath}")
            
            df = pd.read_csv(filepath)
            df['date'] = pd.to_datetime(df['date'])
            # Integrity checks to guard cache corruption
            required_cols = ['date', 'tic', 'close']
            missing = [c for c in required_cols if c not in df.columns]
            if missing:
                raise DataException(f"Cache missing required columns: {missing}")
            if df.empty:
                raise DataException("Cache file is empty")
            # Ensure dates are monotonic per ticker and duplicates are reasonable
            dup_count = df.duplicated(subset=['date', 'tic']).sum()
            if dup_count > 0:
                self.logger.warning(f"Cache contains {dup_count} duplicate (date,tic) rows; deduplicating")
                df = df.drop_duplicates(subset=['date', 'tic'])
            # Basic sanity on price data
            if (df['close'] <= 0).any():
                raise DataException("Cache contains non-positive close prices")
            self.logger.info(f"Loaded processed data from {filepath}")
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to load processed data: {e}")
            raise DataException(f"Failed to load processed data: {e}")
    
    def get_data_statistics(self, df: pd.DataFrame) -> Dict:
        """Get data statistics for analysis"""
        try:
            stats = {
                'total_rows': len(df),
                'unique_tickers': df['tic'].nunique(),
                'date_range': {
                    'start': df['date'].min(),
                    'end': df['date'].max()
                },
                'missing_values': df.isnull().sum().to_dict(),
                'price_stats': {}
            }
            
            # Price statistics for each ticker
            for ticker in df['tic'].unique():
                ticker_data = df[df['tic'] == ticker]
                if 'close' in ticker_data.columns:
                    stats['price_stats'][ticker] = {
                        'mean': ticker_data['close'].mean(),
                        'std': ticker_data['close'].std(),
                        'min': ticker_data['close'].min(),
                        'max': ticker_data['close'].max(),
                        'rows': len(ticker_data)
                    }
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Failed to get data statistics: {e}")
            return {}

"""
Enhanced Data Manager for PPO Trading System
Implements advanced data validation, caching, and management features
"""

import os
import pandas as pd
import numpy as np
import pickle
import gzip
import logging
from datetime import datetime, timedelta
from pathlib import Path
import json

logger = logging.getLogger(__name__)

class EnhancedDataManager:
    """Enhanced data management with validation, caching, and backup features"""
    
    def __init__(self, config):
        self.config = config
        self.data_config = config.get('DATA_CONFIG', {})
        self.cache_config = config.get('CACHE_CONFIG', {})
        self.quality_config = self.data_config.get('data_quality', {})
        
        # Create directory structure
        self._create_directories()
        
        # Initialize data validation
        self.validation_results = {}
        self.data_quality_score = 0.0
        
    def _create_directories(self):
        """Create necessary directory structure"""
        directories = [
            'ppo/saved_data/raw_data',
            'ppo/saved_data/processed_data', 
            'ppo/saved_data/validation',
            'ppo/saved_models',
            'ppo/logs'
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
            logger.info(f" Created directory: {directory}")
    
    def validate_data_quality(self, data):
        """Enhanced data quality validation"""
        if not self.quality_config.get('enable_data_validation', False):
            return True
            
        logger.info(" Validating data quality...")
        
        validation_results = {
            'price_validation': self._validate_prices(data),
            'volume_validation': self._validate_volumes(data),
            'returns_validation': self._validate_returns(data),
            'consistency_validation': self._validate_consistency(data),
            'outlier_detection': self._detect_outliers(data),
            'missing_data_check': self._check_missing_data(data)
        }
        
        # Calculate overall quality score
        quality_score = sum(validation_results.values()) / len(validation_results)
        self.data_quality_score = quality_score
        
        threshold = self.quality_config.get('data_quality_threshold', 0.95)
        is_valid = quality_score >= threshold
        
        logger.info(f" Data quality score: {quality_score:.3f} (threshold: {threshold})")
        logger.info(f" Data validation: {'PASSED' if is_valid else 'FAILED'}")
        
        return is_valid
    
    def _validate_prices(self, data):
        """Validate price data"""
        try:
            min_price = self.quality_config.get('price_range_min', 0.01)
            max_price = self.quality_config.get('price_range_max', 10000)
            
            if 'close' in data.columns:
                price_valid = ((data['close'] >= min_price) & (data['close'] <= max_price)).all()
                return 1.0 if price_valid else 0.0
            return 1.0
        except Exception as e:
            logger.warning(f" Price validation error: {e}")
            return 0.0
    
    def _validate_volumes(self, data):
        """Validate volume data"""
        try:
            min_volume = self.quality_config.get('volume_range_min', 1000)
            
            if 'volume' in data.columns:
                volume_valid = (data['volume'] >= min_volume).all()
                return 1.0 if volume_valid else 0.0
            return 1.0
        except Exception as e:
            logger.warning(f" Volume validation error: {e}")
            return 0.0
    
    def _validate_returns(self, data):
        """Validate returns data"""
        try:
            max_return = self.quality_config.get('returns_range_max', 0.5)
            
            if 'returns' in data.columns:
                returns_valid = (abs(data['returns']) <= max_return).all()
                return 1.0 if returns_valid else 0.0
            return 1.0
        except Exception as e:
            logger.warning(f" Returns validation error: {e}")
            return 0.0
    
    def _validate_consistency(self, data):
        """Validate data consistency"""
        try:
            # Check for consistent date ranges
            if 'date' in data.columns:
                dates = pd.to_datetime(data['date'])
                date_range = dates.max() - dates.min()
                min_days = self.quality_config.get('min_data_points', 1000)
                
                if len(data) >= min_days:
                    return 1.0
                else:
                    logger.warning(f" Insufficient data points: {len(data)} < {min_days}")
                    return 0.5
            return 1.0
        except Exception as e:
            logger.warning(f" Consistency validation error: {e}")
            return 0.0
    
    def _detect_outliers(self, data):
        """Detect and handle outliers"""
        try:
            if not self.quality_config.get('enable_outlier_detection', False):
                return 1.0
                
            threshold = self.quality_config.get('outlier_threshold', 0.1)
            
            if 'close' in data.columns:
                # Calculate rolling statistics
                rolling_mean = data['close'].rolling(window=20).mean()
                rolling_std = data['close'].rolling(window=20).std()
                
                # Detect outliers
                outliers = abs(data['close'] - rolling_mean) > (threshold * rolling_std)
                outlier_ratio = outliers.sum() / len(data)
                
                if outlier_ratio < 0.05:  # Less than 5% outliers
                    return 1.0
                elif outlier_ratio < 0.10:  # Less than 10% outliers
                    return 0.8
                else:
                    logger.warning(f" High outlier ratio: {outlier_ratio:.3f}")
                    return 0.5
            return 1.0
        except Exception as e:
            logger.warning(f" Outlier detection error: {e}")
            return 0.0
    
    def _check_missing_data(self, data):
        """Check for missing data"""
        try:
            max_missing = self.quality_config.get('max_missing_days', 30)
            
            missing_ratio = data.isnull().sum().sum() / (len(data) * len(data.columns))
            
            if missing_ratio < 0.01:  # Less than 1% missing
                return 1.0
            elif missing_ratio < 0.05:  # Less than 5% missing
                return 0.8
            else:
                logger.warning(f" High missing data ratio: {missing_ratio:.3f}")
                return 0.5
        except Exception as e:
            logger.warning(f" Missing data check error: {e}")
            return 0.0
    
    def save_compressed_cache(self, data, filename, compress=True):
        """Save data with compression and backup"""
        try:
            cache_path = self.cache_config.get('data_cache_path', 'ppo/saved_data/processed_data/')
            file_path = Path(cache_path) / filename
            
            if compress:
                with gzip.open(f"{file_path}.gz", 'wb') as f:
                    pickle.dump(data, f)
                logger.info(f" Saved compressed cache: {file_path}.gz")
            else:
                with open(file_path, 'wb') as f:
                    pickle.dump(data, f)
                logger.info(f" Saved cache: {file_path}")
            
            # Create backup if enabled
            if self.quality_config.get('enable_data_backup', False):
                self._create_backup(file_path, data)
                
        except Exception as e:
            logger.error(f" Error saving cache: {e}")
    
    def load_compressed_cache(self, filename, compress=True):
        """Load data from compressed cache"""
        try:
            cache_path = self.cache_config.get('data_cache_path', 'ppo/saved_data/processed_data/')
            file_path = Path(cache_path) / filename
            
            if compress:
                cache_file = f"{file_path}.gz"
                if os.path.exists(cache_file):
                    with gzip.open(cache_file, 'rb') as f:
                        data = pickle.load(f)
                    logger.info(f" Loaded compressed cache: {cache_file}")
                    return data
            else:
                if os.path.exists(file_path):
                    with open(file_path, 'rb') as f:
                        data = pickle.load(f)
                    logger.info(f" Loaded cache: {file_path}")
                    return data
                    
            return None
        except Exception as e:
            logger.error(f" Error loading cache: {e}")
            return None
    
    def _create_backup(self, file_path, data):
        """Create backup of data file"""
        try:
            backup_dir = Path('ppo/saved_data/backups')
            backup_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_path = backup_dir / f"{file_path.stem}_{timestamp}.pkl"
            
            with open(backup_path, 'wb') as f:
                pickle.dump(data, f)
                
            logger.info(f" Created backup: {backup_path}")
            
            # Clean old backups
            self._cleanup_old_backups(backup_dir)
            
        except Exception as e:
            logger.warning(f" Backup creation failed: {e}")
    
    def _cleanup_old_backups(self, backup_dir, max_backups=5):
        """Clean up old backup files"""
        try:
            backup_files = list(backup_dir.glob('*.pkl'))
            backup_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            
            # Keep only the most recent backups
            for old_backup in backup_files[max_backups:]:
                old_backup.unlink()
                logger.info(f" Cleaned up old backup: {old_backup}")
                
        except Exception as e:
            logger.warning(f" Backup cleanup failed: {e}")
    
    def check_cache_age(self, filename):
        """Check if cache is still valid"""
        try:
            cache_path = self.cache_config.get('data_cache_path', 'ppo/saved_data/processed_data/')
            file_path = Path(cache_path) / filename
            
            if file_path.exists():
                file_age = datetime.now() - datetime.fromtimestamp(file_path.stat().st_mtime)
                max_age = timedelta(days=self.quality_config.get('cache_expiry_days', 7))
                
                return file_age < max_age
            return False
        except Exception as e:
            logger.warning(f" Cache age check failed: {e}")
            return False
    
    def get_data_summary(self, data):
        """Generate comprehensive data summary"""
        try:
            summary = {
                'total_records': len(data),
                'total_columns': len(data.columns),
                'date_range': {
                    'start': data['date'].min() if 'date' in data.columns else None,
                    'end': data['date'].max() if 'date' in data.columns else None
                },
                'tickers': data['tic'].unique().tolist() if 'tic' in data.columns else [],
                'missing_data': data.isnull().sum().to_dict(),
                'data_types': data.dtypes.to_dict(),
                'quality_score': self.data_quality_score,
                'validation_results': self.validation_results
            }
            
            return summary
        except Exception as e:
            logger.warning(f" Data summary generation failed: {e}")
            return {}

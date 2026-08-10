"""
FRED Data Provider for PPO Trading System
Provides economic data from Federal Reserve Economic Data (FRED) API
Replaces yfinance for economic indicators and market regime detection
"""

import os
import hashlib
import json
import time
import requests
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

class FREDProvider:
    """
    FRED API data provider for economic indicators and market regime detection
    Replaces yfinance for economic data in the PPO trading system
    """
    
    def __init__(self, api_key: str = None):
        """
        Initialize FRED provider
        
        Args:
            api_key: FRED API key (defaults to environment variable FRED_API_KEY)
        """
        self.api_key = api_key or os.getenv('FRED_API_KEY', '699bddb6f2275eef76586286fd101473')
        self.base_url = "https://api.stlouisfed.org/fred/series/observations"
        self.rate_limit_delay = 1  # FRED allows 1200 requests per hour
        self.calls_made = 0
        self.max_calls = int(os.getenv('FRED_MAX_CALLS', '50'))  # Conservative limit
        self.cache_ttl_seconds = int(os.getenv('FRED_CACHE_TTL_SECONDS', str(24*3600)))  # 1 day
        
        # Cache directory
        cache_root = os.getenv('FRED_CACHE_DIR', str(Path.home() / 'fred_cache'))
        self.cache_dir = Path(cache_root)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Critical economic indicators for PPO system
        self.critical_indicators = {
            # Market Regime Detection (Phase 1)
            'DGS10': '10-Year Treasury Constant Maturity Rate',
            'VIXCLS': 'CBOE Volatility Index: VIX',
            'FEDFUNDS': 'Federal Funds Effective Rate',
            'UNRATE': 'Unemployment Rate',
            
            # Risk Management (Phase 2)
            'CPIAUCSL': 'Consumer Price Index for All Urban Consumers',
            'PAYEMS': 'All Employees: Total Nonfarm Payrolls',
            'BAA10Y': 'Moody\'s Seasoned Baa Corporate Bond Yield',
            'TOTCI': 'Total Commercial and Industrial Loans',
            
            # Sector Analysis (Phase 3)
            'HOUST': 'Housing Starts: Total: New Privately Owned Housing Units',
            'INDPRO': 'Industrial Production: Total Index',
            'DCOILWTICO': 'Crude Oil Prices: West Texas Intermediate',
            'UMCSENT': 'University of Michigan: Consumer Sentiment'
        }
        
        logger.info(f"FRED Provider initialized with {len(self.critical_indicators)} critical indicators")
    
    def _get_cache_key(self, params: Dict) -> Path:
        """Generate cache key for parameters"""
        key_str = '&'.join(f"{k}={v}" for k, v in sorted(params.items()))
        key_hash = hashlib.sha256(key_str.encode('utf-8')).hexdigest()[:32]
        return self.cache_dir / f"{key_hash}.json"
    
    def _is_cache_valid(self, cache_path: Path) -> bool:
        """Check if cache is still valid"""
        try:
            if cache_path.exists():
                mtime = cache_path.stat().st_mtime
                return (time.time() - mtime) <= self.cache_ttl_seconds
        except Exception:
            pass
        return False
    
    def _get_cached_data(self, cache_path: Path) -> Optional[Dict]:
        """Load data from cache"""
        try:
            with open(cache_path, 'r') as f:
                return json.load(f)
        except Exception:
            return None
    
    def _save_to_cache(self, cache_path: Path, data: Dict):
        """Save data to cache"""
        try:
            with open(cache_path, 'w') as f:
                json.dump(data, f)
        except Exception:
            pass
    
    def _make_fred_request(self, series_id: str, **params) -> Dict:
        """Make FRED API request with caching and rate limiting"""
        cache_path = self._get_cache_key({'series_id': series_id, **params})
        
        # Check cache first
        if self._is_cache_valid(cache_path):
            cached_data = self._get_cached_data(cache_path)
            if cached_data:
                logger.debug(f"FRED Cache hit for {series_id}")
                return cached_data
        
        # Check rate limit
        if self.calls_made >= self.max_calls:
            logger.warning(f"FRED call budget reached ({self.max_calls}). Using cached data if available.")
            cached_data = self._get_cached_data(cache_path)
            if cached_data:
                return cached_data
            else:
                raise Exception(f"FRED call budget reached and no cached data available")
        
        # Make API request
        try:
            params['series_id'] = series_id
            params['api_key'] = self.api_key
            params['file_type'] = 'json'
            
            response = requests.get(self.base_url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            self.calls_made += 1
            
            # Cache the response
            self._save_to_cache(cache_path, data)
            
            # Rate limiting
            time.sleep(self.rate_limit_delay)
            
            return data
            
        except requests.exceptions.RequestException as e:
            logger.error(f"FRED API request failed for {series_id}: {e}")
            # Try to use cached data as fallback
            cached_data = self._get_cached_data(cache_path)
            if cached_data:
                logger.info(f"Using cached data for {series_id} due to API error")
                return cached_data
            else:
                raise
    
    def get_economic_data(
        self,
        series_id: str,
        limit: int = 1000,
        frequency: str = 'd',
        observation_start: Optional[Union[str, datetime]] = None,
        observation_end: Optional[Union[str, datetime]] = None,
        sort_order: str = 'asc'
    ) -> pd.DataFrame:
        """
        Get economic data for a specific series
        
        Args:
            series_id: FRED series ID
            limit: Number of observations to retrieve
            frequency: Data frequency (d=daily, m=monthly, q=quarterly)
            
        Returns:
            DataFrame with economic data
        """
        try:
            params: Dict[str, Any] = {
                'limit': limit,
                'frequency': frequency,
                'sort_order': sort_order,
            }
            if observation_start is not None:
                if isinstance(observation_start, datetime):
                    params['observation_start'] = observation_start.strftime('%Y-%m-%d')
                else:
                    params['observation_start'] = str(observation_start)
            if observation_end is not None:
                if isinstance(observation_end, datetime):
                    params['observation_end'] = observation_end.strftime('%Y-%m-%d')
                else:
                    params['observation_end'] = str(observation_end)
            data = self._make_fred_request(series_id, **params)
            
            if 'observations' not in data:
                logger.warning(f"No observations found for {series_id}")
                return pd.DataFrame()
            
            observations = data['observations']
            if not observations:
                logger.warning(f"Empty observations for {series_id}")
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame(observations)
            
            # Convert date and value columns
            df['date'] = pd.to_datetime(df['date'])
            df['value'] = pd.to_numeric(df['value'], errors='coerce')
            
            # Set date as index
            df.set_index('date', inplace=True)
            
            # Remove rows with missing values
            df = df.dropna(subset=['value'])
            
            # Ensure tz-naive, normalized DatetimeIndex for robust comparisons
            try:
                df.index = pd.DatetimeIndex(df.index).tz_localize(None)
            except TypeError:
                # Already tz-aware -> convert to naive
                df.index = pd.DatetimeIndex(df.index).tz_convert(None)
            df.index = df.index.normalize()
            
            logger.debug(f"Retrieved {len(df)} observations for {series_id}")
            return df
            
        except Exception as e:
            logger.error(f"Error retrieving data for {series_id}: {e}")
            return pd.DataFrame()
    
    def get_market_regime_indicators(self) -> Dict[str, float]:
        """
        Get market regime detection indicators
        
        Returns:
            Dictionary with latest values for market regime indicators
        """
        regime_indicators = {
            'DGS10': '10-Year Treasury',
            'VIXCLS': 'VIX Volatility',
            'FEDFUNDS': 'Federal Funds Rate',
            'UNRATE': 'Unemployment Rate'
        }
        
        results = {}
        
        for series_id, description in regime_indicators.items():
            try:
                df = self.get_economic_data(series_id, limit=10)
                if not df.empty:
                    latest_value = df['value'].iloc[-1]
                    results[series_id] = latest_value
                    logger.debug(f"{description}: {latest_value}")
                else:
                    results[series_id] = None
            except Exception as e:
                logger.error(f"Error getting {description}: {e}")
                results[series_id] = None
        
        return results
    
    def get_risk_management_indicators(self) -> Dict[str, float]:
        """
        Get risk management indicators
        
        Returns:
            Dictionary with latest values for risk management indicators
        """
        risk_indicators = {
            'CPIAUCSL': 'Consumer Price Index',
            'PAYEMS': 'Non-farm Payrolls',
            'BAA10Y': 'Corporate Bond Yield',
            'TOTCI': 'Commercial Loans'
        }
        
        results = {}
        
        for series_id, description in risk_indicators.items():
            try:
                df = self.get_economic_data(series_id, limit=10)
                if not df.empty:
                    latest_value = df['value'].iloc[-1]
                    results[series_id] = latest_value
                    logger.debug(f"{description}: {latest_value}")
                else:
                    results[series_id] = None
            except Exception as e:
                logger.error(f"Error getting {description}: {e}")
                results[series_id] = None
        
        return results
    
    def get_sector_indicators(self) -> Dict[str, float]:
        """
        Get sector analysis indicators
        
        Returns:
            Dictionary with latest values for sector indicators
        """
        sector_indicators = {
            'HOUST': 'Housing Starts',
            'INDPRO': 'Industrial Production',
            'DCOILWTICO': 'Oil Price',
            'UMCSENT': 'Consumer Sentiment'
        }
        
        results = {}
        
        for series_id, description in sector_indicators.items():
            try:
                df = self.get_economic_data(series_id, limit=10)
                if not df.empty:
                    latest_value = df['value'].iloc[-1]
                    results[series_id] = latest_value
                    logger.debug(f"{description}: {latest_value}")
                else:
                    results[series_id] = None
            except Exception as e:
                logger.error(f"Error getting {description}: {e}")
                results[series_id] = None
        
        return results
    
    def get_all_indicators(self) -> Dict[str, Dict[str, float]]:
        """
        Get all economic indicators for PPO system
        
        Returns:
            Dictionary with all indicator categories and their latest values
        """
        return {
            'market_regime': self.get_market_regime_indicators(),
            'risk_management': self.get_risk_management_indicators(),
            'sector_analysis': self.get_sector_indicators()
        }
    
    def get_risk_free_rate(self) -> float:
        """
        Get current risk-free rate (10-Year Treasury) for Sharpe ratio calculations
        
        Returns:
            Risk-free rate as decimal (e.g., 0.04 for 4%)
        """
        try:
            df = self.get_economic_data('DGS10', limit=1)
            if not df.empty:
                rate = df['value'].iloc[-1] / 100.0  # Convert percentage to decimal
                logger.debug(f"Risk-free rate: {rate:.4f}")
                return rate
            else:
                logger.warning("Could not retrieve risk-free rate, using default 0.02")
                return 0.02
        except Exception as e:
            logger.error(f"Error getting risk-free rate: {e}")
            return 0.02
    
    def get_volatility_context(self) -> float:
        """
        Get VIX volatility for position sizing context
        
        Returns:
            VIX value (typically 10-50)
        """
        try:
            df = self.get_economic_data('VIXCLS', limit=1)
            if not df.empty:
                vix = df['value'].iloc[-1]
                logger.debug(f"VIX volatility: {vix:.2f}")
                return vix
            else:
                logger.warning("Could not retrieve VIX, using default 20")
                return 20.0
        except Exception as e:
            logger.error(f"Error getting VIX: {e}")
            return 20.0
    
    def get_remaining_calls(self) -> int:
        """Get remaining API calls for this session"""
        return max(0, self.max_calls - self.calls_made)
    
    def reset_call_count(self):
        """Reset the API call counter"""
        self.calls_made = 0
        logger.info("FRED API call counter reset")
    
    def get_available_indicators(self) -> Dict[str, str]:
        """Get list of available indicators with descriptions"""
        return self.critical_indicators.copy()

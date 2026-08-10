"""
Alpha Vantage Data Provider for PPO Trading System
ONLY for technical indicators and economic data (Alpaca handles stock data)
"""

import os
import hashlib
import json
import time
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class AlphaVantageProvider:
    """Alpha Vantage provider for technical indicators and economic data only"""
    
    def __init__(self, api_key=None, cache_dir=None, max_calls_per_day=25):
        self.api_key = api_key or "CP66SNE2O9JWZ10C"  # Your API key
        self.base_url = "https://www.alphavantage.co/query"
        self.max_calls_per_day = max_calls_per_day
        self.calls_made_today = 0
        self.last_call_date = None
        
        # Cache setup
        cache_root = cache_dir or str(Path.home() / 'alpha_vantage_cache')
        self.cache_dir = Path(cache_root)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_ttl_seconds = 24 * 60 * 60  # 1 day
        
        # Load call count from cache
        self._load_call_count()
    
    def _load_call_count(self):
        """Load today's call count from cache"""
        call_count_file = self.cache_dir / 'call_count.json'
        try:
            if call_count_file.exists():
                with open(call_count_file, 'r') as f:
                    data = json.load(f)
                    last_date = data.get('date')
                    if last_date == datetime.now().strftime('%Y-%m-%d'):
                        self.calls_made_today = data.get('count', 0)
                        self.last_call_date = last_date
        except Exception:
            pass
    
    def _save_call_count(self):
        """Save today's call count to cache"""
        call_count_file = self.cache_dir / 'call_count.json'
        try:
            data = {
                'date': datetime.now().strftime('%Y-%m-%d'),
                'count': self.calls_made_today
            }
            with open(call_count_file, 'w') as f:
                json.dump(data, f)
        except Exception:
            pass
    
    def _get_cache_key(self, params):
        """Generate cache key for parameters"""
        key_str = '&'.join(f"{k}={v}" for k, v in sorted(params.items()))
        key_hash = hashlib.sha256(key_str.encode('utf-8')).hexdigest()[:32]
        return self.cache_dir / f"{key_hash}.json"
    
    def _is_cache_valid(self, cache_path):
        """Check if cache is still valid"""
        try:
            if cache_path.exists():
                mtime = cache_path.stat().st_mtime
                return (time.time() - mtime) <= self.cache_ttl_seconds
        except Exception:
            pass
        return False
    
    def _get_cached_data(self, cache_path):
        """Load data from cache"""
        try:
            with open(cache_path, 'r') as f:
                return json.load(f)
        except Exception:
            return None
    
    def _save_to_cache(self, cache_path, data):
        """Save data to cache"""
        try:
            with open(cache_path, 'w') as f:
                json.dump(data, f)
        except Exception:
            pass
    
    def _make_api_request(self, params):
        """Make API request with caching and rate limiting"""
        cache_path = self._get_cache_key(params)
        
        # Check cache first
        if self._is_cache_valid(cache_path):
            cached_data = self._get_cached_data(cache_path)
            if cached_data:
                logger.debug(f" Cache hit for {params.get('function', 'unknown')}")
                return cached_data
        
        # Check rate limit
        if self.calls_made_today >= self.max_calls_per_day:
            logger.warning(f" Daily API limit reached ({self.max_calls_per_day}). Using cached data if available.")
            cached_data = self._get_cached_data(cache_path)
            if cached_data:
                return cached_data
            else:
                raise Exception(f"Daily API limit reached and no cached data available")
        
        # Make API request
        try:
            logger.debug(f"Making API call: {params.get('function', 'unknown')}")
            response = requests.get(self.base_url, params=params, timeout=30)
            data = response.json()
            
            # Check for errors
            if 'Error Message' in data or 'Note' in data:
                error_msg = data.get('Error Message') or data.get('Note', 'Unknown error')
                logger.error(f" API Error: {error_msg}")
                raise Exception(f"Alpha Vantage API error: {error_msg}")
            
            # Increment call count
            self.calls_made_today += 1
            self._save_call_count()
            
            # Save to cache
            self._save_to_cache(cache_path, data)
            
            # Rate limiting delay
            time.sleep(0.2)  # 5 requests per minute = 12 second delay
            
            return data
            
        except Exception as e:
            logger.error(f" API request failed: {e}")
            # Try to return cached data even if expired
            cached_data = self._get_cached_data(cache_path)
            if cached_data:
                logger.info("Using expired cached data as fallback")
                return cached_data
            raise
    
    def get_technical_indicator(self, symbol, indicator, interval='daily', **params):
        """Get technical indicator data (Alpaca doesn't provide these)"""
        try:
            # Add common parameters
            request_params = {
                'function': indicator,
                'symbol': symbol,
                'interval': interval,
                'series_type': 'close',
                'apikey': self.api_key,
                **params
            }
            
            data = self._make_api_request(request_params)
            
            # Find the technical analysis block
            ta_key = None
            for key in data.keys():
                if key.startswith('Technical Analysis'):
                    ta_key = key
                    break
            
            if not ta_key:
                raise Exception(f"No technical analysis data found for {indicator}")
            
            # Convert to DataFrame
            df = pd.DataFrame.from_dict(data[ta_key], orient='index')
            df.index = pd.to_datetime(df.index)
            
            # Convert to numeric
            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            return df
            
        except Exception as e:
            logger.error(f" Failed to get {indicator} for {symbol}: {e}")
            raise
    
    def get_economic_data(self, indicator, interval='monthly'):
        """Get economic data (Alpaca doesn't provide these)"""
        try:
            params = {
                'function': indicator,
                'interval': interval,
                'apikey': self.api_key
            }
            
            data = self._make_api_request(params)
            
            if 'data' not in data:
                raise Exception(f"No data found for {indicator}")
            
            # Convert to DataFrame
            df = pd.DataFrame(data['data'])
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
            elif 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
            
            # Convert value to numeric
            if 'value' in df.columns:
                df['value'] = pd.to_numeric(df['value'], errors='coerce')
            
            return df
            
        except Exception as e:
            logger.error(f" Failed to get economic data for {indicator}: {e}")
            raise
    
    def get_news_sentiment(self, tickers=None, topics=None, limit=50):
        """Get news sentiment (Alpaca doesn't provide this)"""
        try:
            params = {
                'function': 'NEWS_SENTIMENT',
                'apikey': self.api_key,
                'limit': limit
            }
            
            if tickers:
                params['tickers'] = tickers
            if topics:
                params['topics'] = topics
            
            data = self._make_api_request(params)
            
            if 'feed' not in data:
                raise Exception("No news feed data found")
            
            return data['feed']
            
        except Exception as e:
            logger.error(f" Failed to get news sentiment: {e}")
            raise
    
    def get_market_status(self):
        """Get market status (Alpaca doesn't provide this)"""
        try:
            params = {
                'function': 'MARKET_STATUS',
                'apikey': self.api_key
            }
            
            data = self._make_api_request(params)
            
            if 'markets' not in data:
                raise Exception("No market status data found")
            
            return data['markets']
            
        except Exception as e:
            logger.error(f" Failed to get market status: {e}")
            raise
    
    def get_available_indicators(self):
        """Get list of available technical indicators"""
        return [
            'SMA', 'EMA', 'WMA', 'DEMA', 'TEMA', 'TRIMA', 'KAMA', 'T3',
            'RSI', 'STOCH', 'STOCHF', 'STOCHRSI', 'WILLR', 'MOM', 'ROC', 'ROCR',
            'CCI', 'CMO', 'MFI', 'TRIX', 'ULTOSC', 'ATR', 'NATR', 'TRANGE',
            'BBANDS', 'OBV', 'AD', 'ADOSC', 'BOP', 'ADX', 'ADXR', 'APO', 'PPO',
            'AROON', 'AROONOSC', 'DX', 'MINUS_DI', 'PLUS_DI', 'MINUS_DM', 'PLUS_DM',
            'HT_TRENDLINE', 'HT_SINE', 'HT_TRENDMODE', 'HT_DCPERIOD', 'HT_DCPHASE', 'HT_PHASOR',
            'SAR', 'MIDPOINT', 'MIDPRICE'
        ]
    
    def get_available_economic_indicators(self):
        """Get list of available economic indicators"""
        return [
            'TREASURY_YIELD', 'FEDERAL_FUNDS_RATE', 'CPI', 'REAL_GDP',
            'UNEMPLOYMENT', 'NONFARM_PAYROLL'
        ]
    
    def get_remaining_calls(self):
        """Get remaining API calls for today"""
        return max(0, self.max_calls_per_day - self.calls_made_today)
    
    def reset_daily_count(self):
        """Reset daily call count (useful for testing)"""
        self.calls_made_today = 0
        self._save_call_count()
        logger.info("Daily call count reset")

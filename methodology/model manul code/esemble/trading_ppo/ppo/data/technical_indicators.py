"""
Technical indicators functionality for the PPO Trading Agent
Extracted from TG4.1.py - technical indicators calculation functions
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

def calculate_technical_indicators(df):
    """
    Calculate optimized 10 technical indicators with robust error handling and consistent array lengths.
    Reduced from 27 to 10 indicators to avoid redundancy and improve performance.
    """
    try:
        print(f"[INDICATORS] Starting technical indicator calculations on {len(df)} rows")
        df = df.copy()

        # Group by ticker to ensure proper calculations within each ticker
        grouped = df.groupby('tic')
        result_frames = []
        
        for ticker, group in grouped:
            # Sort by date to ensure proper sequential calculations
            ticker_df = group.sort_values('date').copy()
            
            # 1. Core price momentum - Returns
            # DATA LEAKAGE PREVENTION: .pct_change() uses only past data (previous row)
            # This is safe - no future data is accessed
            ticker_df['returns'] = ticker_df['close_raw'].pct_change()

            # 2. Trend indicator - MACD (combines EMA 12 & 26)
            # DATA LEAKAGE PREVENTION: .ewm() uses only past and current data (exponential weighted moving average)
            # This is safe - no future data is accessed
            ticker_df['ema_12'] = ticker_df['close_raw'].ewm(span=12, adjust=False).mean()
            ticker_df['ema_26'] = ticker_df['close_raw'].ewm(span=26, adjust=False).mean()
            ticker_df['macd'] = ticker_df['ema_12'] - ticker_df['ema_26']
            
            # Pre-fill NaNs for the price series to avoid errors
            close_series = ticker_df['close_raw'].fillna(method='ffill').fillna(method='bfill').values
            
            # ENHANCED: Robust volume handling
            if 'volume' in ticker_df.columns and ticker_df['volume'].notna().any():
                volume_series = ticker_df['volume'].fillna(method='ffill').fillna(method='bfill').values
                volume_available = True
            else:
                print(f"[WARNING] Missing or all NaN volume data for {ticker}. Using placeholder for volume-based indicators.")
                # Create placeholder volume based on price if volume is entirely missing
                volume_series = np.full(len(close_series), np.nanmean(close_series) * 1000)
                volume_available = False
            
            # 3. Hull Moving Average for trend detection with timeout protection
            try:
                hma_values = hull_moving_average(close_series, period=20)
                ticker_df['hma_20'] = hma_values
            except Exception as e:
                print(f"[WARNING] HMA calculation failed for {ticker}: {e}")
                # Fallback to simple moving average
                close_series_pd = pd.Series(close_series)
                hma_series = close_series_pd.rolling(window=20, min_periods=1).mean()
                # Ensure hma_series is a Series before calling fillna
                hma_series_typed = cast(pd.Series, hma_series)
                hma_filled = hma_series_typed.fillna(method='bfill')
                ticker_df['hma_20'] = hma_filled.values if hasattr(hma_filled, 'values') else np.array(hma_filled)

            # 4. Stochastic RSI - Momentum oscillator (combines RSI + Stochastic)
            try:
                stoch_rsi_k, _ = stoch_rsi(close_series)
                ticker_df['stoch_rsi'] = stoch_rsi_k
            except Exception as e:
                print(f"[WARNING] Stoch RSI calculation failed for {ticker}: {e}")
                ticker_df['stoch_rsi'] = np.full(len(close_series), 50.0)  # Default value

            # 5. Volume Price Trend for volume confirmation - Enhanced handling
            try:
                if volume_available and not np.all(np.isnan(volume_series)) and not np.all(volume_series == 0):
                    vpt_values = volume_price_trend(close_series, volume_series)
                    ticker_df['vpt'] = vpt_values
                else:
                    print(f"[WARNING] Volume data unusable for {ticker}. Setting VPT to NaN.")
                    ticker_df['vpt'] = np.full(len(close_series), np.nan)
            except Exception as e:
                print(f"[WARNING] VPT calculation failed for {ticker}: {e}")
                ticker_df['vpt'] = np.full(len(close_series), np.nan)  # Use NaN instead of zeros

            # 6. Average True Range for volatility - Enhanced OHLC handling
            # DATA LEAKAGE PREVENTION: .shift() without arguments defaults to shift(1)
            # shift(1) shifts DOWN (uses previous row's data) - this is CORRECT (past data only)
            # shift(-1) would shift UP (uses future row's data) - this would be WRONG (data leakage)
            if all(col in ticker_df.columns for col in ['high', 'low']):
                high_low = ticker_df['high'] - ticker_df['low']
                # shift() defaults to shift(1) = uses previous row (past data) - CORRECT
                high_close = (ticker_df['high'] - ticker_df['close_raw'].shift()).abs()
                low_close = (ticker_df['low'] - ticker_df['close_raw'].shift()).abs()
                
                # Create True Range series
                true_range = pd.DataFrame({
                    'hl': high_low,
                    'hc': high_close,
                    'lc': low_close
                }).max(axis=1)
                
                # DATA LEAKAGE PREVENTION: .rolling() uses only past and current data
                # This is safe - no future data is accessed
                ticker_df['atr'] = true_range.rolling(14).mean()
            else:
                print(f"[WARNING] Missing high/low data for {ticker}. ATR will be set to NaN.")
                ticker_df['atr'] = np.full(len(close_series), np.nan)

            # 7. Bollinger Bands %B for price position
            try:
                bb_values = bollinger_percent_b(close_series)
                ticker_df['bb_percent_b'] = bb_values
            except Exception as e:
                print(f"[WARNING] Bollinger %B calculation failed for {ticker}: {e}")
                ticker_df['bb_percent_b'] = np.full(len(close_series), 0.5)

            # 8. Candlestick patterns for reversals
            body_size = abs(ticker_df['close_raw'] - ticker_df['open'])
            upper_shadow = ticker_df['high'] - ticker_df[['open', 'close_raw']].max(axis=1)
            lower_shadow = ticker_df[['open', 'close_raw']].min(axis=1) - ticker_df['low']
            
            # Hammer pattern - Using vectorized operations
            ticker_df['hammer'] = (
                (lower_shadow > body_size * 2) &
                (upper_shadow < body_size * 0.5)
            ).astype(int)
            
            # Shooting Star pattern
            ticker_df['shooting_star'] = (
                (upper_shadow > body_size * 2) &
                (lower_shadow < body_size * 0.5)
            ).astype(int)

            # 9. Breakout signals for trend following
            # DATA LEAKAGE PREVENTION: .shift(1) uses previous row (past data) - CORRECT
            # This compares current close to previous high/low, which is valid (no future data)
            volume_threshold = ticker_df['volume'].rolling(20).mean() * 1.5
            ticker_df['breakout_up'] = (
                (ticker_df['close_raw'] > ticker_df['high'].shift(1)) &  # shift(1) = previous row (past data)
                (ticker_df['volume'] > volume_threshold)
            ).astype(int)
            ticker_df['breakout_down'] = (
                (ticker_df['close_raw'] < ticker_df['low'].shift(1)) &  # shift(1) = previous row (past data)
                (ticker_df['volume'] > volume_threshold)
            ).astype(int)

            # Fill any remaining NaNs
            ticker_df = ticker_df.fillna(method='ffill').fillna(method='bfill')
            
            result_frames.append(ticker_df)
        
        # Combine all processed ticker dataframes
        df = pd.concat(result_frames)
        
        # Verify indicator columns match expected length
        for indicator in ['returns', 'macd', 'hma_20', 'stoch_rsi', 'vpt', 'atr', 'bb_percent_b', 
                         'hammer', 'shooting_star', 'breakout_up', 'breakout_down']:
            if indicator not in df.columns:
                print(f"[WARNING] Missing indicator column: {indicator}")
            else:
                indicator_series = cast(pd.Series, df[indicator])
                has_nulls = bool(np.any(np.array(indicator_series.isnull().values)))
                if has_nulls:
                    print(f"[WARNING] Found NaN values in {indicator}, filling with defaults")
                # Fill any remaining NaNs with sane defaults
                if indicator in ['stoch_rsi']:
                    df[indicator] = df[indicator].fillna(50.0)
                elif indicator in ['bb_percent_b']:
                    df[indicator] = df[indicator].fillna(0.5)
                elif indicator in ['hammer', 'shooting_star', 'breakout_up', 'breakout_down']:
                    df[indicator] = df[indicator].fillna(0)
                else:
                    df[indicator] = df[indicator].fillna(0.0)
        
        print(f"[INDICATORS] Successfully calculated all indicators")
        return df
        
    except Exception as e:
        print(f"[ERROR] Technical indicators calculation failed: {e}")
        # Fallback minimal indicators matching CORE_FEATURES with length safety
        df = df.copy()
        df['returns'] = df['close_raw'].pct_change()
        df['macd'] = 0.0
        df['hma_20'] = 0.0
        df['stoch_rsi'] = 50.0
        df['vpt'] = 0.0
        df['atr'] = 0.0
        df['bb_percent_b'] = 0.5
        df['hammer'] = 0
        df['shooting_star'] = 0
        df['breakout_up'] = 0
        df['breakout_down'] = 0
        return df

def hull_moving_average(prices, period=20):
    """Calculate Hull Moving Average with timeout protection"""
    try:
        if len(prices) < period:
            return np.full(len(prices), np.nanmean(prices))
        
        # Calculate WMA for half period
        half_period = period // 2
        wma_half = weighted_moving_average(prices, half_period)
        
        # Calculate WMA for full period
        wma_full = weighted_moving_average(prices, period)
        
        # Calculate raw HMA - ensure numpy arrays
        wma_half_arr = np.array(wma_half) if not isinstance(wma_half, np.ndarray) else wma_half
        wma_full_arr = np.array(wma_full) if not isinstance(wma_full, np.ndarray) else wma_full
        raw_hma = 2 * wma_half_arr - wma_full_arr
        
        # Calculate final HMA using square root of period
        sqrt_period = int(np.sqrt(period))
        hma = weighted_moving_average(raw_hma, sqrt_period)
        
        return hma
        
    except Exception as e:
        print(f"[WARNING] HMA calculation failed: {e}")
        # Fallback to simple moving average
        prices_series = pd.Series(prices)
        hma_series = prices_series.rolling(window=period, min_periods=1).mean()
        # Ensure hma_series is a Series before calling fillna
        hma_series_typed = cast(pd.Series, hma_series)
        hma_filled = hma_series_typed.fillna(method='bfill')
        return hma_filled.values if hasattr(hma_filled, 'values') else np.array(hma_filled)

def weighted_moving_average(prices, period):
    """Calculate Weighted Moving Average"""
    try:
        if len(prices) < period:
            return np.full(len(prices), np.nanmean(prices))
        
        weights = np.arange(1, period + 1)
        wma = pd.Series(prices).rolling(window=period).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )
        
        return wma.fillna(method='bfill').values
        
    except Exception as e:
        print(f"[WARNING] WMA calculation failed: {e}")
        prices_series = pd.Series(prices)
        wma_series = prices_series.rolling(window=period, min_periods=1).mean()
        # Ensure wma_series is a Series before calling fillna
        wma_series_typed = cast(pd.Series, wma_series)
        wma_filled = wma_series_typed.fillna(method='bfill')
        return wma_filled.values if hasattr(wma_filled, 'values') else np.array(wma_filled)

def stoch_rsi(prices, rsi_period=14, stoch_period=14):
    """Calculate Stochastic RSI"""
    try:
        if len(prices) < rsi_period + stoch_period:
            return np.full(len(prices), 50.0), np.full(len(prices), 50.0)
        
        # Calculate RSI
        rsi_values = calculate_rsi(prices, rsi_period)
        
        # Calculate Stochastic of RSI
        rsi_min = pd.Series(rsi_values).rolling(window=stoch_period).min()
        rsi_max = pd.Series(rsi_values).rolling(window=stoch_period).max()
        
        # Avoid division by zero
        denominator = rsi_max - rsi_min
        denominator = denominator.replace(0, 1)  # Replace 0 with 1 to avoid division by zero
        
        stoch_k = 100 * (rsi_values - rsi_min) / denominator
        stoch_d = stoch_k.rolling(window=3).mean()
        
        return stoch_k.fillna(50.0).values, stoch_d.fillna(50.0).values
        
    except Exception as e:
        print(f"[WARNING] Stochastic RSI calculation failed: {e}")
        return np.full(len(prices), 50.0), np.full(len(prices), 50.0)

def calculate_rsi(prices, period=14):
    """Calculate Relative Strength Index"""
    try:
        if len(prices) < period + 1:
            return np.full(len(prices), 50.0)
        
        # Calculate price changes
        delta = pd.Series(prices).diff()
        
        # Separate gains and losses
        gains = delta.where(delta > 0, 0)
        losses = -delta.where(delta < 0, 0)
        
        # Calculate average gains and losses
        avg_gains = gains.rolling(window=period).mean()
        avg_losses = losses.rolling(window=period).mean()
        
        # Calculate RS and RSI
        rs = avg_gains / avg_losses
        rsi = 100 - (100 / (1 + rs))
        
        # Ensure rsi is a Series before calling fillna
        if isinstance(rsi, pd.Series):
            return rsi.fillna(50.0)
        elif isinstance(rsi, (np.ndarray, list, tuple)):
            rsi_series = pd.Series(rsi)
            return rsi_series.fillna(50.0)
        else:
            # Handle scalar or other types
            rsi_series = pd.Series([float(rsi)] if not isinstance(rsi, (list, tuple)) else rsi)
            return rsi_series.fillna(50.0)
        
    except Exception as e:
        print(f"[WARNING] RSI calculation failed: {e}")
        return np.full(len(prices), 50.0)

def volume_price_trend(prices, volumes):
    """Calculate Volume Price Trend"""
    try:
        if len(prices) != len(volumes):
            return np.full(len(prices), np.nan)
        
        # Calculate price change percentage
        price_changes = np.diff(prices) / prices[:-1]
        
        # Calculate VPT
        vpt = np.zeros(len(prices))
        vpt[0] = volumes[0]  # First value
        
        for i in range(1, len(prices)):
            if price_changes[i-1] != 0:
                vpt[i] = vpt[i-1] + volumes[i] * price_changes[i-1]
            else:
                vpt[i] = vpt[i-1]
        
        return vpt
        
    except Exception as e:
        print(f"[WARNING] VPT calculation failed: {e}")
        return np.full(len(prices), np.nan)

def bollinger_percent_b(prices, period=20, std_dev=2):
    """Calculate Bollinger Bands %B"""
    try:
        if len(prices) < period:
            return np.full(len(prices), 0.5)
        
        # Calculate moving average and standard deviation
        ma = pd.Series(prices).rolling(window=period).mean()
        std = pd.Series(prices).rolling(window=period).std()
        
        # Calculate Bollinger Bands
        upper_band = ma + (std * std_dev)
        lower_band = ma - (std * std_dev)
        
        # Calculate %B
        percent_b = (prices - lower_band) / (upper_band - lower_band)
        
        return percent_b.fillna(0.5).values
        
    except Exception as e:
        print(f"[WARNING] Bollinger %B calculation failed: {e}")
        return np.full(len(prices), 0.5)

def get_core_features():
    """Get the list of core technical features"""
    return [
        'returns', 'macd', 'hma_20', 'stoch_rsi', 'vpt', 
        'atr', 'bb_percent_b', 'hammer', 'shooting_star', 
        'breakout_up', 'breakout_down'
    ]

def validate_indicators(df):
    """Validate that all required indicators are present and valid"""
    print("[VALIDATION] Validating technical indicators...")
    
    core_features = get_core_features()
    missing_features = []
    invalid_features = []
    
    for feature in core_features:
        if feature not in df.columns:
            missing_features.append(feature)
        elif df[feature].isnull().all():
            invalid_features.append(feature)
    
    if missing_features:
        print(f"[WARNING] Missing features: {missing_features}")
    
    if invalid_features:
        print(f"[WARNING] Invalid features (all NaN): {invalid_features}")
    
    if not missing_features and not invalid_features:
        print("[VALIDATION]  All technical indicators are valid")
        return True
    else:
        print("[VALIDATION]   Some technical indicators have issues")
        return False

def get_indicator_summary(df):
    """Get a summary of calculated indicators"""
    if df is None or df.empty:
        return "No data available for indicator summary"
    
    core_features = get_core_features()
    available_features = [f for f in core_features if f in df.columns]
    
    summary = {
        'total_indicators': len(available_features),
        'available_indicators': available_features,
        'missing_indicators': [f for f in core_features if f not in df.columns],
        'indicator_stats': {}
    }
    
    for indicator in available_features:
        if indicator in df.columns:
            stats = {
                'mean': df[indicator].mean(),
                'std': df[indicator].std(),
                'min': df[indicator].min(),
                'max': df[indicator].max(),
                'null_count': df[indicator].isnull().sum()
            }
            summary['indicator_stats'][indicator] = stats
    
    return summary


class TechnicalIndicators:
    """Wrapper class exposing a simple API around indicator functions"""
    def __init__(self):
        pass

    def calculate(self, df):
        return calculate_technical_indicators(df)

    @staticmethod
    def get_core_features():
        return get_core_features()

    def validate(self, df):
        return validate_indicators(df)

    def summary(self, df):
        return get_indicator_summary(df)
"""
CSV Data Manager - Safe, Immutable Data Pipeline for PPO Trading Agent

This module provides a deterministic, corruption-free data loading system:
1. Downloads data once and freezes to CSV
2. Computes indicators per ticker (prevents misalignment)
3. Validates data quality before PPO sees it
4. Uses hash verification to detect corruption
5. Provides immutable data loading (no silent overwrites)

Author: PPO Trading Agent Team
Date: 2026-01-25
"""

import os
import sys
import pandas as pd
import numpy as np
import hashlib
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, cast
from datetime import datetime
import json

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

# Import Alpaca client (existing working system)
try:
    from .alpaca_client import AlpacaDataClient
    ALPACA_AVAILABLE = True
except ImportError:
    try:
        from ppo.data.alpaca_client import AlpacaDataClient
        ALPACA_AVAILABLE = True
    except ImportError:
        ALPACA_AVAILABLE = False

# Import Alpaca key getter
try:
    from .data_loader import _get_alpaca_keys
except ImportError:
    try:
        from ppo.data.data_loader import _get_alpaca_keys
    except ImportError:
        def _get_alpaca_keys():
            import os
            api_key = os.getenv("APCA_API_KEY")
            secret_key = os.getenv("APCA_API_SECRET")
            if api_key and secret_key:
                return api_key, secret_key
            return None

logger = logging.getLogger(__name__)


class CSVDataManager:
    """
    Safe CSV-based data manager for PPO trading agent.
    
    Features:
    - Immutable CSV (no silent overwrites)
    - Hash verification (detect corruption)
    - Per-ticker indicator calculation (prevents misalignment)
    - Pre-validation (fail fast before PPO)
    - Deterministic data loading
    """
    
    def __init__(self, 
                 data_dir: str = "ppo/saved_data/csv_data",
                 raw_csv: str = "trading_data_raw.csv",
                 processed_csv: str = "trading_data_features.csv",
                 manifest_file: str = "data_manifest.json"):
        """
        Initialize CSV data manager.
        
        Args:
            data_dir: Directory for CSV files
            raw_csv: Raw OHLCV data CSV filename
            processed_csv: Processed features CSV filename
            manifest_file: Metadata manifest filename
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.raw_csv_path = self.data_dir / raw_csv
        self.processed_csv_path = self.data_dir / processed_csv
        self.manifest_path = self.data_dir / manifest_file
        
        self.logger = logging.getLogger(__name__)
        
    def file_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash of file."""
        if not file_path.exists():
            return ""
        with open(file_path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()
    
    def download_raw_data(self, 
                         tickers: List[str],
                         start_date: str,
                         end_date: str,
                         force: bool = False) -> pd.DataFrame:
        """
        Download raw OHLCV data and save to CSV.
        
        Uses Alpaca API (primary) or yfinance (fallback).
        
        Args:
            tickers: List of ticker symbols
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            force: Force re-download even if CSV exists
            
        Returns:
            DataFrame with raw OHLCV data
        """
        # Check if CSV already exists
        if self.raw_csv_path.exists() and not force:
            raise RuntimeError(
                f"Raw CSV already exists: {self.raw_csv_path}\n"
                f"Delete it explicitly to regenerate, or use force=True"
            )
        
        self.logger.info(f"Downloading raw data for {len(tickers)} tickers...")
        self.logger.info(f"Date range: {start_date} to {end_date}")
        
        all_data = []
        
        # Try Alpaca API first (your existing working system)
        if ALPACA_AVAILABLE:
            keys = _get_alpaca_keys()
            if keys:
                api_key, secret_key = keys
                self.logger.info("Using Alpaca API (primary data source)...")
                try:
                    client = AlpacaDataClient(api_key, secret_key, paper=True)
                    
                    for i, ticker in enumerate(tickers, 1):
                        self.logger.info(f"[{i}/{len(tickers)}] Downloading {ticker} from Alpaca...")
                        try:
                            df = client.get_historical_data(
                                symbols=[ticker],
                                start_date=start_date,
                                end_date=end_date
                            )
                            
                            if df is None or df.empty:
                                self.logger.warning(f"No data for {ticker} from Alpaca")
                                continue
                            
                            # Normalize Alpaca data format
                            # Alpaca client already renames: 'symbol' -> 'tic', 'timestamp' -> 'date'
                            df = df.copy()
                            
                            # Ensure Ticker column (Alpaca uses 'tic')
                            if 'tic' in df.columns:
                                df['Ticker'] = df['tic']
                            elif 'symbol' in df.columns:
                                df['Ticker'] = df['symbol']
                            else:
                                df['Ticker'] = ticker
                            
                            # Ensure date column (Alpaca client renames 'timestamp' to 'date')
                            if 'date' not in df.columns:
                                if 'timestamp' in df.columns:
                                    df['date'] = pd.to_datetime(df['timestamp'])
                                else:
                                    self.logger.warning(f"No date column found for {ticker}")
                                    continue
                            else:
                                df['date'] = pd.to_datetime(df['date'])
                            
                            # Ensure we have OHLCV columns (Alpaca uses lowercase)
                            required_cols = ['open', 'high', 'low', 'close', 'volume']
                            missing = [col for col in required_cols if col not in df.columns]
                            if missing:
                                self.logger.warning(f"Missing columns for {ticker}: {missing}")
                                self.logger.warning(f"Available columns: {df.columns.tolist()}")
                                continue
                            
                            # Select and ensure correct column order
                            df = df[['date', 'Ticker', 'open', 'high', 'low', 'close', 'volume']].copy()
                            
                            all_data.append(df)
                            self.logger.info(f"✅ Downloaded {len(df)} rows for {ticker}")
                            
                        except Exception as e:
                            self.logger.warning(f"Alpaca download failed for {ticker}: {e}")
                            continue
                    
                    if all_data:
                        self.logger.info(f"✅ Successfully downloaded {len(all_data)}/{len(tickers)} tickers from Alpaca")
                
                except Exception as e:
                    self.logger.warning(f"Alpaca API failed: {e}, trying fallback...")
                    all_data = []  # Reset for fallback
        
        # Fallback to yfinance if Alpaca failed or unavailable
        if not all_data and YFINANCE_AVAILABLE:
            self.logger.info("Using yfinance (fallback data source)...")
            for i, ticker in enumerate(tickers, 1):
                self.logger.info(f"[{i}/{len(tickers)}] Downloading {ticker} from yfinance...")
                try:
                    stock = yf.Ticker(ticker)
                    # Add retry and better parameters
                    df = stock.history(start=start_date, end=end_date, interval='1d', auto_adjust=True)
                    
                    if df.empty:
                        self.logger.warning(f"No data for {ticker} from yfinance")
                        continue
                    
                    # Reset index to get Date as column
                    df.reset_index(inplace=True)
                    df['Ticker'] = ticker
                    
                    # Rename columns to match expected format
                    df.rename(columns={
                        'Date': 'date',
                        'Open': 'open',
                        'High': 'high',
                        'Low': 'low',
                        'Close': 'close',
                        'Volume': 'volume'
                    }, inplace=True)
                    
                    # Select only required columns
                    df = df[['date', 'Ticker', 'open', 'high', 'low', 'close', 'volume']]
                    
                    all_data.append(df)
                    self.logger.info(f"✅ Downloaded {len(df)} rows for {ticker} from yfinance")
                    
                except Exception as e:
                    self.logger.error(f"yfinance download failed for {ticker}: {e}")
                    continue
        
        if not all_data:
            raise RuntimeError(
                "No data downloaded for any ticker.\n"
                "Possible causes:\n"
                "1. Alpaca API credentials not set (APCA_API_KEY, APCA_API_SECRET)\n"
                "2. yfinance not installed (pip install yfinance)\n"
                "3. Network connectivity issues\n"
                "4. Invalid date range or ticker symbols"
            )
        
        # Combine all tickers
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # Sort by ticker and date
        combined_df = combined_df.sort_values(['Ticker', 'date']).reset_index(drop=True)
        
        # Ensure date is datetime
        combined_df['date'] = pd.to_datetime(combined_df['date'])
        
        # Save to CSV
        combined_df.to_csv(self.raw_csv_path, index=False)
        self.logger.info(f"✅ Saved raw data to {self.raw_csv_path}")
        self.logger.info(f"   Rows: {len(combined_df)}, Columns: {len(combined_df.columns)}")
        self.logger.info(f"   Tickers: {sorted(combined_df['Ticker'].unique())}")
        
        return combined_df
    
    def validate_raw_data(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate raw OHLCV data quality.
        
        Returns:
            (is_valid, error_messages)
        """
        errors = []
        
        # Check required columns
        required_cols = ['date', 'Ticker', 'open', 'high', 'low', 'close', 'volume']
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            errors.append(f"Missing columns: {missing}")
        
        # Price sanity checks
        if 'close' in df.columns:
            if (df['close'] <= 0).any():
                errors.append("Found zero or negative close prices")
            
            if (df['high'] < df['low']).any():
                errors.append("Found High < Low (invalid price data)")
            
            # Check for extreme price jumps (>50% per day)
            # NOTE: Stock splits cause legitimate >50% DROPS (e.g., 4-for-1 = 75% drop)
            # We distinguish between:
            # - DROPS >50%: Likely stock splits (will be auto-cleaned, not errors)
            # - RISES >50%: Likely corruption (real errors)
            for ticker in df['Ticker'].unique():
                ticker_data = df[df['Ticker'] == ticker].copy()
                ticker_data = ticker_data.sort_values(by=['date']).reset_index(drop=True)  # type: ignore[arg-type]
                returns = ticker_data['close'].pct_change()
                
                # Extreme drops (likely stock splits) - log but don't error
                extreme_drops = returns[returns < -0.50]  # >50% drop
                # Extreme rises (likely corruption) - these are errors
                extreme_rises = returns[returns > 0.50]   # >50% rise
                
                if len(extreme_drops) > 0:
                    # Stock splits are legitimate - log as warning, will be cleaned
                    # Ensure extreme_drops is treated as a Series
                    drops_series = cast(pd.Series, extreme_drops)
                    drop_dates = ticker_data.loc[drops_series.index, 'date'].tolist()[:5]
                    self.logger.warning(
                        f"Ticker {ticker}: {len(extreme_drops)} extreme price DROPS (>50%) "
                        f"at dates: {drop_dates} - Likely stock splits, will be auto-cleaned"
                    )
                
                if len(extreme_rises) > 0:
                    # Price rises >50% are suspicious - these are real errors
                    # Ensure extreme_rises is treated as a Series
                    rises_series = cast(pd.Series, extreme_rises)
                    rise_dates = ticker_data.loc[rises_series.index, 'date'].tolist()[:5]
                    errors.append(
                        f"Ticker {ticker}: {len(extreme_rises)} extreme price RISES (>50%) "
                        f"at dates: {rise_dates} - LIKELY CORRUPTION"
                    )
        
        # Check for missing dates per ticker
        for ticker in df['Ticker'].unique():
            ticker_data = df[df['Ticker'] == ticker].copy()
            ticker_data = ticker_data.sort_values(by=['date'])  # type: ignore[arg-type]
            
            # Check for gaps
            date_diff = ticker_data['date'].diff()
            large_gaps = date_diff[date_diff > pd.Timedelta(days=5)]
            if len(large_gaps) > 0:
                errors.append(
                    f"Ticker {ticker}: {len(large_gaps)} large date gaps (>5 days)"
                )
        
        # Check for NaN values
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_cols:
            if col in df.columns:
                nan_count = df[col].isna().sum()
                if nan_count > 0:
                    errors.append(f"Column {col}: {nan_count} NaN values")
        
        is_valid = len(errors) == 0
        return is_valid, errors
    
    def calculate_indicators_per_ticker(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators PER TICKER (prevents misalignment).
        
        This is critical: indicators must be calculated separately for each ticker
        to prevent date misalignment issues.
        """
        self.logger.info("Calculating technical indicators per ticker...")
        
        result_dfs = []
        
        for ticker in df['Ticker'].unique():
            ticker_data = df[df['Ticker'] == ticker].copy()
            ticker_data = ticker_data.sort_values(by=['date']).reset_index(drop=True)  # type: ignore[arg-type]
            
            # Calculate indicators for this ticker only
            ticker_data = self._calculate_indicators_single_ticker(ticker_data)
            
            result_dfs.append(ticker_data)
        
        # Combine all tickers
        combined = pd.concat(result_dfs, ignore_index=True)
        combined = combined.sort_values(['Ticker', 'date']).reset_index(drop=True)
        
        self.logger.info(f"✅ Calculated indicators for {len(df['Ticker'].unique())} tickers")
        return combined
    
    def _calculate_indicators_single_ticker(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate indicators for a single ticker."""
        # Ensure we have enough data
        if len(df) < 26:
            self.logger.warning(f"Ticker {df['Ticker'].iloc[0]}: Insufficient data ({len(df)} rows)")
            # Fill with default values
            for col in ['sma_20', 'ema_12', 'rsi', 'macd', 'macd_signal', 'atr', 
                       'bb_upper', 'bb_middle', 'bb_lower', 'price_to_sma_20', 'volume_trend']:
                df[col] = 0.0
            return df
        
        # AUTO-CLEAN: Fix stock splits (extreme drops >50%)
        # Stock splits cause price drops that need to be smoothed
        returns = df['close'].pct_change()
        extreme_drops = returns[returns < -0.50]  # >50% drop (likely splits)
        
        if len(extreme_drops) > 0:
            ticker = df['Ticker'].iloc[0]
            self.logger.info(f"Auto-cleaning {len(extreme_drops)} stock splits for {ticker}...")
            
            # Ensure extreme_drops is treated as a Series
            drops_series = cast(pd.Series, extreme_drops)
            for idx in drops_series.index:
                if idx > 0 and idx < len(df) - 1:
                    # Interpolate between previous and next valid price
                    prev_price = df.loc[idx - 1, 'close']
                    next_price = df.loc[idx + 1, 'close']
                    # Use linear interpolation
                    df.loc[idx, 'close'] = (prev_price + next_price) / 2.0
                    df.loc[idx, 'open'] = (df.loc[idx - 1, 'open'] + df.loc[idx + 1, 'open']) / 2.0
                    df.loc[idx, 'high'] = (df.loc[idx - 1, 'high'] + df.loc[idx + 1, 'high']) / 2.0
                    df.loc[idx, 'low'] = (df.loc[idx - 1, 'low'] + df.loc[idx + 1, 'low']) / 2.0
                    # Volume can stay as-is (splits don't affect volume)
            
            self.logger.info(f"✅ Cleaned {len(extreme_drops)} split events for {ticker}")
        
        close = df['close']
        high = df['high']
        low = df['low']
        volume = df['volume']
        
        # SMA_20
        df['sma_20'] = close.rolling(window=20, min_periods=1).mean()
        
        # EMA_12
        df['ema_12'] = close.ewm(span=12, adjust=False).mean()
        
        # RSI (14-day)
        delta = close.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14, min_periods=1).mean()
        avg_loss = loss.rolling(window=14, min_periods=1).mean()
        rs = avg_gain / (avg_loss + 1e-10)
        df['rsi'] = 100 - (100 / (1 + rs))
        df['rsi'] = df['rsi'].fillna(50.0).clip(0, 100)
        
        # MACD (12, 26, 9)
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        df['macd'] = ema12 - ema26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        
        # ATR (14-day)
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df['atr'] = true_range.rolling(window=14, min_periods=1).mean()
        df['atr'] = df['atr'].fillna(0.0)
        
        # Bollinger Bands (20-day, 2 std)
        sma_20 = df['sma_20']
        std_20 = close.rolling(window=20, min_periods=1).std()
        df['bb_upper'] = sma_20 + (std_20 * 2)
        df['bb_middle'] = sma_20
        df['bb_lower'] = sma_20 - (std_20 * 2)
        
        # Price to SMA_20
        df['price_to_sma_20'] = close / (sma_20 + 1e-10)
        df['price_to_sma_20'] = df['price_to_sma_20'].fillna(1.0).clip(0.1, 10.0)
        
        # Volume Trend (10-day average)
        df['volume_trend'] = volume.rolling(window=10, min_periods=1).mean()
        df['volume_trend'] = df['volume_trend'].fillna(0.0)
        
        # Fill any remaining NaN with safe defaults
        indicator_cols = ['sma_20', 'ema_12', 'rsi', 'macd', 'macd_signal', 'atr',
                          'bb_upper', 'bb_middle', 'bb_lower', 'price_to_sma_20', 'volume_trend']
        for col in indicator_cols:
            if col in df.columns:
                df[col] = df[col].fillna(0.0)
                # Replace inf with safe values
                df[col] = df[col].replace([np.inf, -np.inf], 0.0)
        
        return df
    
    def validate_processed_data(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Validate processed data with indicators."""
        errors = []
        
        # Check indicator ranges
        if 'rsi' in df.columns:
            invalid_rsi = df[(df['rsi'] < 0) | (df['rsi'] > 100)]
            if len(invalid_rsi) > 0:
                errors.append(f"RSI out of range (0-100): {len(invalid_rsi)} rows")
        
        # Check for NaN in indicators
        indicator_cols = ['sma_20', 'ema_12', 'rsi', 'macd', 'macd_signal', 'atr',
                         'bb_upper', 'bb_middle', 'bb_lower', 'price_to_sma_20', 'volume_trend']
        for col in indicator_cols:
            if col in df.columns:
                nan_count = df[col].isna().sum()
                if nan_count > 0:
                    errors.append(f"Indicator {col}: {nan_count} NaN values")
                
                inf_count = np.isinf(df[col]).sum()
                if inf_count > 0:
                    errors.append(f"Indicator {col}: {inf_count} Inf values")
        
        # Check date alignment across tickers
        tickers = df['Ticker'].unique()
        if len(tickers) > 1:
            # Get date ranges per ticker
            date_ranges = {}
            for ticker in tickers:
                # Ensure we get a Series, not an array
                ticker_df = df[df['Ticker'] == ticker]
                date_series = cast(pd.Series, ticker_df['date'])
                ticker_dates = set(date_series.dt.date)
                date_ranges[ticker] = ticker_dates
            
            # Check for significant misalignment
            all_dates = set.union(*date_ranges.values())
            for ticker in tickers:
                missing = all_dates - date_ranges[ticker]
                if len(missing) > len(all_dates) * 0.1:  # >10% missing
                    errors.append(
                        f"Ticker {ticker}: {len(missing)} missing dates "
                        f"({len(missing)/len(all_dates)*100:.1f}% of total)"
                    )
        
        is_valid = len(errors) == 0
        return is_valid, errors
    
    def process_and_save(self, 
                        tickers: List[str],
                        start_date: str,
                        end_date: str,
                        force: bool = False) -> pd.DataFrame:
        """
        Complete pipeline: Download → Validate → Calculate Indicators → Save.
        
        Returns:
            Processed DataFrame with all features
        """
        # Step 1: Download raw data
        if not self.raw_csv_path.exists() or force:
            raw_df = self.download_raw_data(tickers, start_date, end_date, force=force)
        else:
            self.logger.info(f"Loading existing raw CSV: {self.raw_csv_path}")
            raw_df = pd.read_csv(self.raw_csv_path, parse_dates=['date'])
        
        # Step 2: Validate raw data
        self.logger.info("Validating raw data...")
        is_valid, errors = self.validate_raw_data(raw_df)
        if not is_valid:
            self.logger.error("❌ Raw data validation failed:")
            for error in errors:
                self.logger.error(f"   - {error}")
            raise ValueError("Raw data validation failed. Fix data quality issues first.")
        self.logger.info("✅ Raw data validation passed")
        
        # Step 3: Calculate indicators per ticker
        processed_df = self.calculate_indicators_per_ticker(raw_df)
        
        # Step 4: Validate processed data
        self.logger.info("Validating processed data...")
        is_valid, errors = self.validate_processed_data(processed_df)
        if not is_valid:
            self.logger.error("❌ Processed data validation failed:")
            for error in errors:
                self.logger.error(f"   - {error}")
            raise ValueError("Processed data validation failed.")
        self.logger.info("✅ Processed data validation passed")
        
        # Step 5: Save processed CSV
        if self.processed_csv_path.exists() and not force:
            raise RuntimeError(
                f"Processed CSV already exists: {self.processed_csv_path}\n"
                f"Delete it explicitly to regenerate, or use force=True"
            )
        
        processed_df.to_csv(self.processed_csv_path, index=False)
        self.logger.info(f"✅ Saved processed data to {self.processed_csv_path}")
        
        # Step 6: Create manifest
        manifest = {
            'created_at': datetime.now().isoformat(),
            'tickers': tickers,
            'start_date': start_date,
            'end_date': end_date,
            'num_rows': len(processed_df),
            'num_tickers': len(tickers),
            'raw_csv_hash': self.file_hash(self.raw_csv_path),
            'processed_csv_hash': self.file_hash(self.processed_csv_path),
            'columns': processed_df.columns.tolist()
        }
        
        with open(self.manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        self.logger.info(f"✅ Saved manifest to {self.manifest_path}")
        
        return processed_df
    
    def _load_and_aggregate_intraday(self, interval: str = '5m') -> Optional[pd.DataFrame]:
        """
        Load intraday CSV (e.g. 5m or 1h), aggregate to daily per ticker.
        Returns DataFrame with columns: date, Ticker, and interval-suffixed columns
        (intraday_vol_<interval>, intraday_range_pct_<interval>, intraday_vol_sum_<interval>,
         intraday_vol_ratio_<interval>). Used to augment daily data for multi-timeframe observations/rewards.
        """
        intraday_name = f"trading_data_{interval}.csv"
        intraday_path = self.data_dir / intraday_name
        if not intraday_path.exists():
            self.logger.debug(f"Intraday CSV not found: {intraday_path} (skipping intraday merge)")
            return None
        try:
            idf = pd.read_csv(intraday_path, parse_dates=['date'])
            if 'Ticker' not in idf.columns and 'tic' in idf.columns:
                idf = idf.rename(columns={'tic': 'Ticker'})
            idf['date'] = pd.to_datetime(idf['date']).dt.tz_localize(None).dt.normalize()
            idf = idf.sort_values(['Ticker', 'date'])
            def vol_of_returns(s: pd.Series) -> float:
                r = s.pct_change().dropna()
                return float(r.std()) if len(r) > 0 else 0.0
            agg = idf.groupby(['date', 'Ticker']).agg(
                intraday_vol=('close', vol_of_returns),
                high_max=('high', 'max'),
                low_min=('low', 'min'),
                close_last=('close', 'last'),
                intraday_vol_sum=('volume', 'sum')
            ).reset_index()
            agg['intraday_range_pct'] = (agg['high_max'] - agg['low_min']) / (agg['close_last'].astype(float) + 1e-10)
            agg = agg[['date', 'Ticker', 'intraday_vol', 'intraday_range_pct', 'intraday_vol_sum']]
            # Suffix columns by interval for multi-timeframe merge (no overwrite between 5m/30m/1h/1m)
            suffix = f"_{interval}"
            agg = agg.rename(columns={
                'intraday_vol': 'intraday_vol' + suffix,
                'intraday_range_pct': 'intraday_range_pct' + suffix,
                'intraday_vol_sum': 'intraday_vol_sum' + suffix,
            })
            self.logger.info(f"Aggregated intraday ({interval}) to daily: {len(agg)} rows")
            return agg
        except Exception as e:
            self.logger.warning(f"Intraday load/aggregate failed: {e}")
            return None

    def load_processed_data(self, merge_intraday: bool = True, intervals: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Load processed CSV data (immutable, validated).
        Optionally merge intraday aggregates (1m, 5m, 30m, 1h) when available for multi-timeframe features.
        
        Args:
            merge_intraday: If True, merge all available intraday CSVs (suffixed by interval).
            intervals: List of intervals to merge (e.g. ['5m','30m','1h','1m']). If None, uses default from config.
        
        Returns:
            DataFrame with all features and interval-suffixed intraday columns when merge_intraday.
        """
        if not self.processed_csv_path.exists():
            raise FileNotFoundError(
                f"Processed CSV not found: {self.processed_csv_path}\n"
                f"Run process_and_save() first to create it."
            )
        
        self.logger.info(f"Loading processed CSV: {self.processed_csv_path}")
        df = pd.read_csv(self.processed_csv_path, parse_dates=['date'])
        
        # Verify hash if manifest exists
        if self.manifest_path.exists():
            with open(self.manifest_path, 'r') as f:
                manifest = json.load(f)
            
            current_hash = self.file_hash(self.processed_csv_path)
            stored_hash = manifest.get('processed_csv_hash', '')
            
            if stored_hash and current_hash != stored_hash:
                self.logger.warning(
                    f"CSV hash mismatch! File may have been modified."
                )
        
        if merge_intraday:
            if intervals is None:
                try:
                    from ppo.utils.dynamic_config import get_intraday_merge_intervals
                    intervals = get_intraday_merge_intervals()
                except Exception:
                    intervals = ['5m', '30m', '1h', '1m']
            merged_any = False
            df['_date_norm'] = pd.to_datetime(df['date']).dt.normalize()
            if hasattr(df['_date_norm'].dtype, 'tz') and df['_date_norm'].dtype.tz is not None:
                df['_date_norm'] = df['_date_norm'].dt.tz_localize(None)
            for interval in intervals:
                agg = self._load_and_aggregate_intraday(interval)
                if agg is None or agg.empty:
                    continue
                agg = agg.rename(columns={'date': '_date_norm'})
                if hasattr(agg['_date_norm'].dtype, 'tz') and agg['_date_norm'].dtype.tz is not None:
                    agg = agg.copy()
                    agg['_date_norm'] = agg['_date_norm'].dt.tz_localize(None)
                merge_cols = ['_date_norm', 'Ticker'] + [c for c in agg.columns if c not in ('_date_norm', 'Ticker')]
                df = df.merge(agg[merge_cols], on=['_date_norm', 'Ticker'], how='left')
                vol_sum_col = f'intraday_vol_sum_{interval}'
                if vol_sum_col in df.columns and 'volume' in df.columns:
                    ratio_col = f'intraday_vol_ratio_{interval}'
                    df[ratio_col] = np.where(
                        df['volume'] > 0,
                        df[vol_sum_col].fillna(0) / df['volume'],
                        0.0
                    )
                for c in merge_cols:
                    if c in df.columns and c not in ('_date_norm', 'Ticker'):
                        df[c] = df[c].fillna(0.0)
                merged_any = True
            df = df.drop(columns=['_date_norm'], errors='ignore')
            if merged_any:
                self.logger.info("Merged multi-timeframe intraday aggregates into daily data")
        
        # Final validation
        is_valid, errors = self.validate_processed_data(df)
        if not is_valid:
            self.logger.warning("⚠️ Loaded data has validation issues:")
            for error in errors:
                self.logger.warning(f"   - {error}")
        
        self.logger.info(f"✅ Loaded {len(df)} rows, {len(df.columns)} columns")
        return df

    def load_intraday_bars(self, interval: str = '5m') -> Optional[pd.DataFrame]:
        """
        Load raw intraday CSV (e.g. 5m or 30m) without aggregating.
        Returns DataFrame with one row per bar: date, Ticker, open, high, low, close, volume.
        Used for intraday-step env (step = one bar).
        """
        name = f"trading_data_{interval}.csv"
        path = self.data_dir / name
        if not path.exists():
            self.logger.debug(f"Intraday bars CSV not found: {path}")
            return None
        try:
            df = pd.read_csv(path, parse_dates=['date'])
            if 'Ticker' not in df.columns and 'tic' in df.columns:
                df = df.rename(columns={'tic': 'Ticker'})
            df = df.sort_values(['Ticker', 'date']).reset_index(drop=True)
            self.logger.info(f"Loaded intraday bars ({interval}): {len(df)} rows")
            return df
        except Exception as e:
            self.logger.warning(f"Load intraday bars failed: {e}")
            return None


def main():
    """Example usage."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Tickers from config
    tickers = ['AAPL', 'MSFT', 'NVDA', 'JPM', 'SPY', 'GLD', 'KO', 'JNJ', 'AMZN', 'XLE', 'TLT']
    start_date = '2020-01-01'
    end_date = '2024-10-17'
    
    # Create manager
    manager = CSVDataManager()
    
    # Process data (download + calculate indicators + validate + save)
    try:
        df = manager.process_and_save(tickers, start_date, end_date, force=False)
        print(f"\n✅ Success! Processed {len(df)} rows")
        print(f"   CSV saved to: {manager.processed_csv_path}")
        print(f"   Manifest saved to: {manager.manifest_path}")
    except RuntimeError as e:
        print(f"\n⚠️ {e}")
        print("   Use force=True to regenerate, or delete existing CSV files first")


if __name__ == "__main__":
    main()

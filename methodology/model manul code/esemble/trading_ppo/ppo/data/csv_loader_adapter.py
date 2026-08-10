"""
CSV Loader Adapter - Bridge between CSV data and trading environment

This adapter loads CSV data and converts it to the matrix format
required by the trading environment, ensuring proper alignment.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, cast
from .csv_data_manager import CSVDataManager

logger = logging.getLogger(__name__)


class CSVLoaderAdapter:
    """
    Adapter to load CSV data and convert to environment matrices.
    
    Ensures:
    - All tickers aligned to same date range
    - No shape mismatches
    - Proper feature alignment
    - Deterministic matrix building
    """
    
    def __init__(self, csv_manager: Optional[CSVDataManager] = None):
        """Initialize adapter with CSV manager."""
        self.csv_manager = csv_manager or CSVDataManager()
        self.logger = logging.getLogger(__name__)
    
    def load_and_convert_to_matrices(self, 
                                    tickers: List[str],
                                    feature_cols: Optional[List[str]] = None,
                                    train_split: float = 0.75) -> Dict:
        """
        Load CSV and convert to matrices for trading environment.
        
        Args:
            tickers: List of ticker symbols (must match CSV)
            feature_cols: List of feature column names; if None, uses canonical list (base + intraday).
            train_split: Train/test split ratio (default 0.75)
            
        Returns:
            Dictionary with:
            - train_data: {price_matrix, feature_matrix, volume_matrix, dates}
            - test_data: {price_matrix, feature_matrix, volume_matrix, dates}
            - tickers: List of tickers
            - feature_cols: List of feature columns
        """
        if feature_cols is None:
            try:
                from ppo.utils.dynamic_config import get_canonical_feature_cols
                feature_cols = get_canonical_feature_cols(include_intraday=True)
            except Exception:
                feature_cols = ['sma_20', 'ema_12', 'rsi', 'macd', 'macd_signal', 'atr',
                               'bb_upper', 'bb_middle', 'bb_lower', 'price_to_sma_20', 'volume_trend']
        # Load processed CSV (merge_intraday=True, intervals from config)
        df = self.csv_manager.load_processed_data(merge_intraday=True)
        
        # Validate tickers match
        csv_tickers = sorted(df['Ticker'].unique())
        if set(tickers) != set(csv_tickers):
            self.logger.warning(
                f"Ticker mismatch: CSV has {csv_tickers}, requested {tickers}\n"
                f"Using tickers from CSV: {csv_tickers}"
            )
            tickers = csv_tickers
        
        # Ensure date column is datetime
        if not pd.api.types.is_datetime64_any_dtype(df['date']):
            df['date'] = pd.to_datetime(df['date'])
        
        # CRITICAL: Align all tickers to same date range
        # Get intersection of dates across all tickers
        all_dates = set()
        for ticker in tickers:
            # Ensure we get a Series, not an array
            ticker_df = df[df['Ticker'] == ticker]
            date_series = cast(pd.Series, ticker_df['date'])
            ticker_dates = set(date_series.dt.date)
            if not all_dates:
                all_dates = ticker_dates
            else:
                all_dates = all_dates.intersection(ticker_dates)
        
        # Convert back to sorted list
        all_dates = sorted(list(all_dates))
        self.logger.info(f"Aligned dates: {len(all_dates)} common dates across all tickers")
        
        # Filter to common dates only
        df_aligned = df[df['date'].dt.date.isin(all_dates)].copy()
        # Sort by Ticker and date - use type ignore for pandas type stub limitation
        df_aligned = df_aligned.sort_values(['Ticker', 'date']).reset_index(drop=True)  # type: ignore[arg-type]
        
        # Split train/test
        split_idx = int(len(all_dates) * train_split)
        train_dates = set(all_dates[:split_idx])
        test_dates = set(all_dates[split_idx:])
        
        df_train = df_aligned[df_aligned['date'].dt.date.isin(train_dates)].copy()
        df_test = df_aligned[df_aligned['date'].dt.date.isin(test_dates)].copy()
        
        self.logger.info(f"Train dates: {len(train_dates)}, Test dates: {len(test_dates)}")
        
        # Convert to matrices - ensure we have DataFrames, not Series
        train_df = cast(pd.DataFrame, df_train)
        test_df = cast(pd.DataFrame, df_test)
        train_data = self._dataframe_to_matrices(train_df, tickers, feature_cols)
        test_data = self._dataframe_to_matrices(test_df, tickers, feature_cols)
        
        return {
            'train_data': train_data,
            'test_data': test_data,
            'tickers': tickers,
            'feature_cols': feature_cols
        }
    
    def _dataframe_to_matrices(self, 
                               df: pd.DataFrame,
                               tickers: List[str],
                               feature_cols: List[str]) -> Dict:
        """
        Convert DataFrame to matrices (price, feature, volume).
        
        CRITICAL: This ensures all tickers are aligned to same dates.
        """
        # Get unique dates (already aligned)
        dates = sorted(df['date'].unique())
        num_dates = len(dates)
        num_tickers = len(tickers)
        num_features = len(feature_cols)
        
        # Initialize matrices
        price_matrix = np.zeros((num_dates, num_tickers))
        feature_matrix = np.zeros((num_dates, num_tickers, num_features))
        volume_matrix = np.zeros((num_dates, num_tickers))
        
        # Create mappings
        date_to_idx = {pd.Timestamp(date): idx for idx, date in enumerate(dates)}
        ticker_to_idx = {ticker: idx for idx, ticker in enumerate(tickers)}
        
        # Fill matrices
        for _, row in df.iterrows():
            date = row['date']
            ticker = row['Ticker']
            
            # Ensure date is a Timestamp and ticker is a string
            # row['date'] from iterrows() returns a scalar, but type checker may not recognize it
            if isinstance(date, pd.Timestamp):
                date_ts = date
            elif isinstance(date, pd.Series):
                date_ts = pd.Timestamp(date.iloc[0])
            elif isinstance(date, np.ndarray):
                date_ts = pd.Timestamp(date.item() if date.size == 1 else date[0])
            else:
                # Convert scalar to Timestamp
                date_ts = pd.Timestamp(cast(Any, date))
            
            ticker_str = str(ticker) if not isinstance(ticker, str) else ticker
            
            date_idx = date_to_idx.get(date_ts)
            ticker_idx = ticker_to_idx.get(ticker_str)
            
            if date_idx is None or ticker_idx is None:
                continue
            
            # Price
            price_matrix[date_idx, ticker_idx] = row['close']
            
            # Volume
            volume_matrix[date_idx, ticker_idx] = row.get('volume', 0.0)
            
            # Features (missing cols filled with 0 for optional intraday columns)
            for feat_idx, feat_col in enumerate(feature_cols):
                feature_matrix[date_idx, ticker_idx, feat_idx] = row.get(feat_col, 0.0)
        
        # NaN clean: forward-fill per column so validation does not fail
        if np.isnan(price_matrix).any():
            n_before = int(np.isnan(price_matrix).sum())
            for col in range(price_matrix.shape[1]):
                col_vals = price_matrix[:, col].copy()
                if np.isnan(col_vals).any():
                    last = np.nan
                    for i in range(len(col_vals)):
                        if np.isnan(col_vals[i]) and not np.isnan(last):
                            col_vals[i] = last
                        elif not np.isnan(col_vals[i]):
                            last = col_vals[i]
                    if np.isnan(col_vals).any():
                        idx = np.where(~np.isnan(col_vals))[0]
                        if len(idx):
                            col_vals[:idx[0]] = col_vals[idx[0]]
                        col_vals[np.isnan(col_vals)] = 0.01
                    price_matrix[:, col] = col_vals
            self.logger.info(f"Price matrix had {n_before} NaN values; forward-filled and replaced remaining with 0.01")
        
        # Validate matrices
        self._validate_matrices(price_matrix, feature_matrix, volume_matrix, dates, tickers)
        
        return {
            'price_matrix': price_matrix,
            'feature_matrix': feature_matrix,
            'volume_matrix': volume_matrix,
            'dates': dates
        }
    
    def _validate_matrices(self, 
                          price_matrix: np.ndarray,
                          feature_matrix: np.ndarray,
                          volume_matrix: np.ndarray,
                          dates: List,
                          tickers: List[str]):
        """Validate matrix shapes and data quality."""
        # Shape validation
        assert price_matrix.shape[0] == len(dates), "Price matrix date mismatch"
        assert price_matrix.shape[1] == len(tickers), "Price matrix ticker mismatch"
        assert feature_matrix.shape[0] == len(dates), "Feature matrix date mismatch"
        assert feature_matrix.shape[1] == len(tickers), "Feature matrix ticker mismatch"
        assert volume_matrix.shape[0] == len(dates), "Volume matrix date mismatch"
        assert volume_matrix.shape[1] == len(tickers), "Volume matrix ticker mismatch"
        
        # Data quality
        assert (price_matrix > 0).all(), "Found zero or negative prices"
        assert not np.isnan(price_matrix).any(), "Found NaN in price matrix"
        assert not np.isinf(price_matrix).any(), "Found Inf in price matrix"
        assert not np.isnan(feature_matrix).any(), "Found NaN in feature matrix"
        assert not np.isinf(feature_matrix).any(), "Found Inf in feature matrix"
        
        self.logger.info("Matrix validation passed")
        self.logger.info(f"   Price matrix: {price_matrix.shape}")
        self.logger.info(f"   Feature matrix: {feature_matrix.shape}")
        self.logger.info(f"   Volume matrix: {volume_matrix.shape}")

    def load_intraday_bar_matrices(self, interval: str = '5m', tickers: Optional[List[str]] = None) -> Optional[Dict]:
        """
        Load intraday bar data and build matrices for bar-level stepping (one step = one bar).
        Aligns to common bar timestamps across all tickers. Returns dict with price_matrix,
        feature_matrix, volume_matrix, dates; or None if CSV missing.
        Feature matrix: (num_bars, num_tickers, num_features) with close + intraday technical features.
        """
        df = self.csv_manager.load_intraday_bars(interval)
        if df is None or df.empty:
            return None
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])

        # Determine tradable tickers (requested tickers only)
        if tickers is None:
            tickers = sorted(df['Ticker'].unique().tolist())
        tickers = [str(t).upper() for t in tickers]
        df['Ticker'] = df['Ticker'].astype(str).str.upper()

        # Use ONLY requested tickers for matrix alignment (avoid universe bleed)
        df_trade = df[df['Ticker'].isin(tickers)].copy()
        if df_trade.empty:
            self.logger.warning(f"Intraday bars ({interval}): no rows found for requested tickers={tickers}")
            return None

        # Common bar timestamps (intersection across tradable tickers)
        # Crypto (BTC-USD, ETH-USD, etc.) trades 24/7 - different timestamps than stocks (market hours).
        # If intersection is empty/small, retry with stocks-only for alignment.
        _24_7_TICKERS = {'BTC-USD', 'ETH-USD', 'BTC', 'ETH'}  # Crypto trades 24/7
        stock_tickers = [t for t in tickers if t not in _24_7_TICKERS]
        all_dates: set = set()
        tickers_for_alignment = tickers
        if stock_tickers:
            for t in stock_tickers:
                ticker_dates = set(df_trade.loc[df_trade['Ticker'] == t, 'date'].dropna().unique())
                if not all_dates:
                    all_dates = ticker_dates.copy()
                else:
                    all_dates = all_dates.intersection(ticker_dates)
            if len(all_dates) >= 50:
                tickers_for_alignment = stock_tickers
                self.logger.info(f"Intraday bars: using stocks-only alignment ({len(stock_tickers)} tickers) - crypto excluded (different schedule)")
        if len(all_dates) < 50:
            all_dates = set()
            for t in tickers:
                ticker_dates = set(df_trade.loc[df_trade['Ticker'] == t, 'date'].dropna().unique())
                if not all_dates:
                    all_dates = ticker_dates.copy()
                else:
                    all_dates = all_dates.intersection(ticker_dates)
        all_dates = sorted(list(all_dates))
        if len(all_dates) < 50:
            self.logger.warning(f"Intraday bars: only {len(all_dates)} common bars across tickers={tickers_for_alignment}")
            return None
        tickers = tickers_for_alignment

        # Optional market factor (SPY) – does NOT affect date intersection. Missing values become 0.
        spy_close = None
        spy_df = df[df['Ticker'] == 'SPY'].copy()
        if not spy_df.empty and 'close' in spy_df.columns:
            spy_df = spy_df.sort_values('date')
            spy_close = spy_df.set_index('date')['close']

        def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
            delta = series.diff()
            gain = delta.clip(lower=0.0)
            loss = (-delta).clip(lower=0.0)
            avg_gain = gain.rolling(period, min_periods=period).mean()
            avg_loss = loss.rolling(period, min_periods=period).mean()
            rs = avg_gain / (avg_loss + 1e-9)
            rsi = 100.0 - (100.0 / (1.0 + rs))
            return rsi.fillna(50.0)

        def _ema(series: pd.Series, span: int) -> pd.Series:
            return series.ewm(span=span, adjust=False).mean()

        # Feature schema (reduced 2026-02: drop redundant/noisy; README "Feature set & noise")
        # Dropped: log_ret_1 (redundant with ret_1), vol_z_20 (noisy on thin bars)
        feature_cols = [
            'close',
            'ret_1',
            'rsi_14',
            'macd',
            'macd_signal',
            'macd_hist',
            'bb_z',
            'bb_width',
            'atr_14',
            'vol_ratio_20',
            'tod_sin',
            'tod_cos',
            'spy_ret_1',
            'spy_rsi_14',
        ]

        num_bars = len(all_dates)
        num_tickers = len(tickers)
        price_matrix = np.zeros((num_bars, num_tickers), dtype=np.float32)
        volume_matrix = np.zeros((num_bars, num_tickers), dtype=np.float32)
        nf = len(feature_cols)
        feature_matrix = np.zeros((num_bars, num_tickers, nf), dtype=np.float32)

        # Build SPY factor arrays on the bar timeline (forward-filled; missing -> 0)
        spy_ret_1 = np.zeros(num_bars, dtype=np.float32)
        spy_rsi_14 = np.zeros(num_bars, dtype=np.float32)
        if spy_close is not None:
            spy_aligned = spy_close.reindex(all_dates).ffill()
            spy_ret = spy_aligned.pct_change().fillna(0.0).astype(float)
            spy_rsi = _rsi(spy_aligned.astype(float), period=14).astype(float)
            spy_ret_1 = spy_ret.to_numpy(dtype=np.float32, na_value=0.0)
            spy_rsi_14 = (spy_rsi / 100.0).to_numpy(dtype=np.float32, na_value=0.5)  # normalize to 0..1

        # Time-of-day features
        # (Using minute-of-day derived from bar timestamp; stable across assets.)
        tod_sin = np.zeros(num_bars, dtype=np.float32)
        tod_cos = np.zeros(num_bars, dtype=np.float32)
        try:
            minutes = np.array([d.hour * 60 + d.minute for d in all_dates], dtype=np.float32)
            phase = (2.0 * np.pi * minutes) / (24.0 * 60.0)
            tod_sin = np.sin(phase).astype(np.float32)
            tod_cos = np.cos(phase).astype(np.float32)
        except Exception:
            pass

        # Per-ticker feature computation (vectorized per series; no row-iter loops)
        for ti, tkr in enumerate(tickers):
            tdf = df_trade[df_trade['Ticker'] == tkr].sort_values('date').set_index('date')
            tdf = tdf.reindex(all_dates)

            close = pd.to_numeric(tdf.get('close', 0.0), errors='coerce').fillna(method='ffill').fillna(0.0).astype(float)
            high = pd.to_numeric(tdf.get('high', close), errors='coerce').fillna(method='ffill').fillna(close).astype(float)
            low = pd.to_numeric(tdf.get('low', close), errors='coerce').fillna(method='ffill').fillna(close).astype(float)
            vol = pd.to_numeric(tdf.get('volume', 0.0), errors='coerce').fillna(0.0).astype(float)

            # Price + volume matrices
            close_arr = close.to_numpy(dtype=np.float32, na_value=0.0)
            vol_arr = vol.to_numpy(dtype=np.float32, na_value=0.0)
            price_matrix[:, ti] = close_arr
            volume_matrix[:, ti] = vol_arr

            # Returns (ret_1 only; log_ret_1 dropped as redundant)
            ret_1 = close.pct_change().fillna(0.0)

            # RSI (normalized to 0..1)
            rsi_14 = (_rsi(close, 14) / 100.0).clip(0.0, 1.0)

            # MACD
            ema12 = _ema(close, 12)
            ema26 = _ema(close, 26)
            macd = ema12 - ema26
            macd_signal = _ema(macd, 9)
            macd_hist = macd - macd_signal

            # Bollinger (z-score + width)
            bb_mid = close.rolling(20, min_periods=20).mean()
            bb_std = close.rolling(20, min_periods=20).std()
            bb_z = ((close - bb_mid) / (bb_std + 1e-9)).fillna(0.0)
            bb_upper = bb_mid + 2.0 * bb_std
            bb_lower = bb_mid - 2.0 * bb_std
            bb_width = ((bb_upper - bb_lower) / (bb_mid.abs() + 1e-9)).fillna(0.0)

            # ATR (true range)
            prev_close = close.shift(1)
            tr = pd.concat([
                (high - low).abs(),
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ], axis=1).max(axis=1).fillna(0.0)
            atr_14 = tr.rolling(14, min_periods=14).mean().fillna(0.0)
            # Normalize ATR by price to keep bounded-ish
            atr_14 = (atr_14 / (close.abs() + 1e-9)).clip(0.0, 1.0)

            # Volume ratio (vol_z_20 dropped - noisy on thin bars)
            vol_mean20 = vol.rolling(20, min_periods=20).mean()
            vol_ratio_20 = (vol / (vol_mean20 + 1e-9)).fillna(1.0).clip(0.0, 10.0)

            # Assemble features (match feature_cols order)
            feats = np.column_stack([
                close.to_numpy(dtype=np.float32, na_value=0.0),
                ret_1.to_numpy(dtype=np.float32, na_value=0.0),
                rsi_14.to_numpy(dtype=np.float32, na_value=0.5),
                macd.fillna(0.0).to_numpy(dtype=np.float32, na_value=0.0),
                macd_signal.fillna(0.0).to_numpy(dtype=np.float32, na_value=0.0),
                macd_hist.fillna(0.0).to_numpy(dtype=np.float32, na_value=0.0),
                bb_z.to_numpy(dtype=np.float32, na_value=0.0),
                bb_width.to_numpy(dtype=np.float32, na_value=0.0),
                atr_14.to_numpy(dtype=np.float32, na_value=0.0),
                vol_ratio_20.to_numpy(dtype=np.float32, na_value=1.0),
                tod_sin,
                tod_cos,
                spy_ret_1,
                spy_rsi_14,
            ])
            feature_matrix[:, ti, :] = feats

        self.logger.info(f"Intraday bar matrices ({interval}): bars={num_bars}, tickers={num_tickers}, features={nf}")
        return {
            'price_matrix': price_matrix,
            'feature_matrix': feature_matrix,
            'volume_matrix': volume_matrix,
            'dates': all_dates,
            'tickers': tickers,
            'feature_cols': feature_cols,
        }

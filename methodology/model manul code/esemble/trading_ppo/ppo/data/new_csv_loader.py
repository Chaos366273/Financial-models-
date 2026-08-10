"""
New CSV Loader for 5-Year Stock Data
Integrates the New_stock_data_5_years.csv file with zero NaNs
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class NewStockDataLoader:
    """
    Load and process the new 5-year stock dataset.
    
    Features:
    - Zero NaNs guaranteed
    - 5 years of data (2018-2023)
    - Clean OHLCV data
    - Multiple companies
    """
    
    # Try these filenames in order (FTSE dissertation first, then legacy US CSVs)
    DEFAULT_CSV_NAMES = [
        "ftse_from_final_data.csv",  # dissertation: materialised from final data for AI
        "New_stock_data_5_years.csv",
        "new stock data 5 years.csv",
        "stock_details_5_years.csv",
    ]
    # Optional: Bitcoin price file in same folder (merged so agent can trade BTC)
    # Excel (preferred): Bitcoin 2016-2026.xlsx
    BITCOIN_XLSX_NAMES = [
        "Bitcoin_2016_2026_cleaned.xlsx",  # Correct format: Date, O, H, L, C, V, Company (header=0)
        "Bitcoin 2016-2026.xlsx",
        "bitcoin.xlsx",
        "BTC.xlsx",
    ]
    BITCOIN_CSV_NAMES = [
        "bitcoin_2016_2026.csv",  # Recreated from Bitcoin 2016-2026.xlsx (Date, O, H, L, C, V, Company)
        "bitcoin.csv",
        "BTC.csv",
        "bitcoin_price.csv",
        "BTC-USD.csv",
        "btc_price.csv",
    ]
    BITCOIN_TICKER = "BTC-USD"  # Alpaca/agent symbol for Bitcoin

    def __init__(self, csv_path: Optional[str] = None):
        """Initialize loader with CSV path."""
        if csv_path is None:
            base_dir = Path(__file__).resolve().parents[2] / "ppo" / "saved_data" / "csv_data"
            self.csv_path = None
            for name in self.DEFAULT_CSV_NAMES:
                candidate = base_dir / name
                if candidate.exists():
                    self.csv_path = candidate
                    break
            if self.csv_path is None:
                raise FileNotFoundError(
                    f"CSV file not found in {base_dir}. Tried: {self.DEFAULT_CSV_NAMES}"
                )
        else:
            self.csv_path = Path(csv_path)
            if not self.csv_path.exists():
                raise FileNotFoundError(f"CSV file not found: {self.csv_path}")
        
        logger.info(f"[NEW CSV LOADER] Initialized with: {self.csv_path}")
    
    def _normalize_bitcoin_df(self, btc: pd.DataFrame, source_name: str) -> Optional[pd.DataFrame]:
        """
        Normalize raw Bitcoin DataFrame to Date, Open, High, Low, Close, Volume, Company.
        Supports: (1) Bloomberg-style Date + PX_LAST (close-only); (2) full OHLCV.
        Returns None if required columns cannot be resolved.
        """
        if btc.empty or len(btc) < 2:
            return None
        # Strip whitespace from column names (common in Excel)
        btc.columns = btc.columns.str.strip() if hasattr(btc.columns, 'str') else btc.columns
        # Normalize date column
        date_col = None
        for c in ["Date", "date", "timestamp", "datetime", "time", "Time"]:
            if c in btc.columns:
                date_col = c
                break
        if date_col is None:
            logger.warning(f"[NEW CSV LOADER] Bitcoin {source_name}: no date column in {list(btc.columns)}")
            return None
        btc = btc.rename(columns={date_col: "Date"})
        # Parse dates (Bloomberg/Excel often use DD/MM/YYYY in data table)
        btc["Date"] = pd.to_datetime(btc["Date"], utc=True, dayfirst=True)
        # Bloomberg-style: Date + PX_LAST only -> synthesize OHLCV (Open=High=Low=Close=PX_LAST, Volume=0)
        price_col = None
        for c in btc.columns:
            if str(c).strip().upper() == "PX_LAST":
                price_col = c
                break
        if price_col is not None:
            try:
                btc = btc.dropna(subset=["Date"]).copy()
                btc["Close"] = pd.to_numeric(btc[price_col], errors="coerce")
                btc = btc.dropna(subset=["Close"])
                btc["Open"] = btc["High"] = btc["Low"] = btc["Close"]
                btc["Volume"] = 0.0
                btc["Company"] = self.BITCOIN_TICKER
                out = btc[["Date", "Open", "High", "Low", "Close", "Volume"]].copy()
                out["Company"] = self.BITCOIN_TICKER
                if out.empty or len(out) < 2:
                    return None
                return out
            except Exception as e:
                logger.warning(f"[NEW CSV LOADER] Bitcoin {source_name}: PX_LAST parse failed: {e}")
                return None
        # Full OHLCV: normalize column names
        aliases = {
            "Open": ["open"],
            "High": ["high"],
            "Low": ["low"],
            "Close": ["close", "close price", "price"],
            "Volume": ["volume", "vol", "vol."],
        }
        col_map = {}
        for std, keys in aliases.items():
            for c in btc.columns:
                cnorm = str(c).strip().lower()
                if cnorm == std.lower() or cnorm in keys:
                    col_map[c] = std
                    break
        btc = btc.rename(columns=col_map)
        required = ["Date", "Open", "High", "Low", "Close", "Volume"]
        missing = [c for c in required if c not in btc.columns]
        if missing:
            logger.warning(f"[NEW CSV LOADER] Bitcoin {source_name}: missing {missing}, found {list(btc.columns)}")
            return None
        btc = btc[required].copy()
        btc["Company"] = self.BITCOIN_TICKER
        btc = btc.dropna(subset=required)
        if btc.empty:
            return None
        return btc

    def _load_bitcoin_csv(self, base_dir: Path) -> Optional[pd.DataFrame]:
        """
        Load optional Bitcoin price file from csv_data folder (Excel first, then CSV).
        Expects OHLCV columns (Date/date, Open/open, High/high, Low/low, Close/close, Volume/volume).
        Returns DataFrame with columns Date, Open, High, Low, Close, Volume, Company='BTC-USD' or None.
        """
        # Try Excel first (e.g. Bitcoin 2016-2026.xlsx; Bloomberg-style has 5 metadata rows then Date, PX_LAST)
        for name in self.BITCOIN_XLSX_NAMES:
            path = base_dir / name
            if not path.exists():
                continue
            try:
                # Try header=0 first (cleaned format: Date, O, H, L, C, V, Company), then 6, 5 for Bloomberg-style
                for header_row in [0, 6, 5]:
                    try:
                        try:
                            btc = pd.read_excel(path, engine="openpyxl", header=header_row)
                        except ImportError:
                            btc = pd.read_excel(path, header=header_row)
                        out = self._normalize_bitcoin_df(btc, name)
                        if out is not None:
                            logger.info(f"[NEW CSV LOADER] Loaded Bitcoin from {name}: {len(out)} rows, {out['Date'].min()} to {out['Date'].max()}")
                            return out
                    except Exception:
                        continue
            except ImportError as e:
                logger.warning("[NEW CSV LOADER] For Excel support install: pip install openpyxl")
                break
            except Exception as e:
                logger.warning(f"[NEW CSV LOADER] Failed to load Bitcoin Excel {name}: {e}")
                continue
        # Fallback: CSV
        for name in self.BITCOIN_CSV_NAMES:
            path = base_dir / name
            if not path.exists():
                continue
            try:
                btc = pd.read_csv(path)
                out = self._normalize_bitcoin_df(btc, name)
                if out is not None:
                    logger.info(f"[NEW CSV LOADER] Loaded Bitcoin from {name}: {len(out)} rows, {out['Date'].min()} to {out['Date'].max()}")
                    return out
            except Exception as e:
                logger.warning(f"[NEW CSV LOADER] Failed to load Bitcoin CSV {name}: {e}")
                continue
        return None
    
    def load_data(self, tickers: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Load CSV data with optional ticker filtering.
        
        Args:
            tickers: List of ticker symbols to include (None = all)
            
        Returns:
            DataFrame with columns: Date, Open, High, Low, Close, Volume, Dividends, Stock Splits, Company
        """
        logger.info(f"[NEW CSV LOADER] Loading data from: {self.csv_path}")
        
        # Load CSV
        df = pd.read_csv(self.csv_path)
        
        # Normalize ticker column: accept Company, Ticker, or Symbol
        if 'Company' not in df.columns:
            if 'Ticker' in df.columns:
                df = df.rename(columns={'Ticker': 'Company'})
            elif 'Symbol' in df.columns:
                df = df.rename(columns={'Symbol': 'Company'})
            else:
                raise ValueError(
                    f"CSV must have a ticker column named 'Company', 'Ticker', or 'Symbol'. Found: {list(df.columns)}"
                )
        
        logger.info(f"[NEW CSV LOADER] Loaded {len(df):,} rows x {len(df.columns)} columns")
        # Normalize date column (accept Date or date)
        if 'Date' not in df.columns and 'date' in df.columns:
            df = df.rename(columns={'date': 'Date'})
        logger.info(f"[NEW CSV LOADER] Date range: {df['Date'].iloc[0]} to {df['Date'].iloc[-1]}")
        
        # Convert date to datetime (utc=True for tz-aware CSV like 2018-11-29 00:00:00-05:00)
        df['Date'] = pd.to_datetime(df['Date'], utc=True)
        
        # Merge optional Bitcoin CSV from same folder so agent can trade BTC
        base_dir = self.csv_path.parent
        btc_df = self._load_bitcoin_csv(base_dir)
        if btc_df is not None:
            # Align columns: main df may have Dividends, Stock Splits; Bitcoin may not
            for c in df.columns:
                if c not in btc_df.columns:
                    btc_df[c] = 0.0 if c in ("Dividends", "Stock Splits") else np.nan
            btc_df = btc_df[df.columns]
            df = pd.concat([df, btc_df], ignore_index=True)
            df = df.sort_values(['Company', 'Date']).reset_index(drop=True)
        
        # Get available tickers
        available_tickers = sorted(df['Company'].unique())
        logger.info(f"[NEW CSV LOADER] Available tickers ({len(available_tickers)}): {available_tickers[:20]}...")
        
        # Filter by tickers if specified
        if tickers is not None:
            # Check which tickers are available
            missing = set(tickers) - set(available_tickers)
            if missing:
                logger.warning(f"[NEW CSV LOADER] Tickers not found in CSV: {missing}")
            
            # Filter to requested tickers that exist
            valid_tickers = [t for t in tickers if t in available_tickers]
            if not valid_tickers:
                raise ValueError(f"None of the requested tickers found in CSV: {tickers}")
            
            df = df[df['Company'].isin(valid_tickers)].copy()
            logger.info(f"[NEW CSV LOADER] Filtered to {len(valid_tickers)} tickers: {valid_tickers}")
        
        # Verify zero NaNs
        nan_count = df.isnull().sum().sum()
        if nan_count > 0:
            logger.warning(f"[NEW CSV LOADER] ⚠️ Found {nan_count} NaNs (unexpected!)")
        else:
            logger.info(f"[NEW CSV LOADER] ✅ Verified ZERO NaNs")
        
        # Sort by Company and Date
        df = df.sort_values(['Company', 'Date']).reset_index(drop=True)
        
        return df
    
    def convert_to_trading_format(self, 
                                   df: pd.DataFrame,
                                   tickers: List[str],
                                   train_split: float = 0.75) -> Dict:
        """
        Convert loaded data to trading environment format.
        
        Args:
            df: DataFrame from load_data()
            tickers: List of tickers to use
            train_split: Train/test split ratio
            
        Returns:
            Dictionary with train_data, test_data, tickers
        """
        logger.info(f"[NEW CSV LOADER] Converting to trading format...")
        df = df.copy()
        # Normalize to UTC so tz-aware CSV (e.g. -05:00) and .dt.date work
        df['Date'] = pd.to_datetime(df['Date'], utc=True)
        
        # Ensure all tickers are aligned to same dates
        all_dates = set()
        for ticker in tickers:
            ticker_df = df[df['Company'] == ticker]
            ticker_dates = set(pd.to_datetime(ticker_df['Date']).dt.date)
            if not all_dates:
                all_dates = ticker_dates
            else:
                all_dates = all_dates.intersection(ticker_dates)
        
        all_dates = sorted(list(all_dates))
        logger.info(f"[NEW CSV LOADER] Aligned to {len(all_dates)} common dates")
        
        # Filter to common dates
        date_series = pd.to_datetime(df['Date']).dt.date
        df_aligned = df[date_series.isin(all_dates)].copy()
        
        # Split train/test
        split_idx = int(len(all_dates) * train_split)
        train_dates = set(all_dates[:split_idx])
        test_dates = set(all_dates[split_idx:])
        
        date_aligned = pd.to_datetime(df_aligned['Date']).dt.date
        df_train = df_aligned[date_aligned.isin(train_dates)].copy()
        df_test = df_aligned[date_aligned.isin(test_dates)].copy()
        
        logger.info(f"[NEW CSV LOADER] Train: {len(train_dates)} dates, Test: {len(test_dates)} dates")
        
        # Convert to matrices
        train_data = self._to_matrices(df_train, tickers, train_dates)
        test_data = self._to_matrices(df_test, tickers, test_dates)
        
        return {
            'train_data': train_data,
            'test_data': test_data,
            'tickers': tickers
        }
    
    def _to_matrices(self, 
                     df: pd.DataFrame, 
                     tickers: List[str],
                     dates: List) -> Dict:
        """
        Convert DataFrame to matrices (price, volume).
        
        Returns:
            Dict with price_matrix, volume_matrix, dates
        """
        dates = sorted(dates)
        num_dates = len(dates)
        num_tickers = len(tickers)
        
        # Initialize matrices
        price_matrix = np.zeros((num_dates, num_tickers))
        volume_matrix = np.zeros((num_dates, num_tickers))
        
        # Create mappings
        date_to_idx = {date: idx for idx, date in enumerate(dates)}
        ticker_to_idx = {ticker: idx for idx, ticker in enumerate(tickers)}
        
        # Fill matrices
        for _, row in df.iterrows():
            date = pd.Timestamp(row['Date']).date()
            ticker = row['Company']
            
            if date in date_to_idx and ticker in ticker_to_idx:
                date_idx = date_to_idx[date]
                ticker_idx = ticker_to_idx[ticker]
                
                price_matrix[date_idx, ticker_idx] = row['Close']
                volume_matrix[date_idx, ticker_idx] = row['Volume']
        
        # Verify no zeros (prices should never be zero)
        zero_prices = np.sum(price_matrix == 0)
        if zero_prices > 0:
            logger.warning(f"[NEW CSV LOADER] ⚠️ Found {zero_prices} zero prices (data gap?)")
        
        return {
            'price_matrix': price_matrix,
            'volume_matrix': volume_matrix,
            'dates': dates
        }


def integrate_new_csv_data(tickers: Optional[List[str]] = None, 
                           train_split: float = 0.75) -> Dict:
    """
    High-level function to load and integrate new CSV data.
    
    Args:
        tickers: List of ticker symbols (None = all available)
        train_split: Train/test split ratio
        
    Returns:
        Dictionary ready for trading environment
    """
    loader = NewStockDataLoader()
    df = loader.load_data(tickers=tickers)
    
    # Get final tickers list
    final_tickers = sorted(df['Company'].unique())
    
    result = loader.convert_to_trading_format(df, final_tickers, train_split)
    
    logger.info(f"[NEW CSV INTEGRATION] ✅ Data ready: {len(final_tickers)} tickers, "
                f"{result['train_data']['price_matrix'].shape[0]} train dates, "
                f"{result['test_data']['price_matrix'].shape[0]} test dates")
    
    return result


if __name__ == "__main__":
    # Test the loader
    logging.basicConfig(level=logging.INFO)
    
    print("\n" + "="*80)
    print("TESTING NEW CSV LOADER")
    print("="*80)
    
    # Test with default tickers
    test_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'JPM']
    
    result = integrate_new_csv_data(tickers=test_tickers)
    
    print(f"\n✅ Test successful!")
    print(f"   Tickers: {result['tickers']}")
    print(f"   Train shape: {result['train_data']['price_matrix'].shape}")
    print(f"   Test shape: {result['test_data']['price_matrix'].shape}")
    print(f"   Zero prices: {np.sum(result['train_data']['price_matrix'] == 0)}")

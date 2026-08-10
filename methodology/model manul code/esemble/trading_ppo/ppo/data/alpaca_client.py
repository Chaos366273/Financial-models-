"""
Alpaca API client for data fetching
Handles market data retrieval and account management
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple, Any, cast
import logging

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

from ..core.exceptions import APIException, DataException

# Try to import crypto client (may not be available in all plans)
try:
    from alpaca.data.historical import CryptoHistoricalDataClient
    from alpaca.data.requests import CryptoBarsRequest
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False
    CryptoHistoricalDataClient = None
    CryptoBarsRequest = None

class AlpacaDataClient:
    """Alpaca API client for data fetching and trading"""
    
    def __init__(self, api_key: str, secret_key: str, paper: bool = True):
        """
        Initialize Alpaca client
        
        Args:
            api_key: Alpaca API key
            secret_key: Alpaca secret key
            paper: Use paper trading
        """
        self.api_key = api_key
        self.secret_key = secret_key
        self.paper = paper
        
        # Initialize clients
        self.data_client = StockHistoricalDataClient(api_key, secret_key)
        self.trading_client = TradingClient(api_key, secret_key, paper=paper)
        
        # Initialize crypto client if available
        if CRYPTO_AVAILABLE and CryptoHistoricalDataClient:
            try:
                self.crypto_client = CryptoHistoricalDataClient(api_key, secret_key)
            except Exception:
                self.crypto_client = None
        else:
            self.crypto_client = None
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Rate limiting
        self.request_count = 0
        self.request_minute = int(datetime.now().timestamp() // 60)
        self.request_limit_per_min = 200
        
    def increment_request_count(self):
        """Track API request rate limiting"""
        current_minute = int(datetime.now().timestamp() // 60)
        if current_minute != self.request_minute:
            self.request_minute = current_minute
            self.request_count = 0
        self.request_count += 1
        
        if self.request_count > self.request_limit_per_min * 0.8:
            self.logger.warning(f"Approaching API rate limit: {self.request_count}/{self.request_limit_per_min}")
    
    def get_historical_data(
        self, 
        symbols: List[str], 
        start_date: str, 
        end_date: str,
        timeframe: Any = TimeFrame.Day
    ) -> pd.DataFrame:
        """
        Fetch historical data from Alpaca
        
        Args:
            symbols: List of ticker symbols
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            timeframe: Data timeframe
            
        Returns:
            DataFrame with historical data
        """
        try:
            self.increment_request_count()
            
            # CRITICAL FIX: Add detailed logging for each symbol (ASCII only for Windows consoles)
            print(f"[ALPACA] Downloading data for {len(symbols)} symbols: {symbols}")
            
            # Convert string dates to datetime objects
            if isinstance(start_date, str):
                try:
                    start_ts = pd.Timestamp(start_date)
                    if start_ts is not pd.NaT:
                        start_dt_val = start_ts.to_pydatetime()
                        start_dt = cast(Optional[datetime], start_dt_val)
                    else:
                        start_dt = None
                except (ValueError, TypeError):
                    start_dt = None
            else:
                start_dt = start_date if isinstance(start_date, datetime) else None
                
            if isinstance(end_date, str):
                try:
                    end_ts = pd.Timestamp(end_date)
                    if end_ts is not pd.NaT:
                        end_dt_val = end_ts.to_pydatetime()
                        end_dt = cast(Optional[datetime], end_dt_val)
                    else:
                        end_dt = None
                except (ValueError, TypeError):
                    end_dt = None
            else:
                end_dt = end_date if isinstance(end_date, datetime) else None
            
            request_params = StockBarsRequest(
                symbol_or_symbols=symbols,
                timeframe=timeframe,
                start=start_dt,
                end=end_dt
            )
            
            bars = self.data_client.get_stock_bars(request_params)
            bars_obj = cast(Any, bars)
            df = bars_obj.df.reset_index()
            
            if df.empty:
                print(f"[ALPACA] ERROR: No data received for symbols: {symbols}")
                raise DataException(f"No data received for symbols: {symbols}")
            
            # CRITICAL FIX: Validate each symbol has data
            if 'symbol' in df.columns:
                unique_symbols = df['symbol'].unique()
                missing_symbols = set(symbols) - set(unique_symbols)
                if missing_symbols:
                    print(f"[ALPACA] WARNING: Missing data for symbols: {missing_symbols}")
                    print(f"[ALPACA] OK: Downloaded data for symbols: {list(unique_symbols)}")
            
            # Rename columns for consistency
            df = df.rename(columns={
                'symbol': 'tic',
                'timestamp': 'date'
            })
            
            print(f"[ALPACA] OK: Successfully downloaded {len(df)} rows for {len(df['tic'].unique())} symbols")
            return df
            
        except Exception as e:
            print(f"[ALPACA] ERROR: Download failed for symbols {symbols}: {e}")
            # CRITICAL FIX: Try downloading symbols individually
            print(f"[ALPACA] INFO: Attempting individual symbol downloads...")
            individual_dfs = []
            
            for symbol in symbols:
                try:
                    # Check if symbol is crypto (BTC-USD, ETH-USD, etc.)
                    is_crypto = '-' in symbol and self.crypto_client is not None
                    
                    if is_crypto and self.crypto_client is not None and CryptoBarsRequest is not None:
                        # Use crypto client for crypto symbols
                        if isinstance(start_date, str):
                            try:
                                start_ts = pd.Timestamp(start_date)
                                if start_ts is not pd.NaT:
                                    start_dt_val = start_ts.to_pydatetime()
                                    start_dt = cast(Optional[datetime], start_dt_val)
                                else:
                                    start_dt = None
                            except (ValueError, TypeError):
                                start_dt = None
                        else:
                            start_dt = start_date if isinstance(start_date, datetime) else None
                            
                        if isinstance(end_date, str):
                            try:
                                end_ts = pd.Timestamp(end_date)
                                if end_ts is not pd.NaT:
                                    end_dt_val = end_ts.to_pydatetime()
                                    end_dt = cast(Optional[datetime], end_dt_val)
                                else:
                                    end_dt = None
                            except (ValueError, TypeError):
                                end_dt = None
                        else:
                            end_dt = end_date if isinstance(end_date, datetime) else None
                            
                        crypto_request = CryptoBarsRequest(
                            symbol_or_symbols=[symbol],
                            timeframe=timeframe,
                            start=start_dt,
                            end=end_dt
                        )
                        individual_bars = self.crypto_client.get_crypto_bars(crypto_request)
                        bars_obj = cast(Any, individual_bars)
                        individual_df = bars_obj.df.reset_index()
                    else:
                        # Use stock client for regular symbols
                        if isinstance(start_date, str):
                            try:
                                start_ts = pd.Timestamp(start_date)
                                if start_ts is not pd.NaT:
                                    start_dt_val = start_ts.to_pydatetime()
                                    start_dt = cast(Optional[datetime], start_dt_val)
                                else:
                                    start_dt = None
                            except (ValueError, TypeError):
                                start_dt = None
                        else:
                            start_dt = start_date if isinstance(start_date, datetime) else None
                            
                        if isinstance(end_date, str):
                            try:
                                end_ts = pd.Timestamp(end_date)
                                if end_ts is not pd.NaT:
                                    end_dt_val = end_ts.to_pydatetime()
                                    end_dt = cast(Optional[datetime], end_dt_val)
                                else:
                                    end_dt = None
                            except (ValueError, TypeError):
                                end_dt = None
                        else:
                            end_dt = end_date if isinstance(end_date, datetime) else None
                            
                        individual_request = StockBarsRequest(
                            symbol_or_symbols=[symbol],
                            timeframe=timeframe,
                            start=start_dt,
                            end=end_dt
                        )
                        individual_bars = self.data_client.get_stock_bars(individual_request)
                        bars_obj = cast(Any, individual_bars)
                        individual_df = bars_obj.df.reset_index()
                    
                    if not individual_df.empty:
                        individual_df = individual_df.rename(columns={
                            'symbol': 'tic',
                            'timestamp': 'date'
                        })
                        individual_dfs.append(individual_df)
                        print(f"[ALPACA] OK: Individual download successful for {symbol}")
                    else:
                        print(f"[ALPACA] ERROR: Individual download failed for {symbol} - no data")
                        
                except Exception as individual_error:
                    print(f"[ALPACA] ERROR: Individual download failed for {symbol}: {individual_error}")
            
            if individual_dfs:
                combined_df = pd.concat(individual_dfs, ignore_index=True)
                print(f"[ALPACA] OK: Combined {len(individual_dfs)} successful downloads: {len(combined_df)} total rows")
                return combined_df
            else:
                print(f"[ALPACA] ERROR: All individual downloads failed")
                return pd.DataFrame()
    
    def get_account_info(self) -> Dict:
        """Get account information"""
        try:
            self.increment_request_count()
            account = self.trading_client.get_account()
            account_obj = cast(Any, account)
            
            cash_val = account_obj.cash if hasattr(account_obj, 'cash') else "0"
            equity_val = account_obj.equity if hasattr(account_obj, 'equity') else "0"
            buying_power_val = account_obj.buying_power if hasattr(account_obj, 'buying_power') else "0"
            
            return {
                'cash': float(cash_val) if cash_val is not None else 0.0,
                'equity': float(equity_val) if equity_val is not None else 0.0,
                'buying_power': float(buying_power_val) if buying_power_val is not None else 0.0,
                'daytrade_count': getattr(account_obj, 'daytrade_count', 0),
                'status': getattr(account_obj, 'status', 'UNKNOWN')
            }
        except Exception as e:
            self.logger.error(f"Failed to get account info: {e}")
            raise APIException(f"Account info fetch failed: {e}")
    
    def get_positions(self) -> Dict[str, float]:
        """Get current positions"""
        try:
            self.increment_request_count()
            positions = self.trading_client.get_all_positions()
            
            result = {}
            for p in positions:
                pos_obj = cast(Any, p)
                symbol = getattr(pos_obj, 'symbol', '')
                qty = getattr(pos_obj, 'qty', 0)
                if symbol:
                    result[symbol] = float(qty) if qty is not None else 0.0
            return result
        except Exception as e:
            self.logger.error(f"Failed to get positions: {e}")
            raise APIException(f"Positions fetch failed: {e}")
    
    def place_order(
        self, 
        symbol: str, 
        qty: float, 
        side: OrderSide,
        order_type: str = 'market'
    ) -> Dict:
        """
        Place a trading order
        
        Args:
            symbol: Stock symbol
            qty: Quantity to trade
            side: Buy or sell
            order_type: Order type (market, limit, etc.)
            
        Returns:
            Order information
        """
        try:
            self.increment_request_count()
            
            order = MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=side,
                time_in_force=TimeInForce.DAY
            )
            
            submitted_order = self.trading_client.submit_order(order)
            order_obj = cast(Any, submitted_order)
            
            qty_val = getattr(order_obj, 'qty', 0)
            side_obj = getattr(order_obj, 'side', None)
            side_val = side_obj.value if side_obj is not None and hasattr(side_obj, 'value') else str(side_obj) if side_obj is not None else 'UNKNOWN'
            status_obj = getattr(order_obj, 'status', None)
            status_val = status_obj.value if status_obj is not None and hasattr(status_obj, 'value') else str(status_obj) if status_obj is not None else 'UNKNOWN'
            
            return {
                'id': getattr(order_obj, 'id', ''),
                'symbol': getattr(order_obj, 'symbol', ''),
                'qty': float(qty_val) if qty_val is not None else 0.0,
                'side': side_val,
                'status': status_val
            }
            
        except Exception as e:
            self.logger.error(f"Failed to place order: {e}")
            raise APIException(f"Order placement failed: {e}")
    
    def get_market_status(self) -> Dict:
        """Get current market status"""
        try:
            self.increment_request_count()
            clock = self.trading_client.get_clock()
            clock_obj = cast(Any, clock)
            
            return {
                'is_open': getattr(clock_obj, 'is_open', False),
                'next_open': getattr(clock_obj, 'next_open', None),
                'next_close': getattr(clock_obj, 'next_close', None),
                'timestamp': getattr(clock_obj, 'timestamp', None)
            }
        except Exception as e:
            self.logger.error(f"Failed to get market status: {e}")
            raise APIException(f"Market status fetch failed: {e}")
    
    def validate_data_quality(self, df: pd.DataFrame) -> bool:
        """
        Validate data quality
        
        Args:
            df: DataFrame to validate
            
        Returns:
            True if data is valid
        """
        try:
            # Check for required columns
            required_cols = ['date', 'tic', 'open', 'high', 'low', 'close', 'volume']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                raise DataException(f"Missing required columns: {missing_cols}")
            
            # Check for empty data
            if df.empty:
                raise DataException("DataFrame is empty")
            
            # Check for invalid prices
            price_cols = ['open', 'high', 'low', 'close']
            for col in price_cols:
                if (df[col] <= 0).any():
                    self.logger.warning(f"Invalid prices found in {col}")
                    return False
            
            # Check for missing values
            null_check = df[price_cols].isnull()
            if isinstance(null_check, pd.DataFrame):
                # Check if any column has any null values using numpy
                has_nulls = False
                for col in null_check.columns:
                    col_series = null_check[col]
                    if isinstance(col_series, pd.Series):
                        # Convert to numpy array explicitly for type safety
                        col_array = np.array(col_series.values)
                        if np.any(col_array):
                            has_nulls = True
                            break
                if has_nulls:
                    self.logger.warning("Missing values found in price data")
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Data validation failed: {e}")
            return False

"""
Data loader utilities for PPO Trading System.
Provides:
- load_cached_data(cache_file) -> Optional[pd.DataFrame]
- download_stock_data(tickers, start_date, end_date, cache_file=None) -> pd.DataFrame

This implementation avoids hard dependencies on optional modules by using
importlib at runtime and TYPE_CHECKING for editor hints.
"""

import os
import logging
from typing import List, Optional
import importlib
from typing import TYPE_CHECKING
from multiprocessing import Pool, cpu_count
from functools import partial

import pandas as pd

logger = logging.getLogger(__name__)
if TYPE_CHECKING:
    # Type-only imports to satisfy IDE without creating hard runtime deps
    from . import cache_manager as _cm  # type: ignore # noqa: F401
    from . import alpaca_client as _ac  # type: ignore # noqa: F401
    from . import data_downloader as _dd  # type: ignore # noqa: F401



def load_cached_data(cache_file: str) -> Optional[pd.DataFrame]:
	"""Load cached market data from a pickle file if it exists and is readable."""
	try:
		if not cache_file or not os.path.exists(cache_file):
			return None
		df = pd.read_pickle(cache_file)
		if isinstance(df, pd.DataFrame) and not df.empty:
			return df
		return None
	except Exception as e:
		logger.warning(f"Failed to read cached data '{cache_file}': {e}")
		return None


def _cache_dataframe(df: pd.DataFrame, cache_file: Optional[str]) -> None:
	"""Safely cache a DataFrame to pickle if path provided."""
	if not cache_file:
		return
	try:
		os.makedirs(os.path.dirname(cache_file), exist_ok=True)
		df.to_pickle(cache_file)
		logger.info(f"Cached data to {cache_file}")
	except Exception as e:
		logger.warning(f"Failed to cache data to '{cache_file}': {e}")


def get_cache_path(start_date, end_date, timeframe='1Day'):
	"""Generate cache file path based on date range and timeframe"""
	cache_dir = 'ppo/saved_data/cache'
	os.makedirs(cache_dir, exist_ok=True)
	filename = f"market_data_{start_date}_{end_date}_{timeframe}.pkl"
	return os.path.join(cache_dir, filename)


def load_or_download_data(tickers, start_date, end_date, timeframe='1Day', force_download=False, use_parallel=True):
	"""
	Load from cache or download if not cached
	
	Args:
		tickers: List of stock tickers
		start_date: Start date for data
		end_date: End date for data
		timeframe: '1Day' (default), '5Min', '1Min' (planned)
		force_download: Force download even if cache exists
		use_parallel: Use parallel processing for multiple tickers
	"""
	# Verify timeframe is supported
	if timeframe not in ['1Day']:
		raise ValueError(f"Timeframe '{timeframe}' not yet supported. Use '1Day'")
	
	cache_file = get_cache_path(start_date, end_date, timeframe)
	cached_data = load_cached_data(cache_file)
	
	if not force_download:
		if cached_data is not None:
			logger.info(f"Loaded cached data from {cache_file}")
			return cached_data
	
	# Download if cache miss or force_download
	try:
		data = download_stock_data(tickers, start_date, end_date, cache_file, use_parallel=use_parallel)
		return data
	except (RuntimeError, Exception) as e:
		# If download fails, fall back to cached data if available
		if cached_data is not None and not cached_data.empty:
			logger.warning(f"Download failed ({e}), falling back to cached data")
			return cached_data
		else:
			# Re-raise if no cached data available
			raise


def _get_alpaca_client():
	"""Dynamically import and construct AlpacaDataClient if available, else None."""
	try:
		mod = importlib.import_module('ppo.data.alpaca_client')
		AlpacaDataClient = getattr(mod, 'AlpacaDataClient', None)
		return AlpacaDataClient
	except Exception as e:
		logger.warning(f"Alpaca client not available: {e}")
		return None


def _get_alpaca_keys() -> Optional[tuple[str, str]]:
	"""Resolve Alpaca keys with env-first, then config.py fallback."""
	api_key = os.getenv("APCA_API_KEY")
	secret_key = os.getenv("APCA_API_SECRET")
	if api_key and secret_key:
		return api_key, secret_key
	# Fallback: config.py in project root
	try:
		cfg_mod = importlib.import_module('config')
		# Common attribute names
		candidates = [
			('APCA_API_KEY', 'APCA_API_SECRET'),
			('ALPACA_API_KEY', 'ALPACA_API_SECRET'),
		]
		for k1, k2 in candidates:
			if hasattr(cfg_mod, k1) and hasattr(cfg_mod, k2):
				return getattr(cfg_mod, k1), getattr(cfg_mod, k2)
		# Nested CONFIG dict fallback
		if hasattr(cfg_mod, 'CONFIG'):
			cfg = getattr(cfg_mod, 'CONFIG')
			creds = None
			# try typical nests
			for key in ('ALPACA', 'alpaca', 'credentials', 'data', 'DATA_CONFIG'):
				section = cfg.get(key, {}) if isinstance(cfg, dict) else {}
				if isinstance(section, dict):
					creds = section.get('credentials') or section
					if isinstance(creds, dict):
						ak = creds.get('APCA_API_KEY') or creds.get('ALPACA_API_KEY')
						sk = creds.get('APCA_API_SECRET') or creds.get('ALPACA_API_SECRET')
						if ak and sk:
							return ak, sk
	except Exception:
		pass
	return None


def _try_composite_downloader(tickers: List[str], start_date: str, end_date: str) -> Optional[pd.DataFrame]:
	"""Try optional high-level downloader if present; return DataFrame or None."""
	try:
		mod = importlib.import_module('ppo.data.data_downloader')
		dl_func = getattr(mod, 'download_and_preprocess_data', None)
		if callable(dl_func):
			result = dl_func(tickers, start_date, end_date)
			# Ensure result is a DataFrame or None
			if isinstance(result, pd.DataFrame):
				return result
			return None
	except Exception:
		# Optional path not available; proceed without warning
		pass
	return None


def _download_single_ticker(ticker: str, start_date: str, end_date: str, client) -> Optional[pd.DataFrame]:
	"""Download data for a single ticker - used for parallel processing"""
	try:
		tdf = client.get_historical_data(symbols=[ticker], start_date=start_date, end_date=end_date)
		if isinstance(tdf, pd.DataFrame) and not tdf.empty:
			tdf = tdf.copy()
			# Normalize expected columns
			if 'symbol' not in tdf.columns and 'tic' in tdf.columns:
				tdf['symbol'] = tdf['tic']
			else:
				tdf['symbol'] = ticker
			if 'timestamp' not in tdf.columns and 'date' in tdf.columns:
				# Ensure datetime
				tdf['timestamp'] = pd.to_datetime(tdf['date'])
			logger.info(f"Downloaded {len(tdf)} rows for {ticker}")
			return tdf
	except Exception as e:
		logger.warning(f"Failed to download {ticker}: {e}")
	return None


def download_stock_data(
	tickers: List[str],
	start_date: str,
	end_date: str,
	cache_file: Optional[str] = None,
	use_parallel: bool = True,
) -> pd.DataFrame:
	"""
	Download OHLCV data for tickers between start_date and end_date.
	Tries optional composite downloader first, then Alpaca API.
	"""
	# Optional composite path
	df = _try_composite_downloader(tickers, start_date, end_date)
	if isinstance(df, pd.DataFrame) and not df.empty:
		_cache_dataframe(df, cache_file)
		return df

	# Direct Alpaca path
	AlpacaDataClient = _get_alpaca_client()
	if AlpacaDataClient is None:
		raise RuntimeError("Alpaca API not available and no composite downloader found.")

	keys = _get_alpaca_keys()
	if not keys:
		raise RuntimeError("Missing Alpaca API credentials. Set APCA_API_KEY/APCA_API_SECRET in env or config.py.")
	api_key, secret_key = keys

	client = AlpacaDataClient(api_key, secret_key, paper=True)
	all_data = []
	
	# Use parallel processing for multiple tickers
	if use_parallel and len(tickers) > 1:
		logger.info(f"Using parallel processing for {len(tickers)} tickers")
		# Create partial function with client bound
		download_func = partial(_download_single_ticker, start_date=start_date, end_date=end_date, client=client)
		
		# Use multiprocessing pool
		with Pool(min(cpu_count(), len(tickers))) as pool:
			results = pool.map(download_func, tickers)
		
		# Filter out None results
		all_data = [df for df in results if df is not None]
	else:
		# Sequential processing (fallback or single ticker)
		logger.info(f"Using sequential processing for {len(tickers)} tickers")
		for ticker in tickers:
			tdf = _download_single_ticker(ticker, start_date, end_date, client)
			if tdf is not None:
				all_data.append(tdf)

	if not all_data:
		raise RuntimeError("No data downloaded for requested tickers.")

	df = pd.concat(all_data, ignore_index=True)
	# Basic normalization: ensure timestamp exists and is datetime
	if "timestamp" in df.columns:
		try:
			df["timestamp"] = pd.to_datetime(df["timestamp"])
		except Exception:
			pass

	_cache_dataframe(df, cache_file)
	return df

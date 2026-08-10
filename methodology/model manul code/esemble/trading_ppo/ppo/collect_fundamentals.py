# -*- coding: utf-8 -*-
import os
import sys
import pickle
import logging
from pathlib import Path
from datetime import datetime
import time
import requests
from typing import Dict, Optional

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Alpaca API Configuration (hardcoded as per user requirement)
ALPACA_API_KEY = "PK7XOIVRKYPXMM4PIA6RDK5EMF"
ALPACA_SECRET_KEY = "3R4H1yBurMRHEirWoivTJSdd1veUA4P49qRttn2NhsDL"
ALPACA_BASE_URL = "https://paper-api.alpaca.markets/v2"

# Try to import various data sources (in order of preference)
YFINANCE_AVAILABLE = False
DEFEATBETA_AVAILABLE = False
FINANCETOOLKIT_AVAILABLE = False

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    pass

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    from defeatbeta_api.data.ticker import Ticker as DefeatBetaTicker  # type: ignore
    DEFEATBETA_AVAILABLE = True
    logger.info("✅ defeat-beta-api available (no rate limits!)")
except ImportError:
    pass

try:
    from financetoolkit import Toolkit  # type: ignore
    FINANCETOOLKIT_AVAILABLE = True
    logger.info("✅ FinanceToolkit available")
except ImportError:
    pass

TICKERS = ['AAPL', 'MSFT', 'NVDA', 'JPM', 'SPY', 'GLD', 'BTC-USD', 'KO', 'JNJ', 'AMZN', 'XLE', 'TLT']
CACHE_DIR = Path("ppo/saved_data/fundamental_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
# Same path as extract_fundamental_5y_yahoo.py output; used when CSV exists so PPO can use it
FUNDAMENTAL_CSV_DIR = Path("ppo/saved_data/csv_data")
DEFAULT_FUNDAMENTAL_CSV = FUNDAMENTAL_CSV_DIR / "Fundamental_Data.csv"
FUNDAMENTAL_CSV_NEW = FUNDAMENTAL_CSV_DIR / "Fundamental_Data_new.csv"


def get_fundamental_csv_path() -> Path:
    """Return path to use for fundamental CSV: main file, or _new if main is missing/older (e.g. written when main was locked)."""
    main = DEFAULT_FUNDAMENTAL_CSV
    fallback = FUNDAMENTAL_CSV_NEW
    if main.exists() and fallback.exists():
        if fallback.stat().st_mtime > main.stat().st_mtime:
            return fallback
        return main
    if fallback.exists():
        return fallback
    return main

# Rate limiting constants - increased to avoid Yahoo Finance rate limits
YFINANCE_MIN_DELAY = 5.0  # Minimum delay between yfinance requests (seconds) - increased from 2.0
YFINANCE_MAX_RETRIES = 3  # Reduced retries to avoid long waits
YFINANCE_BACKOFF_BASE = 2.5  # Exponential backoff multiplier
YFINANCE_INITIAL_DELAY = 10.0  # Initial delay before first request (let rate limit reset)

def get_fundamental_data_alpaca(ticker: str) -> Optional[Dict]:
    """
    NOTE: Alpaca Market Data API does NOT provide fundamental data (PE, ROE, etc.)
    It only provides price/market data. This function checks asset info but returns None
    since Alpaca doesn't have fundamental metrics.
    """
    headers = {
        "APCA-API-KEY-ID": ALPACA_API_KEY,
        "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY,
        "accept": "application/json"
    }
    
    # Alpaca doesn't provide fundamental data, but we can verify the asset exists
    try:
        asset_url = f"{ALPACA_BASE_URL}/assets/{ticker}"
        response = requests.get(asset_url, headers=headers, timeout=5)
        
        if response.status_code == 200:
            asset_info = response.json()
            logger.debug(f"  ℹ️ Alpaca asset verified for {ticker}: {asset_info.get('name', 'N/A')}")
        else:
            logger.debug(f"  ⚠️ Alpaca asset endpoint: HTTP {response.status_code} for {ticker}")
    except Exception as e:
        logger.debug(f"  ⚠️ Alpaca asset endpoint failed for {ticker}: {e}")
    
    # Alpaca doesn't provide fundamental data, return None to use yfinance
    return None

def _extract_alpaca_fundamentals(raw: dict) -> Optional[Dict]:
    """Extract and normalize fundamental metrics from Alpaca response"""
    if not raw or not isinstance(raw, dict):
        return None
    
    metrics = {}
    
    # Map Alpaca field names to our standard names
    field_mapping = {
        'peRatioTTM': 'pe_ratio',
        'priceToBook': 'pb_ratio',
        'debtToEquity': 'debt_to_equity',
        'returnOnEquityTTM': 'roe',
        'currentRatio': 'current_ratio',
        'revenueGrowthTTMYoy': 'revenue_growth',
        'epsGrowthTTMYoy': 'earnings_growth',
        # Also try direct keys (normalized to match trading environment)
        'pe_ratio': 'pe_ratio',
        'roe': 'roe',
        'debt_equity': 'debt_equity',
        'debt_to_equity': 'debt_equity',  # Map to debt_equity for compatibility
        'revenue_growth': 'revenue_growth',
        'earnings_growth': 'earnings_growth',
        'profit_margin': 'profit_margin',
    }
    
    for src_key, dst_key in field_mapping.items():
        if src_key in raw and raw[src_key] is not None:
            try:
                metrics[dst_key] = float(raw[src_key])
            except (ValueError, TypeError):
                continue
    
    # Calculate profit margin if we have revenue and net income
    if 'profit_margin' not in metrics:
        if 'netIncome' in raw and 'totalRevenue' in raw:
            try:
                net_income = float(raw['netIncome'])
                revenue = float(raw['totalRevenue'])
                if revenue != 0:
                    metrics['profit_margin'] = net_income / revenue
            except (ValueError, TypeError, ZeroDivisionError):
                pass
    
    return metrics if metrics else None

def get_fundamental_data_defeatbeta(ticker: str) -> Optional[Dict]:
    """
    Fetch fundamental data using defeat-beta-api (no rate limits!).
    This is a reliable alternative to yfinance that uses Hugging Face datasets.
    """
    if not DEFEATBETA_AVAILABLE:
        return None
    
    try:
        ticker_obj = DefeatBetaTicker(ticker)
        
        # Get key metrics
        metrics = {}
        
        # Try to get income statement for revenue/earnings
        try:
            income = ticker_obj.quarterly_income_statement()
            if income is not None and not income.empty:
                # Get latest quarter data
                latest = income.iloc[0] if len(income) > 0 else None
                if latest is not None:
                    # Revenue growth (if we have multiple quarters)
                    if len(income) > 1:
                        prev_revenue = income.iloc[1].get('Total Revenue', 0) if len(income) > 1 else 0
                        curr_revenue = latest.get('Total Revenue', 0)
                        if prev_revenue and prev_revenue > 0:
                            metrics['revenue_growth'] = (curr_revenue - prev_revenue) / prev_revenue
        except Exception as e:
            logger.debug(f"defeat-beta income statement failed for {ticker}: {e}")
        
        # Try to get balance sheet for debt/equity
        try:
            balance = ticker_obj.quarterly_balance_sheet()
            if balance is not None and not balance.empty:
                latest = balance.iloc[0] if len(balance) > 0 else None
                if latest is not None:
                    total_debt = latest.get('Total Debt', 0) or latest.get('Total Liabilities', 0) or 0
                    equity = latest.get('Total Stockholders Equity', 0) or latest.get('Shareholders Equity', 0) or 0
                    if equity and equity > 0:
                        metrics['debt_equity'] = total_debt / equity
        except Exception as e:
            logger.debug(f"defeat-beta balance sheet failed for {ticker}: {e}")
        
        # Try to get key metrics/ratios
        try:
            # Get TTM (Trailing Twelve Months) metrics if available
            key_metrics = ticker_obj.key_metrics()
            if key_metrics is not None:
                # Extract PE ratio, ROE, etc. from key metrics
                # Note: Structure may vary, adjust based on actual API response
                pass
        except Exception as e:
            logger.debug(f"defeat-beta key metrics failed for {ticker}: {e}")
        
        # Try to get financial ratios
        try:
            ratios = ticker_obj.financial_ratios()
            if ratios is not None and not ratios.empty:
                latest = ratios.iloc[0] if len(ratios) > 0 else None
                if latest is not None:
                    if 'PE Ratio' in latest or 'Price to Earnings' in latest:
                        pe = latest.get('PE Ratio') or latest.get('Price to Earnings')
                        if pe:
                            metrics['pe_ratio'] = float(pe)
                    if 'ROE' in latest or 'Return on Equity' in latest:
                        roe = latest.get('ROE') or latest.get('Return on Equity')
                        if roe:
                            metrics['roe'] = float(roe)
                    if 'Current Ratio' in latest:
                        cr = latest.get('Current Ratio')
                        if cr:
                            metrics['current_ratio'] = float(cr)
                    if 'Price to Book' in latest or 'PB Ratio' in latest:
                        pb = latest.get('Price to Book') or latest.get('PB Ratio')
                        if pb:
                            metrics['pb_ratio'] = float(pb)
        except Exception as e:
            logger.debug(f"defeat-beta ratios failed for {ticker}: {e}")
        
        if metrics:
            logger.info(f"✅ Collected {len(metrics)} metrics for {ticker} from defeat-beta-api")
            return metrics
        else:
            logger.debug(f"No metrics extracted from defeat-beta-api for {ticker}")
            return None
            
    except Exception as e:
        logger.debug(f"defeat-beta-api failed for {ticker}: {e}")
        return None

def get_fundamental_data_financetoolkit(ticker: str, api_key: Optional[str] = None) -> Optional[Dict]:
    """
    Fetch fundamental data using FinanceToolkit (requires Financial Modeling Prep API key).
    Falls back to Yahoo Finance if API key not provided.
    """
    if not FINANCETOOLKIT_AVAILABLE:
        return None
    
    try:
        # FinanceToolkit requires API key for full access, but can work without for basic data
        if api_key:
            toolkit = Toolkit(tickers=[ticker], api_key=api_key)
        else:
            # Try without API key (limited functionality)
            toolkit = Toolkit(tickers=[ticker])
        
        metrics = {}
        
        # Get financial ratios
        try:
            ratios = toolkit.ratios.get_all_ratios(ticker)
            if ratios is not None and not ratios.empty:
                latest = ratios.iloc[-1]  # Most recent
                
                # Extract key ratios
                if 'Price Earnings Ratio' in latest:
                    metrics['pe_ratio'] = float(latest['Price Earnings Ratio'])
                if 'Return on Equity' in latest:
                    metrics['roe'] = float(latest['Return on Equity'])
                if 'Debt to Equity Ratio' in latest:
                    metrics['debt_equity'] = float(latest['Debt to Equity Ratio'])
                if 'Current Ratio' in latest:
                    metrics['current_ratio'] = float(latest['Current Ratio'])
                if 'Price to Book Ratio' in latest:
                    metrics['pb_ratio'] = float(latest['Price to Book Ratio'])
        except Exception as e:
            logger.debug(f"FinanceToolkit ratios failed for {ticker}: {e}")
        
        if metrics:
            logger.info(f"✅ Collected {len(metrics)} metrics for {ticker} from FinanceToolkit")
            return metrics
        else:
            return None
            
    except Exception as e:
        logger.debug(f"FinanceToolkit failed for {ticker}: {e}")
        return None

def get_fundamental_data_yahoo_direct(ticker: str) -> Optional[Dict]:
    """
    Fetch fundamental data directly from Yahoo Finance API using requests.
    This avoids yfinance library and may have different rate limiting.
    """
    try:
        import json
        import urllib.parse
        
        # Yahoo Finance API endpoint
        url = f"https://query1.finance.yahoo.com/v10/finance/quoteSummary/{ticker}"
        params = {
            'modules': 'financialData,defaultKeyStatistics,summaryDetail',
            'formatted': 'false'
        }
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json'
        }
        
        response = requests.get(url, params=params, headers=headers, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            result = data.get('quoteSummary', {}).get('result', [])
            if not result:
                return None
            
            info = result[0]
            metrics = {}
            
            # Extract financial data
            financial_data = info.get('financialData', {})
            default_key_stats = info.get('defaultKeyStatistics', {})
            summary_detail = info.get('summaryDetail', {})
            
            # PE Ratio
            if default_key_stats.get('trailingPE'):
                metrics['pe_ratio'] = float(default_key_stats['trailingPE'])
            elif default_key_stats.get('forwardPE'):
                metrics['pe_ratio'] = float(default_key_stats['forwardPE'])
            
            # Price to Book
            if default_key_stats.get('priceToBook'):
                metrics['pb_ratio'] = float(default_key_stats['priceToBook'])
            
            # ROE
            if default_key_stats.get('returnOnEquity'):
                metrics['roe'] = float(default_key_stats['returnOnEquity'])
            
            # Debt to Equity
            if default_key_stats.get('debtToEquity'):
                metrics['debt_equity'] = float(default_key_stats['debtToEquity'])
            
            # Current Ratio
            if default_key_stats.get('currentRatio'):
                metrics['current_ratio'] = float(default_key_stats['currentRatio'])
            
            # Revenue Growth
            if financial_data.get('revenueGrowth'):
                metrics['revenue_growth'] = float(financial_data['revenueGrowth'])
            
            # Earnings Growth
            if default_key_stats.get('earningsQuarterlyGrowth'):
                metrics['earnings_growth'] = float(default_key_stats['earningsQuarterlyGrowth'])
            
            # Profit Margin
            if financial_data.get('profitMargins'):
                metrics['profit_margin'] = float(financial_data['profitMargins'])
            
            if metrics:
                logger.info(f"✅ Collected {len(metrics)} metrics for {ticker} from Yahoo Finance API (direct)")
                return metrics
        elif response.status_code == 429:
            logger.warning(f"Rate limited (429) for {ticker} via direct API")
        else:
            logger.debug(f"Direct API returned {response.status_code} for {ticker}")
            
    except Exception as e:
        logger.debug(f"Direct Yahoo Finance API failed for {ticker}: {e}")
    
    return None

def get_fundamental_data_yfinance(ticker: str, is_first_ticker: bool = False) -> Optional[Dict]:
    """
    Get fundamental data from yfinance with aggressive rate limiting and retry logic.
    
    Args:
        ticker: Stock ticker symbol
        is_first_ticker: If True, add extra initial delay to let rate limits reset
    """
    if not YFINANCE_AVAILABLE:
        return None
    
    import random
    
    for attempt in range(YFINANCE_MAX_RETRIES):
        try:
            # Add delay before request (exponential backoff for retries)
            if attempt > 0:
                # Exponential backoff with jitter to avoid pattern detection
                base_delay = YFINANCE_MIN_DELAY * (YFINANCE_BACKOFF_BASE ** (attempt - 1))
                jitter = random.uniform(0.5, 1.5)  # Add random jitter
                delay = base_delay * jitter
                logger.info(f"Retrying {ticker} (attempt {attempt + 1}/{YFINANCE_MAX_RETRIES}) after {delay:.1f}s delay...")
                time.sleep(delay)
            else:
                # First attempt: use minimum delay with jitter
                if is_first_ticker:
                    # Extra delay for first ticker to let rate limits reset
                    delay = YFINANCE_INITIAL_DELAY + random.uniform(0, 5)
                    logger.info(f"Initial delay for {ticker} (letting rate limits reset): {delay:.1f}s")
                    time.sleep(delay)
                else:
                    delay = YFINANCE_MIN_DELAY + random.uniform(0, 2)  # Add jitter
                    time.sleep(delay)
            
            # Use session to maintain cookies and reduce rate limiting
            import yfinance as yf
            stock = yf.Ticker(ticker)
            
            # Try to get info with timeout
            try:
                info = stock.info
            except Exception as e:
                # If info fails, try using download method as fallback
                logger.debug(f"stock.info failed for {ticker}, trying alternative method: {e}")
                try:
                    # Alternative: use fast_info or download
                    hist = stock.history(period="1d")
                    if hist.empty:
                        raise ValueError("No data available")
                    # For now, if we can't get info, return None and retry
                    if attempt < YFINANCE_MAX_RETRIES - 1:
                        continue
                    return None
                except:
                    if attempt < YFINANCE_MAX_RETRIES - 1:
                        continue
                    return None
            
            # Check if we got rate limited (empty or error response)
            if not info or len(info) < 5:
                if attempt < YFINANCE_MAX_RETRIES - 1:
                    continue
                else:
                    logger.warning(f"Empty response from yfinance for {ticker}")
                    return None
            
            metrics = {}
            # PE Ratio
            if 'trailingPE' in info and info['trailingPE'] is not None:
                metrics['pe_ratio'] = float(info['trailingPE'])
            elif 'forwardPE' in info and info['forwardPE'] is not None:
                metrics['pe_ratio'] = float(info['forwardPE'])
            
            # Price to Book Ratio
            if 'priceToBook' in info and info['priceToBook'] is not None:
                metrics['pb_ratio'] = float(info['priceToBook'])
            
            # Return on Equity
            if 'returnOnEquity' in info and info['returnOnEquity'] is not None:
                metrics['roe'] = float(info['returnOnEquity'])
            
            # Debt to Equity (normalized to 'debt_equity' for trading environment compatibility)
            if 'debtToEquity' in info and info['debtToEquity'] is not None:
                metrics['debt_equity'] = float(info['debtToEquity'])
            elif 'totalDebt' in info and 'totalStockholdersEquity' in info:
                if info['totalStockholdersEquity'] and info['totalStockholdersEquity'] != 0:
                    metrics['debt_equity'] = float(info.get('totalDebt', 0) / info['totalStockholdersEquity'])
            
            # Current Ratio
            if 'currentRatio' in info and info['currentRatio'] is not None:
                metrics['current_ratio'] = float(info['currentRatio'])
            
            # Revenue Growth
            if 'revenueGrowth' in info and info['revenueGrowth'] is not None:
                metrics['revenue_growth'] = float(info['revenueGrowth'])
            
            # Earnings Growth
            if 'earningsGrowth' in info and info['earningsGrowth'] is not None:
                metrics['earnings_growth'] = float(info['earningsGrowth'])
            
            # Profit Margin
            if 'profitMargins' in info and info['profitMargins'] is not None:
                metrics['profit_margin'] = float(info['profitMargins'])
            
            if metrics:
                logger.info(f"✅ Collected {len(metrics)} metrics for {ticker} from yfinance")
                return metrics
            else:
                logger.warning(f"No metrics extracted for {ticker}")
                return None
                
        except requests.exceptions.HTTPError as e:
            if e.response and e.response.status_code == 429:
                # Rate limited - exponential backoff
                if attempt < YFINANCE_MAX_RETRIES - 1:
                    delay = YFINANCE_MIN_DELAY * (YFINANCE_BACKOFF_BASE ** attempt)
                    logger.warning(f"Rate limited (429) for {ticker}, waiting {delay:.1f}s before retry...")
                    time.sleep(delay)
                    continue
                else:
                    logger.error(f"Rate limit exceeded for {ticker} after {YFINANCE_MAX_RETRIES} attempts")
                    return None
            else:
                logger.warning(f"HTTP error for {ticker}: {e}")
                if attempt < YFINANCE_MAX_RETRIES - 1:
                    continue
                return None
        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg or "Too Many Requests" in error_msg:
                # Rate limited in error message
                if attempt < YFINANCE_MAX_RETRIES - 1:
                    delay = YFINANCE_MIN_DELAY * (YFINANCE_BACKOFF_BASE ** attempt)
                    logger.warning(f"Rate limited for {ticker}, waiting {delay:.1f}s before retry...")
                    time.sleep(delay)
                    continue
                else:
                    logger.error(f"Rate limit exceeded for {ticker} after {YFINANCE_MAX_RETRIES} attempts")
                    return None
            else:
                logger.warning(f"Failed to get yfinance data for {ticker} (attempt {attempt + 1}): {e}")
                if attempt < YFINANCE_MAX_RETRIES - 1:
                    continue
                return None
    
    return None

def calculate_financial_health_score(metrics):
    score = 50.0
    if 'pe_ratio' in metrics:
        pe = metrics['pe_ratio']
        if 12 <= pe <= 20:
            score += 15
        elif 20 < pe <= 25:
            score += 10
        elif 25 < pe <= 30:
            score += 5
        elif pe > 30:
            score -= 10
    if 'roe' in metrics:
        roe = metrics['roe']
        if roe > 0.15:
            score += 15
        elif roe > 0.10:
            score += 10
        elif roe > 0.05:
            score += 5
        elif roe < 0:
            score -= 15
    if 'debt_equity' in metrics:
        de = metrics['debt_equity']
    elif 'debt_to_equity' in metrics:  # Backward compatibility
        de = metrics['debt_to_equity']
        if de < 0.5:
            score += 10
        elif de < 1.0:
            score += 5
        elif de > 1.0:
            score -= 10
    if 'revenue_growth' in metrics and 'earnings_growth' in metrics:
        rev_growth = metrics['revenue_growth']
        earn_growth = metrics['earnings_growth']
        if rev_growth > 0.10 and earn_growth > 0.10:
            score += 10
        elif rev_growth > 0.05 and earn_growth > 0.05:
            score += 5
        elif rev_growth < 0 or earn_growth < 0:
            score -= 10
    return max(0.0, min(100.0, score))

def get_fundamental_data_from_csv(csv_path: str, ticker: str) -> Optional[Dict]:
    """
    Load fundamental data from a CSV file.
    
    CSV format should have columns like: ticker, pe_ratio, roe, debt_equity, etc.
    Or: Symbol, PE Ratio, ROE, Debt/Equity, etc.
    
    Args:
        csv_path: Path to CSV file containing fundamental data
        ticker: Stock ticker symbol to look up
    """
    try:
        import pandas as pd
        
        if not Path(csv_path).exists():
            logger.debug(f"CSV file not found: {csv_path}")
            return None
        
        df = pd.read_csv(csv_path)
        
        # Try different column name variations
        ticker_col = None
        for col in df.columns:
            if col.lower() in ['ticker', 'symbol', 'sym', 'stock']:
                ticker_col = col
                break
        
        if ticker_col is None:
            logger.warning(f"No ticker column found in CSV: {csv_path}")
            return None
        
        # Find row for this ticker (use latest fiscal year if Fiscal_Year present)
        ticker_upper = ticker.upper()
        row = df[df[ticker_col].str.upper() == ticker_upper]
        
        if row.empty:
            logger.debug(f"Ticker {ticker} not found in CSV")
            return None
        
        if "Fiscal_Year" in df.columns:
            row = row.sort_values("Fiscal_Year").iloc[-1]
        else:
            row = row.iloc[0]
        metrics = {}
        
        # Map common CSV column names to our standard names (extract_fundamental_5y_yahoo.py outputs)
        column_mapping = {
            'pe_ratio': ['pe_ratio', 'pe ratio', 'p/e', 'price/earnings', 'trailing pe'],
            'pb_ratio': ['pb_ratio', 'pb ratio', 'p/b', 'price/book', 'price to book'],
            'roe': ['roe', 'return on equity', 'return_on_equity', 'ROE'],
            'debt_equity': ['debt_equity', 'debt_to_equity', 'debt/equity', 'debt to equity', 'd/e', 'Debt/Equity'],
            'current_ratio': ['current_ratio', 'current ratio', 'cr', 'Current Ratio'],
            'revenue_growth': ['revenue_growth', 'revenue growth', 'revenue_growth_yoy', 'Revenue Growth'],
            'earnings_growth': ['earnings_growth', 'earnings growth', 'eps_growth', 'eps growth', 'Earnings Growth']
        }
        
        # Extract metrics using column mapping
        for metric_key, possible_names in column_mapping.items():
            for col_name in df.columns:
                if col_name.lower().replace(' ', '_').replace('-', '_') in [n.lower().replace(' ', '_').replace('-', '_') for n in possible_names]:
                    try:
                        value = row[col_name]
                        if pd.notna(value):
                            metrics[metric_key] = float(value)
                            break
                    except (ValueError, TypeError):
                        continue

        # Env requires all 7 keys; fill missing with defaults so cache is valid
        required_defaults = {
            'pe_ratio': 20.0, 'pb_ratio': 2.0, 'roe': 0.10, 'debt_equity': 0.5,
            'current_ratio': 1.5, 'revenue_growth': 0.05, 'earnings_growth': 0.05,
        }
        for k, default in required_defaults.items():
            if k not in metrics or metrics[k] is None:
                metrics[k] = default

        if metrics:
            logger.info(f"✅ Loaded {len(metrics)} metrics for {ticker} from CSV: {csv_path}")
            return metrics
        else:
            logger.debug(f"No metrics extracted from CSV for {ticker}")
            return None
            
    except Exception as e:
        logger.debug(f"Failed to load from CSV {csv_path} for {ticker}: {e}")
        return None

def load_cached_fundamentals(max_age_days: int = 7) -> Optional[Dict]:
    """
    Load cached fundamental data if available and not too old.
    
    Args:
        max_age_days: Maximum age of cache in days (default: 7 days)
    """
    latest_file = CACHE_DIR / "fundamental_data_latest.pkl"
    if latest_file.exists():
        try:
            # Check cache age
            cache_age = datetime.now().timestamp() - latest_file.stat().st_mtime
            cache_age_days = cache_age / (24 * 3600)
            
            if cache_age_days > max_age_days:
                logger.info(f"⚠️ Cache is {cache_age_days:.1f} days old (max: {max_age_days} days), will refresh")
                return None
            
            with open(latest_file, 'rb') as f:
                cached_data = pickle.load(f)
            
            if cached_data and isinstance(cached_data, dict) and len(cached_data) > 0:
                logger.info(f"✅ Found cached fundamental data with {len(cached_data)} tickers (age: {cache_age_days:.1f} days)")
                return cached_data
            else:
                logger.warning("Cache file exists but is empty or invalid")
                return None
        except Exception as e:
            logger.warning(f"Failed to load cached data: {e}")
    return None

def collect_all_fundamental_data(force_refresh: bool = False, max_cache_age_days: int = 7, csv_file: Optional[str] = None):
    """
    Collect fundamental data for all tickers.
    
    Strategy: cached data (if fresh) → yfinance (with rate limiting) → defaults
    
    Args:
        force_refresh: If True, ignore cache and fetch fresh data from yfinance
        max_cache_age_days: Maximum age of cache in days before refreshing
    """
    all_data = {}
    logger.info(f"Starting fundamental data collection for {len(TICKERS)} tickers...")
    logger.info(f"Strategy: Cached data (if fresh) → yfinance (with rate limiting) → defaults")
    logger.info(f"Note: Alpaca API doesn't provide fundamental data, using yfinance as primary source")
    
    # Try to load cached data (unless forcing refresh)
    cached_data = None
    if not force_refresh:
        cached_data = load_cached_fundamentals(max_age_days=max_cache_age_days)
    
    if force_refresh:
        logger.info("🔄 Force refresh enabled - ignoring cache and fetching fresh data")
    elif cached_data:
        logger.info(f"✅ Using cached data (will only fetch missing tickers from yfinance)")
    
    # Track which tickers need fresh data
    tickers_to_fetch = []
    for ticker in TICKERS:
        if force_refresh or not cached_data or ticker not in cached_data:
            tickers_to_fetch.append(ticker)
    
    if tickers_to_fetch:
        logger.info(f"📥 Will fetch {len(tickers_to_fetch)} tickers from yfinance: {', '.join(tickers_to_fetch)}")
    
    for i, ticker in enumerate(TICKERS, 1):
        logger.info(f"[{i}/{len(TICKERS)}] Collecting data for {ticker}...")
        metrics = None
        
        # Step 1: Use cached data if available and not forcing refresh
        if not force_refresh and cached_data and ticker in cached_data:
            cached_metrics = cached_data[ticker]
            # Check if cached data has required fields (not just defaults)
            if cached_metrics and isinstance(cached_metrics, dict):
                # Check if it's not just default values
                if cached_metrics.get('pe_ratio') != 20.0 or cached_metrics.get('roe') != 0.10:
                    logger.info(f"  → Using cached data for {ticker}")
                    metrics = cached_metrics.copy()
                    # Remove financial_health_score as we'll recalculate it
                    metrics.pop('financial_health_score', None)
        
        # Step 2: Fetch from various sources if needed (in order of preference)
        # Priority 0: CSV file (if provided)
        if not metrics and ticker in tickers_to_fetch and csv_file:
            logger.info(f"  → Loading from CSV file for {ticker}...")
            metrics = get_fundamental_data_from_csv(csv_file, ticker)
        
        # Step 2: Fetch from various sources if needed (in order of preference)
        if not metrics and ticker in tickers_to_fetch:
            # Priority 1: defeat-beta-api (no rate limits!)
            if DEFEATBETA_AVAILABLE:
                logger.info(f"  → Trying defeat-beta-api for {ticker} (no rate limits)...")
                metrics = get_fundamental_data_defeatbeta(ticker)
            
            # Priority 2: FinanceToolkit (if available)
            if not metrics and FINANCETOOLKIT_AVAILABLE:
                logger.info(f"  → Trying FinanceToolkit for {ticker}...")
                # Check for API key in environment
                fmp_key = os.environ.get('FINANCIAL_MODELING_PREP_API_KEY')
                metrics = get_fundamental_data_financetoolkit(ticker, api_key=fmp_key)
            
            # Priority 3: Direct Yahoo Finance API (no library import needed)
            if not metrics:
                logger.info(f"  → Trying direct Yahoo Finance API for {ticker}...")
                metrics = get_fundamental_data_yahoo_direct(ticker)
            
            # Priority 4: yfinance library as last resort (has rate limits)
            if not metrics and YFINANCE_AVAILABLE:
                logger.info(f"  → Trying yfinance library for {ticker} (may hit rate limits)...")
                # Pass is_first_ticker flag for extra initial delay
                is_first = (i == 1 and ticker == tickers_to_fetch[0])
                metrics = get_fundamental_data_yfinance(ticker, is_first_ticker=is_first)
            
            if not metrics:
                logger.warning(f"  ⚠️ All data sources failed for {ticker}")
        
        # Step 3: Use defaults if nothing worked
        if metrics:
            metrics['financial_health_score'] = calculate_financial_health_score(metrics)
            all_data[ticker] = metrics
            logger.info(f"  ✅ {ticker}: Health score = {metrics['financial_health_score']:.1f}, PE = {metrics.get('pe_ratio', 'N/A')}, ROE = {metrics.get('roe', 'N/A')}")
        else:
            logger.warning(f"  ⚠️ No fundamental data collected for {ticker}, using defaults")
            all_data[ticker] = {
                'pe_ratio': 20.0, 
                'pb_ratio': 2.0,
                'roe': 0.10, 
                'debt_equity': 0.5,  # Using 'debt_equity' to match trading environment
                'current_ratio': 1.5,
                'revenue_growth': 0.05, 
                'earnings_growth': 0.05,
                'profit_margin': 0.10,
                'financial_health_score': 50.0
            }
        
        # Rate limiting: Add delay between tickers when fetching from yfinance
        if ticker in tickers_to_fetch and i < len(TICKERS):
            # Add delay with jitter to avoid pattern detection
            import random
            delay = YFINANCE_MIN_DELAY + random.uniform(1, 3)
            logger.debug(f"  ⏳ Waiting {delay:.1f}s before next ticker to avoid rate limits...")
            time.sleep(delay)
    
    logger.info(f"Collected fundamental data for {len(all_data)}/{len(TICKERS)} tickers")
    return all_data

def save_fundamental_cache(data):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cache_file = CACHE_DIR / f"fundamental_data_{timestamp}.pkl"
    with open(cache_file, 'wb') as f:
        pickle.dump(data, f)
    logger.info(f"Saved fundamental cache to {cache_file}")
    latest_file = CACHE_DIR / "fundamental_data_latest.pkl"
    with open(latest_file, 'wb') as f:
        pickle.dump(data, f)
    logger.info(f"Saved as latest cache: {latest_file}")
    return cache_file

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Collect fundamental data for trading tickers')
    parser.add_argument('--force-refresh', action='store_true', 
                       help='Force refresh from APIs, ignoring cache')
    parser.add_argument('--max-cache-age', type=int, default=7,
                       help='Maximum cache age in days (default: 7)')
    parser.add_argument('--csv-file', type=str, default=None,
                       help='Path to CSV file with fundamental data (default: ppo/saved_data/csv_data/Fundamental_Data.csv if present)')
    args = parser.parse_args()
    # Use extract script output by default; prefer _new.csv when it's newer (e.g. extract wrote there because main was locked)
    if args.csv_file is None:
        chosen = get_fundamental_csv_path()
        if chosen.exists():
            args.csv_file = str(chosen)
            logger.info(f"Using fundamental CSV: {args.csv_file}")
    
    logger.info("=" * 60)
    logger.info("FUNDAMENTAL DATA COLLECTION")
    logger.info("=" * 60)
    logger.info(f"Alpaca API: {ALPACA_BASE_URL}")
    logger.info(f"Note: Alpaca doesn't provide fundamental data")
    logger.info("")
    logger.info("Available data sources:")
    logger.info(f"  - defeat-beta-api: {'✅ Available (NO RATE LIMITS!)' if DEFEATBETA_AVAILABLE else '❌ Not installed (pip install defeatbeta-api)'}")
    logger.info(f"  - FinanceToolkit: {'✅ Available' if FINANCETOOLKIT_AVAILABLE else '❌ Not installed (pip install financetoolkit)'}")
    logger.info(f"  - Direct Yahoo API: ✅ Always available")
    logger.info(f"  - yfinance library: {'✅ Available' if YFINANCE_AVAILABLE else '❌ Not installed (pip install yfinance)'}")
    logger.info("")
    logger.info("Priority order: defeat-beta-api → FinanceToolkit → Direct Yahoo → yfinance")
    
    if not any([DEFEATBETA_AVAILABLE, FINANCETOOLKIT_AVAILABLE, YFINANCE_AVAILABLE]):
        logger.warning("⚠️ No advanced data sources available - will use direct Yahoo Finance API only")
        logger.info("   Recommended: pip install defeatbeta-api (no rate limits!)")
    
    fundamental_data = collect_all_fundamental_data(
        force_refresh=args.force_refresh,
        max_cache_age_days=args.max_cache_age,
        csv_file=args.csv_file
    )
    
    if not fundamental_data:
        logger.error("No fundamental data collected!")
        return
    
    cache_file = save_fundamental_cache(fundamental_data)
    
    logger.info("=" * 60)
    logger.info("COLLECTION SUMMARY")
    logger.info("=" * 60)
    for ticker, metrics in fundamental_data.items():
        health_score = metrics.get('financial_health_score', 0.0)
        pe = metrics.get('pe_ratio', 'N/A')
        roe = metrics.get('roe', 'N/A')
        if isinstance(roe, float):
            logger.info(f"{ticker:8s}: Health={health_score:5.1f}, PE={pe:>6}, ROE={roe:>5.2%}")
        else:
            logger.info(f"{ticker:8s}: Health={health_score:5.1f}, PE={pe:>6}, ROE={roe}")
    logger.info("=" * 60)
    logger.info(f"Fundamental data collection complete!")
    logger.info(f"   Cache file: {cache_file}")
    logger.info(f"   Total tickers: {len(fundamental_data)}")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()

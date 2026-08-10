import os
import time
import json
import logging
from typing import Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

REQUIRED_KEYS = (
    'pe_ratio',
    'pb_ratio',
    'debt_equity',
    'roe',
    'current_ratio',
    'revenue_growth',
    'earnings_growth',
)

# Map from likely Alpaca fields to our required schema
FALLBACK_FIELD_MAP = {
    # Alpaca v2 fundamentals typical fields (names may differ by plan/version)
    'peRatioTTM': 'pe_ratio',
    'priceToBook': 'pb_ratio',
    'debtToEquity': 'debt_equity',
    'returnOnEquityTTM': 'roe',
    'currentRatio': 'current_ratio',
    'revenueGrowthTTMYoy': 'revenue_growth',
    'epsGrowthTTMYoy': 'earnings_growth',
}


def _http_get(url: str, headers: Dict[str, str], params: Optional[Dict] = None) -> Optional[dict]:
    try:
        r = requests.get(url, headers=headers, params=params, timeout=10)
        if r.status_code == 200:
            return r.json()
        elif r.status_code == 404:
            logger.debug(f"Endpoint not found: {url}")
        elif r.status_code == 403:
            logger.warning(f"Access forbidden (may need different plan): {url}")
        else:
            logger.debug(f"HTTP {r.status_code}: {r.text[:200]}")
        return None
    except Exception as e:
        logger.debug(f"HTTP request failed: {e}")
        return None


def _extract_required_fields(raw: dict) -> Optional[Dict[str, float]]:
    try:
        normalized: Dict[str, float] = {}
        # Try direct exact keys first
        for key in REQUIRED_KEYS:
            if key in raw:
                v = raw.get(key)
                if v is None:
                    return None
                normalized[key] = float(v)
        # If missing, try mapped fields
        for src, dst in FALLBACK_FIELD_MAP.items():
            if dst not in normalized and src in raw:
                v = raw.get(src)
                if v is None:
                    return None
                normalized[dst] = float(v)
        # Ensure we have all required
        if all(k in normalized for k in REQUIRED_KEYS):
            return normalized
        return None
    except Exception:
        return None


def fetch_fundamentals_from_alpaca(symbols: List[str], api_key: str, secret_key: str) -> Dict[str, Dict[str, float]]:
    """Fetch fundamentals for symbols using Alpaca Market Data API.
    
    Alpaca provides fundamentals data through IEX and Polygon integration.
    Respects API rate limits: 200 calls/min for Basic plan, 10,000/min for Algo Trader Plus.
    
    Returns symbol->metrics dict where metrics contains REQUIRED_KEYS.
    """
    results: Dict[str, Dict[str, float]] = {}
    headers = {
        'APCA-API-KEY-ID': api_key,
        'APCA-API-SECRET-KEY': secret_key,
        'accept': 'application/json',
    }

    # Use Market Data API - process in batches to respect rate limits
    base_v2 = 'https://data.alpaca.markets/v2'
    batch_size = 200  # Basic plan limit: 200 calls/min
    
    logger.info(f"Fetching fundamentals for {len(symbols)} symbols...")
    
    # Try bulk endpoint first (more efficient)
    try:
        for i in range(0, len(symbols), batch_size):
            batch = symbols[i:i + batch_size]
            symbols_param = ','.join(batch)
            
            bulk_url = f"{base_v2}/stocks/fundamentals"
            bulk = _http_get(bulk_url, headers, params={'symbols': symbols_param})
            
            if isinstance(bulk, dict):
                # Response format: { 'fundamentals': [...] } or { 'data': [...] }
                fund_items = bulk.get('fundamentals') or bulk.get('data') or []
                for item in fund_items:
                    sym = str(item.get('symbol') or item.get('Symbol') or '').upper()
                    if not sym:
                        continue
                    normalized = _extract_required_fields(item)
                    if normalized:
                        results[sym] = normalized
                        logger.debug(f"✅ Fetched fundamentals for {sym}")
            
            # Rate limiting: 200 calls/min = 0.3s between batches
            if i + batch_size < len(symbols):
                time.sleep(0.3)
        
        if results:
            logger.info(f"✅ Successfully fetched {len(results)}/{len(symbols)} fundamentals via bulk endpoint")
            return results
    except Exception as e:
        logger.warning(f"Bulk endpoint failed: {e}, trying per-symbol requests")
    
    # Fallback: Try per-symbol requests (slower but more reliable)
    logger.info("Trying per-symbol requests...")
    for sym in symbols:
        usym = sym.upper()
        if usym in results:
            continue
        
        # Try per-symbol endpoint
        per_symbol_url = f"{base_v2}/stocks/{usym}/fundamentals"
        data = _http_get(per_symbol_url, headers)
        
        if isinstance(data, dict):
            # Handle different response formats
            candidates = []
            if 'fundamentals' in data and isinstance(data['fundamentals'], list) and data['fundamentals']:
                candidates.append(data['fundamentals'][0])
            if 'fundamental' in data and isinstance(data['fundamental'], dict):
                candidates.append(data['fundamental'])
            if not candidates and 'data' in data:
                if isinstance(data['data'], dict):
                    candidates.append(data['data'])
                elif isinstance(data['data'], list) and data['data']:
                    candidates.append(data['data'][0])
            
            for raw in candidates:
                normalized = _extract_required_fields(raw)
                if normalized:
                    results[usym] = normalized
                    logger.debug(f"✅ Fetched fundamentals for {usym}")
                    break
        
        # Rate limiting: 0.3s between calls (200/min)
        time.sleep(0.3)
    
    if results:
        logger.info(f"✅ Successfully fetched {len(results)}/{len(symbols)} fundamentals")
    else:
        logger.warning(f"⚠️ No fundamentals data retrieved. Endpoint may not be available or plan doesn't include fundamentals.")
    
    return results


def save_fundamental_cache(data: Dict[str, Dict[str, float]], output_dir: str) -> Optional[str]:
    """Save fundamental data to cache file."""
    if not data:
        logger.warning("No data to save")
        return None
    try:
        os.makedirs(output_dir, exist_ok=True)
        ts = time.strftime('%Y%m%d_%H%M%S')
        path = os.path.join(output_dir, f"fundamental_data_{ts}.pkl")
        latest_path = os.path.join(output_dir, "fundamental_data_latest.pkl")
        import pickle
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        # Also save as latest for easy access
        with open(latest_path, 'wb') as f:
            pickle.dump(data, f)
        logger.info(f"✅ Fundamental cache saved: {path} ({len(data)} tickers)")
        logger.info(f"✅ Latest cache saved: {latest_path}")
        return latest_path  # Return latest path for consistency
    except Exception as e:
        logger.warning(f"Failed to save fundamental cache: {e}")
        return None


def build_and_save_fundamental_cache(tickers: List[str], api_key: str, secret_key: str, output_dir: str) -> Optional[str]:
    """Download and cache fundamental data for tickers.
    
    Returns saved file path, or None if no data was saved.
    """
    logger.info(f"Downloading fundamentals for {len(tickers)} tickers...")
    data = fetch_fundamentals_from_alpaca(tickers, api_key, secret_key)
    if data:
        return save_fundamental_cache(data, output_dir)
    else:
        logger.warning("No fundamental data downloaded - cache not created")
        return None

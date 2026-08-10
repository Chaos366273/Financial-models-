# -*- coding: utf-8 -*-
"""
Extract 5 years of fundamental data from Yahoo Finance and save to CSV.

Output format matches Fundamental_Data.csv: Symbol, Name, Sector, Price,
Price/Earnings, Dividend Yield, Earnings/Share, 52 Week Low, 52 Week High,
Market Cap, EBITDA, Price/Sales, Price/Book, SEC Filings.
Adds Fiscal_Year so multiple rows per ticker (one per year).
Price = period-end close (last close on or before fiscal period end), not live quote.
Dividend Yield = in percent (e.g. 0.5 = 0.5%); Yahoo decimal 0.004 corrected to 0.4.

Output: ppo/saved_data/csv_data/Fundamental_Data.csv
PPO uses it: collect_fundamentals.py reads this path by default when the file exists,
then writes ppo/saved_data/fundamental_cache/fundamental_data_latest.pkl for the env.

Target: 30 companies from config.DATA_CONFIG['tickers'] (largest-cap universe). Uses
expanded Yahoo key variants and info fallbacks so required fields are collected.

Rate limiting: stays under ~60 GET/min (2s between tickers, internal delays)
to avoid Yahoo rate limits. No DSP/API key required for Yahoo Finance.
"""

import os
import sys
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict, Any

import pandas as pd

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

try:
    import yfinance as yf
except ImportError:
    print("Install yfinance: pip install yfinance")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Rate limits: conservative to avoid Yahoo blocks (DSP baseline 60 GET/min)
SEC_BETWEEN_TICKERS = 2.0
SEC_BETWEEN_REQUESTS_PER_TICKER = 0.5
MAX_TICKERS_PER_HOUR = 300
REQUESTS_PER_TICKER = 4  # info, financials, balance_sheet, history

CSV_DIR = Path(__file__).resolve().parent / "saved_data" / "csv_data"
OUTPUT_CSV = CSV_DIR / "Fundamental_Data.csv"
BACKUP_SUFFIX = ".backup"
TARGET_YEARS = 5

# Same column order as existing Fundamental_Data.csv + Fiscal_Year + env-required metrics
OUTPUT_COLUMNS = [
    "Symbol", "Name", "Sector", "Fiscal_Year",
    "Price", "Price/Earnings", "Dividend Yield", "Earnings/Share",
    "52 Week Low", "52 Week High", "Market Cap", "EBITDA",
    "Price/Sales", "Price/Book",
    "ROE", "Debt/Equity", "Current Ratio", "Revenue Growth", "Earnings Growth",
    "SEC Filings",
]


# Fallback if config not available (same 30 as config.DATA_CONFIG['tickers'])
_PPO_TICKERS = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "META", "AMD", "JPM", "V", "BAC", "MA",
    "AMZN", "WMT", "HD", "NKE", "KO", "COST", "JNJ", "UNH", "PFE", "LLY", "ABBV",
    "XOM", "CVX", "VZ", "T", "DIS", "INTC", "QCOM", "ORCL", "CRM",
]

# Same filenames as ppo/data/new_csv_loader.py (stock_details_5_years / 5-year CSV)
DEFAULT_5Y_CSV_NAMES = [
    "New_stock_data_5_years.csv",
    "new stock data 5 years.csv",
    "stock_details_5_years.csv",
]


def get_tickers_from_5y_csv() -> Optional[List[str]]:
    """
    Read all companies from the 5-year stock CSV (stock_details_5_years.csv or same-name variants).
    Use this so Fundamental_Data.csv has the same companies as the trading data.
    Returns None if no CSV found or no ticker column.
    """
    for name in DEFAULT_5Y_CSV_NAMES:
        path = CSV_DIR / name
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path, nrows=100000)
            col = None
            for c in ("Company", "Ticker", "Symbol", "tic"):
                if c in df.columns:
                    col = c
                    break
            if col is None:
                logger.warning("5y CSV %s has no Company/Ticker/Symbol column", path)
                continue
            tickers = sorted(df[col].dropna().astype(str).str.strip().str.upper().unique().tolist())
            # Skip crypto / non-Yahoo-fundamental symbols
            skip = {"BTC-USD", "BTCUSD", "ETH-USD", "ETHUSD"}
            tickers = [t for t in tickers if t and t not in skip]
            if tickers:
                logger.info("Loaded %s tickers from 5y CSV: %s", path.name, len(tickers))
                return tickers
        except Exception as e:
            logger.warning("Failed to read 5y CSV %s: %s", path, e)
    return None


def get_ppo_tickers() -> List[str]:
    """Tickers used by PPO — from config.DATA_CONFIG['tickers'] (single source of truth) or fallback."""
    try:
        import importlib.util
        config_path = project_root / "config.py"
        if not config_path.exists():
            raise FileNotFoundError("config.py not at project root")
        spec = importlib.util.spec_from_file_location("config_root", config_path)
        if spec is None or spec.loader is None:
            raise ValueError("invalid spec")
        cfg = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cfg)
        t = getattr(cfg, "DATA_CONFIG", {}).get("tickers")
        if t and len(t) > 0:
            return [str(s).strip().upper() for s in t]
    except Exception:
        pass
    return list(_PPO_TICKERS)


def default_tickers() -> List[str]:
    """Same as get_ppo_tickers(); kept for backward compatibility."""
    return get_ppo_tickers()


def _safe_float(val: Any) -> Optional[float]:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _safe_str(val: Any) -> str:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return ""
    return str(val).strip()


def _col_for_period(df: pd.DataFrame, period_ts: Any) -> Any:
    """Resolve column key for period (avoids KeyError from timezone/dtype mismatch)."""
    try:
        p = pd.Timestamp(period_ts)
        p_date = p.date() if hasattr(p, "date") else p
    except Exception:
        return period_ts
    for c in df.columns:
        try:
            tc = pd.Timestamp(c)
            c_date = tc.date() if hasattr(tc, "date") else tc
            if c_date == p_date:
                return c
        except Exception:
            continue
    return period_ts


def _get_from_df(df: pd.DataFrame, col: Any, possible_keys: tuple, fallback: Optional[float] = None) -> Optional[float]:
    """Get value from DataFrame row by trying exact index names, then case-insensitive partial match. Used for large-cap Yahoo data."""
    if df is None or df.empty or col not in df.columns:
        return fallback
    index_str = [str(i).strip() for i in df.index]
    for key in possible_keys:
        if key in df.index:
            val = _safe_float(df.loc[key, col])
            if val is not None:
                return val
    for key in possible_keys:
        key_lower = key.lower()
        for i, idx in enumerate(index_str):
            if key_lower in idx.lower():
                val = _safe_float(df.iloc[i].get(col))
                if val is not None:
                    return val
                break
    return fallback


def fetch_one_ticker_5y(ticker: str) -> List[Dict[str, Any]]:
    """
    Fetch up to 5 years of fundamental data for one ticker via yfinance.
    Returns list of row dicts (one per fiscal year).
    """
    rows = []
    try:
        stock = yf.Ticker(ticker)
        time.sleep(SEC_BETWEEN_REQUESTS_PER_TICKER)

        info = getattr(stock, "info", None) or {}
        time.sleep(SEC_BETWEEN_REQUESTS_PER_TICKER)

        financials = getattr(stock, "financials", None)
        if financials is not None and hasattr(financials, "columns"):
            financials = financials
        else:
            financials = pd.DataFrame()
        time.sleep(SEC_BETWEEN_REQUESTS_PER_TICKER)

        balance_sheet = getattr(stock, "balance_sheet", None)
        if balance_sheet is not None and hasattr(balance_sheet, "columns"):
            balance_sheet = balance_sheet
        else:
            balance_sheet = pd.DataFrame()
        time.sleep(SEC_BETWEEN_REQUESTS_PER_TICKER)

        end = datetime.now()
        start = datetime(end.year - 6, 1, 1)
        hist = stock.history(start=start, end=end, auto_adjust=True)
        if hist is None or not isinstance(hist, pd.DataFrame):
            hist = pd.DataFrame()
        # Normalize to timezone-naive so comparisons with pd.Timestamp don't raise
        if not hist.empty and getattr(hist.index, "tz", None) is not None:
            hist = hist.copy()
            try:
                hist.index = hist.index.tz_localize(None)
            except Exception:
                hist.index = hist.index.tz_convert("UTC").tz_localize(None)

        name = _safe_str(info.get("shortName") or info.get("longName") or ticker)
        sector = _safe_str(info.get("sector") or "")
        sec_url = _safe_str(info.get("website") or "")
        if not sec_url and info.get("symbol"):
            sec_url = f"http://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={ticker}"

        # Fiscal years from financials (columns are period-end dates)
        if financials.empty or len(financials.columns) == 0:
            # Fallback: use latest year from info only
            pe = _safe_float(info.get("trailingPE") or info.get("forwardPE"))
            pb = _safe_float(info.get("priceToBook"))
            price = _safe_float(info.get("regularMarketPrice") or info.get("currentPrice"))
            div_yield = _safe_float(info.get("dividendYield") or info.get("yield"))
            if div_yield is not None and div_yield > 0:
                # Yahoo: decimal 0.004 = 0.4%; sometimes 0.41 = 0.41% (already in % points). Only scale when clearly decimal.
                if div_yield < 0.05:
                    div_yield = div_yield * 100
                elif div_yield >= 1 and div_yield <= 100:
                    pass  # already in percent
                elif div_yield > 100:
                    div_yield = min(100.0, div_yield / 100.0)  # sanity: cap and treat as bps
            eps = _safe_float(info.get("trailingEps"))
            mcap = _safe_float(info.get("marketCap"))
            revenue = _safe_float(info.get("totalRevenue"))
            if revenue and price and revenue > 0:
                shares = _safe_float(info.get("sharesOutstanding")) or 1
                ps = (price * shares) / revenue if shares else None
            else:
                ps = None
            fy = datetime.now().year
            low_52 = _safe_float(info.get("fiftyTwoWeekLow"))
            high_52 = _safe_float(info.get("fiftyTwoWeekHigh"))
            ebitda = _safe_float(info.get("ebitda"))
            roe = _safe_float(info.get("returnOnEquity"))
            debt_equity = _safe_float(info.get("debtToEquity"))
            current_ratio = _safe_float(info.get("currentRatio"))
            rows.append({
                "Symbol": ticker,
                "Name": name,
                "Sector": sector,
                "Fiscal_Year": fy,
                "Price": price,
                "Price/Earnings": pe,
                "Dividend Yield": div_yield,
                "Earnings/Share": eps,
                "52 Week Low": low_52,
                "52 Week High": high_52,
                "Market Cap": mcap,
                "EBITDA": ebitda,
                "Price/Sales": ps,
                "Price/Book": pb,
                "ROE": roe,
                "Debt/Equity": debt_equity,
                "Current Ratio": current_ratio,
                "Revenue Growth": None,
                "Earnings Growth": None,
                "SEC Filings": sec_url,
            })
            return rows

        # Take last TARGET_YEARS fiscal periods (exclude future dates — no price data)
        today = pd.Timestamp(datetime.now().date())
        periods = sorted(financials.columns.tolist(), reverse=True)[:TARGET_YEARS]
        periods = [p for p in periods if pd.Timestamp(p) <= today]
        if not periods:
            return rows

        # Keys used by Yahoo for large-cap companies (financials / balance sheet)
        REVENUE_KEYS = ("Total Revenue", "Revenue", "Operating Revenue", "Revenues", "Net Revenue", "Sales Revenue")
        NET_INCOME_KEYS = ("Net Income", "Net Income Common Stockholders", "Net Income Including Noncontrolling Interests", "Net Income Continuous Operations", "Net Income From Continuing Operations")
        EBITDA_KEYS = ("EBITDA", "EBIT", "Operating Income", "Operating Earnings", "Income Before Tax")
        EQUITY_KEYS = ("Total Stockholder Equity", "Total Equity Gross Minority Interest", "Stockholders Equity", "Total Shareholders Equity", "Stockholders' Equity", "Common Stock Equity")
        LIABILITIES_KEYS = ("Total Liabilities", "Liabilities", "Total Liabilities Net Minority Interest")
        CURRENT_ASSETS_KEYS = ("Current Assets", "Total Current Assets", "Current Assets Other")
        CURRENT_LIAB_KEYS = ("Current Liabilities", "Total Current Liabilities", "Current Liabilities Other")
        SHARES_KEYS = ("Share Issued", "Shares Outstanding", "Common Stock Shares Outstanding", "Ordinary Shares Number")

        # Precompute revenue and net income per period for YoY growth
        period_rev_ni: Dict[Any, tuple] = {}
        for p in periods:
            col_f = _col_for_period(financials, p)
            rev_p = _get_from_df(financials, col_f, REVENUE_KEYS)
            if rev_p is None and len(financials.index) > 0:
                rev_p = _safe_float(financials.iloc[0].get(col_f))
            ni_p = _get_from_df(financials, col_f, NET_INCOME_KEYS)
            period_rev_ni[p] = (rev_p, ni_p)

        for idx, period_ts in enumerate(periods):
            col_fin = _col_for_period(financials, period_ts)
            col_bs = _col_for_period(balance_sheet, period_ts) if balance_sheet is not None and not balance_sheet.empty else period_ts
            if hasattr(period_ts, "year"):
                fiscal_year = period_ts.year
            else:
                try:
                    fiscal_year = pd.Timestamp(period_ts).year
                except Exception:
                    fiscal_year = datetime.now().year

            # Revenue (large-cap Yahoo keys)
            rev = _get_from_df(financials, col_fin, REVENUE_KEYS)
            if rev is None and len(financials.index) > 0:
                rev = _safe_float(financials.iloc[0].get(col_fin))

            # Net Income -> EPS
            ni = _get_from_df(financials, col_fin, NET_INCOME_KEYS)
            shares = None
            if balance_sheet is not None and not balance_sheet.empty and col_bs in balance_sheet.columns:
                shares = _get_from_df(balance_sheet, col_bs, SHARES_KEYS)
            if shares is None or shares <= 0:
                shares = _safe_float(info.get("sharesOutstanding"))
            if shares is None or shares <= 0:
                shares = 1
            eps = (ni / shares) if ni is not None and shares else _safe_float(info.get("trailingEps"))

            # EBITDA (large-cap keys)
            ebitda = _get_from_df(financials, col_fin, EBITDA_KEYS)
            if ebitda is None:
                ebitda = _safe_float(info.get("ebitda"))

            # Price at period end (hist.index is already tz-naive)
            if not hist.empty:
                try:
                    period_naive = pd.Timestamp(period_ts)
                    if period_naive.tzinfo is not None:
                        period_naive = period_naive.tz_localize(None)
                    else:
                        period_naive = pd.Timestamp(period_naive.date())
                    on_or_before = hist.index[hist.index <= period_naive]
                    if len(on_or_before) > 0:
                        last_dt = on_or_before[-1]
                        price = _safe_float(hist.loc[last_dt, "Close"] if "Close" in hist.columns else hist.loc[last_dt].iloc[0])
                    else:
                        price = _safe_float(info.get("regularMarketPrice") or info.get("currentPrice"))
                except Exception:
                    price = _safe_float(info.get("regularMarketPrice") or info.get("currentPrice"))
            else:
                price = _safe_float(info.get("regularMarketPrice") or info.get("currentPrice"))

            if price is None or price <= 0:
                price = _safe_float(info.get("regularMarketPrice") or info.get("currentPrice"))

            # 52-week low/high around period end
            low_52 = high_52 = None
            if not hist.empty and len(hist) >= 5:
                try:
                    period_naive = pd.Timestamp(period_ts)
                    if period_naive.tzinfo is not None:
                        period_naive = period_naive.tz_localize(None)
                    period_d = period_naive.date() if hasattr(period_naive, "date") else period_naive
                    one_year_before = pd.Timestamp(period_d) - pd.Timedelta(days=365)
                    end_ts = pd.Timestamp(period_d)
                    window = hist[(hist.index >= one_year_before) & (hist.index <= end_ts)]
                    if not window.empty and "Close" in hist.columns:
                        low_52 = float(window["Close"].min())
                        high_52 = float(window["Close"].max())
                except Exception:
                    pass
            if low_52 is None:
                low_52 = _safe_float(info.get("fiftyTwoWeekLow"))
            if high_52 is None:
                high_52 = _safe_float(info.get("fiftyTwoWeekHigh"))

            pe = (price / eps) if price and eps and eps > 0 else _safe_float(info.get("trailingPE") or info.get("forwardPE"))
            mcap = (price * shares) if price and shares else _safe_float(info.get("marketCap"))
            ps = (price * shares / rev) if (price and shares and rev and rev > 0) else None
            if ps is None and price and shares:
                rev_info = _safe_float(info.get("totalRevenue"))
                if rev_info and rev_info > 0:
                    ps = (price * shares) / rev_info

            # P/B, ROE, Debt/Equity, Current Ratio from balance sheet (large-cap keys)
            pb = None
            te = _get_from_df(balance_sheet, col_bs, EQUITY_KEYS) if balance_sheet is not None and not balance_sheet.empty and col_bs in balance_sheet.columns else None
            total_liabilities = _get_from_df(balance_sheet, col_bs, LIABILITIES_KEYS) if balance_sheet is not None and not balance_sheet.empty and col_bs in balance_sheet.columns else None
            current_assets = _get_from_df(balance_sheet, col_bs, CURRENT_ASSETS_KEYS) if balance_sheet is not None and not balance_sheet.empty and col_bs in balance_sheet.columns else None
            current_liabilities = _get_from_df(balance_sheet, col_bs, CURRENT_LIAB_KEYS) if balance_sheet is not None and not balance_sheet.empty and col_bs in balance_sheet.columns else None

            if te and shares and te > 0 and price:
                bps = te / shares
                pb = price / bps if bps else None
            if pb is None:
                pb = _safe_float(info.get("priceToBook"))

            roe = (ni / te) if ni is not None and te and te > 0 else _safe_float(info.get("returnOnEquity"))
            debt_equity = (total_liabilities / te) if total_liabilities is not None and te and te > 0 else _safe_float(info.get("debtToEquity"))
            current_ratio = (current_assets / current_liabilities) if current_assets is not None and current_liabilities and current_liabilities > 0 else _safe_float(info.get("currentRatio"))

            rev_prev, ni_prev = period_rev_ni.get(periods[idx + 1], (None, None)) if idx + 1 < len(periods) else (None, None)
            revenue_growth = (rev - rev_prev) / rev_prev if rev is not None and rev_prev is not None and rev_prev != 0 else None
            earnings_growth = (ni - ni_prev) / ni_prev if ni is not None and ni_prev is not None and ni_prev != 0 else None
            if revenue_growth is None:
                revenue_growth = _safe_float(info.get("revenueGrowth"))
            if earnings_growth is None:
                earnings_growth = _safe_float(info.get("earningsGrowth"))

            div_yield = _safe_float(info.get("dividendYield") or info.get("yield"))
            if div_yield is None and price and price > 0:
                div_rate = _safe_float(info.get("dividendRate"))
                if div_rate is not None and div_rate > 0:
                    div_yield = (div_rate / price) * 100  # already in percent
            if div_yield is not None and div_yield > 0:
                if div_yield < 0.05:
                    div_yield = div_yield * 100
                elif div_yield > 100:
                    div_yield = min(100.0, div_yield / 100.0)

            rows.append({
                "Symbol": ticker,
                "Name": name,
                "Sector": sector,
                "Fiscal_Year": fiscal_year,
                "Price": price,
                "Price/Earnings": pe,
                "Dividend Yield": div_yield,
                "Earnings/Share": eps,
                "52 Week Low": low_52,
                "52 Week High": high_52,
                "Market Cap": mcap,
                "EBITDA": ebitda,
                "Price/Sales": ps,
                "Price/Book": pb,
                "ROE": roe,
                "Debt/Equity": debt_equity,
                "Current Ratio": current_ratio,
                "Revenue Growth": revenue_growth,
                "Earnings Growth": earnings_growth,
                "SEC Filings": sec_url,
            })
    except Exception as e:
        logger.warning("Failed %s: %s", ticker, e)
    return rows


def run(
    tickers: Optional[List[str]] = None,
    output_path: Optional[Path] = None,
    backup: bool = True,
) -> Path:
    """
    Fetch 5 years of fundamentals for all tickers and write CSV.
    Respects rate limits (2s between tickers, internal delays).
    """
    output_path = output_path or OUTPUT_CSV
    # Prefer explicit tickers, then PPO/config tickers (so agent and fundamentals stay in sync), then 5y CSV
    tickers = tickers or get_ppo_tickers() or get_tickers_from_5y_csv()
    tickers = [t.strip().upper() for t in tickers if t and isinstance(t, str)]
    if not tickers:
        tickers = default_tickers()

    CSV_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Tickers: %s", tickers)
    logger.info("Rate limit: %.1fs between tickers, %.1fs between requests", SEC_BETWEEN_TICKERS, SEC_BETWEEN_REQUESTS_PER_TICKER)

    if backup and output_path.exists():
        backup_path = Path(str(output_path) + BACKUP_SUFFIX)
        try:
            pd.read_csv(output_path).to_csv(backup_path, index=False)
            logger.info("Backed up existing CSV to %s", backup_path)
        except Exception as e:
            logger.warning("Backup failed: %s", e)

    all_rows = []
    for i, ticker in enumerate(tickers):
        logger.info("[%s/%s] %s", i + 1, len(tickers), ticker)
        rows = fetch_one_ticker_5y(ticker)
        all_rows.extend(rows)
        if i < len(tickers) - 1:
            time.sleep(SEC_BETWEEN_TICKERS)

    if not all_rows:
        logger.warning("No rows collected; check rate limit or tickers.")
        return output_path

    df = pd.DataFrame(all_rows)
    df = df[[c for c in OUTPUT_COLUMNS if c in df.columns]]
    # Write to temp file then replace target (avoids PermissionError if CSV is open in Excel/etc.)
    output_path = Path(output_path)
    temp_path = output_path.parent / (output_path.name + ".tmp")
    df.to_csv(temp_path, index=False)
    written_path = output_path
    try:
        os.replace(temp_path, output_path)
    except OSError:
        if temp_path.exists():
            fallback = output_path.parent / (output_path.stem + "_new.csv")
            try:
                os.replace(temp_path, fallback)
            except OSError:
                temp_path.rename(fallback)
            written_path = fallback
            logger.warning(
                "Target file busy; wrote %s instead. collect_fundamentals.py and reports will use it automatically (newer file). Close the original and rename to replace.",
                fallback,
            )
        else:
            raise
    if temp_path.exists():
        try:
            temp_path.unlink()
        except OSError:
            pass
    logger.info("Wrote %s rows to %s", len(df), written_path)
    return output_path


if __name__ == "__main__":
    run(backup=True)

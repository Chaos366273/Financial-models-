"""Fetch, cache, and build a monthly ML panel from Yahoo Finance + Kenneth French FF5.

Coverage vs dissertation IV groups
----------------------------------
IV1  FF5 factors (MKT_RF, SMB, HML, RMW, CMA)     Kenneth French (cached once)
IV2  Fundamentals (P/E, P/B, ROE, D/E, growth…)    yfinance yearly statements + FY-end price
IV3  Technicals (OHLCV, RSI, MACD, BB, ATR…)      computed locally from daily Yahoo OHLCV
IV4  Bloomberg BSI / news sentiment                 NOT on Yahoo — uses sentiment_proxy only
IV5  Lagged returns & volatility                    computed locally
     Controls (mcap, sector, Amihud, US rates, FX,  partial via Yahoo macro tickers
               commodities)

UK macro (GDP, CPI, unemployment, Bank Rate) is NOT available on Yahoo Finance;
add saved_data/yahoo/uk_macro.csv later if you source ONS / BoE data.

Rate limiting
-------------
The Yahoo DSP Traffic API limits apply to Yahoo Ad Manager, not yfinance.
This module throttles requests conservatively (default >=2 s spacing, <=25 calls/min).

Cache layout (reused by lstm_return.py):
  saved_data/yahoo/daily_features.csv
  saved_data/yahoo/fundamentals.csv
  saved_data/yahoo/ff5_monthly.csv
  saved_data/yahoo/macro_monthly.csv
  saved_data/yahoo/panel_monthly.csv
  saved_data/yahoo/fetch_metadata.json
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any
from zipfile import ZipFile

import numpy as np
import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parent
CACHE_DIR = ROOT / "saved_data" / "yahoo"

FF5_ZIP_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    "F-F_Research_Data_5_Factors_2x3_CSV.zip"
)

MACRO_TICKERS = {
    "^TNX": "us_10y_yield",
    "CL=F": "oil_price",
    "GC=F": "gold_price",
    "SI=F": "silver_price",
    "GBPUSD=X": "gbpusd",
}

DEFAULT_TICKERS = [
    "AAPL", "ABBV", "AMD", "AMZN", "BAC", "COST", "CRM", "CVX",
    "DIS", "GOOGL", "HD", "INTC", "JNJ", "JPM", "KO", "LLY", "META", "MSFT",
]
# IPO / history notes (Yahoo may have shorter daily history than Bloomberg):
#   GOOGL from ~2004-08, CRM ~2004-06, ABBV ~2012-12; rest ~2000+
DEFAULT_START = "2000-01-01"
DEFAULT_END = "2025-12-31"
FUNDAMENTALS_LAG_MONTHS = 6
RISK_FREE_FALLBACK = 0.002 / 12

TECH_COLS = [
    "rsi", "macd", "macd_signal", "atr", "bb_upper", "bb_middle", "bb_lower",
    "price_to_sma_20", "volume_trend", "sma_20", "ema_12", "volatility_30d",
]

FUND_COLS = [
    "pe_ratio", "pb_ratio", "roe", "debt_equity", "revenue_growth", "earnings_growth",
    "profit_margin", "current_ratio", "log_market_cap",
]

FACTOR_COLS = ["MKT_RF", "SMB", "HML", "RMW", "CMA"]
LAG_COLS = ["lagged_return_1m", "lagged_return_3m", "lagged_vol_30d"]
MACRO_COLS = ["us_10y_yield_chg", "oil_ret", "gold_ret", "silver_ret", "gbpusd_ret"]
CONTROL_COLS = ["amihud_illiquidity"]

YAHOO_FEATURE_COLS = TECH_COLS + FUND_COLS + FACTOR_COLS + LAG_COLS + MACRO_COLS + CONTROL_COLS


@dataclass
class RateLimiter:
    min_interval_sec: float = 2.0
    max_per_minute: int = 25
    _last_call: float = field(default=0.0, init=False)
    _window: list[float] = field(default_factory=list, init=False)

    def wait(self, label: str = "") -> None:
        now = time.monotonic()
        elapsed = now - self._last_call
        if elapsed < self.min_interval_sec:
            time.sleep(self.min_interval_sec - elapsed)
            now = time.monotonic()

        self._window = [t for t in self._window if now - t < 60.0]
        if len(self._window) >= self.max_per_minute:
            sleep_for = 60.0 - (now - self._window[0]) + 0.1
            if sleep_for > 0:
                time.sleep(sleep_for)
            now = time.monotonic()
            self._window = [t for t in self._window if now - t < 60.0]

        self._window.append(now)
        self._last_call = now
        if label:
            print(f"  [api] {label}")


def _flatten_yf_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.get_level_values(0)
    return df


def compute_indicators(frame: pd.DataFrame) -> pd.DataFrame:
    g = frame.sort_values("date").copy()
    close = g["close"].astype(float)
    high = g["high"].astype(float)
    low = g["low"].astype(float)
    volume = g["volume"].astype(float)

    g["sma_20"] = close.rolling(20, min_periods=1).mean()
    g["ema_12"] = close.ewm(span=12, adjust=False).mean()

    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14, min_periods=1).mean()
    loss = (-delta.clip(upper=0)).rolling(14, min_periods=1).mean()
    rs = gain / (loss + 1e-9)
    g["rsi"] = 100.0 - (100.0 / (1.0 + rs))

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    g["macd"] = ema12 - ema26
    g["macd_signal"] = g["macd"].ewm(span=9, adjust=False).mean()

    tr = pd.concat(
        [high - low, (high - close.shift()).abs(), (low - close.shift()).abs()],
        axis=1,
    ).max(axis=1)
    g["atr"] = tr.rolling(14, min_periods=1).mean()

    g["bb_middle"] = g["sma_20"]
    std = close.rolling(20, min_periods=1).std().fillna(0.0)
    g["bb_upper"] = g["bb_middle"] + 2.0 * std
    g["bb_lower"] = g["bb_middle"] - 2.0 * std
    g["price_to_sma_20"] = close / (g["sma_20"] + 1e-9)
    g["volume_trend"] = volume.rolling(2, min_periods=1).mean()
    g["volatility_30d"] = close.pct_change().rolling(30, min_periods=5).std()
    g["daily_return"] = close.pct_change()
    dollar_vol = (close * volume).replace(0, np.nan)
    g["amihud_daily"] = g["daily_return"].abs() / dollar_vol
    return g


def fetch_daily_ohlcv(
    tickers: list[str],
    start: str,
    end: str,
    limiter: RateLimiter,
    refresh: bool = False,
) -> pd.DataFrame:
    out_path = CACHE_DIR / "daily_features.csv"
    meta = _load_metadata()
    if out_path.is_file() and not refresh and _cache_ok(meta, tickers, start, end):
        print(f"Using cached daily features: {out_path}")
        return pd.read_csv(out_path, parse_dates=["date"])

    limiter.wait(f"daily OHLCV batch ({', '.join(tickers)})")
    raw = yf.download(
        tickers,
        start=start,
        end=end,
        interval="1d",
        auto_adjust=True,
        progress=False,
        group_by="column",
        threads=False,
    )
    if raw.empty:
        raise ValueError(f"No daily data returned for {tickers}")

    frames: list[pd.DataFrame] = []
    if len(tickers) == 1:
        px = _flatten_yf_columns(raw).reset_index()
        px = px.rename(columns={
            "Date": "date", "Open": "open", "High": "high", "Low": "low",
            "Close": "close", "Volume": "volume",
        })
        px["Ticker"] = tickers[0]
        frames.append(compute_indicators(px[["date", "Ticker", "open", "high", "low", "close", "volume"]]))
    else:
        for t in tickers:
            if t not in raw.columns.get_level_values(1):
                print(f"  warning: no data for {t}")
                continue
            sub = raw.xs(t, axis=1, level=1).reset_index()
            sub = sub.rename(columns={
                "Date": "date", "Open": "open", "High": "high", "Low": "low",
                "Close": "close", "Volume": "volume",
            })
            sub["Ticker"] = t
            frames.append(compute_indicators(sub))

    daily = pd.concat(frames, ignore_index=True)
    daily["date"] = pd.to_datetime(daily["date"]).dt.tz_localize(None)
    daily = daily.sort_values(["Ticker", "date"])
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    daily.to_csv(out_path, index=False)
    return daily


def fetch_ff5_monthly(start: str, end: str, refresh: bool = False) -> pd.DataFrame:
    out_path = CACHE_DIR / "ff5_monthly.csv"
    if out_path.is_file() and not refresh:
        print(f"Using cached FF5 factors: {out_path}")
        ff = pd.read_csv(out_path, parse_dates=["month"])
        mask = (ff["month"] >= pd.Timestamp(start)) & (ff["month"] <= pd.Timestamp(end))
        return ff.loc[mask].copy()

    import urllib.request

    print("  [api] Kenneth French FF5 monthly zip (one download)")
    with urllib.request.urlopen(FF5_ZIP_URL) as r:
        zbytes = r.read()
    with ZipFile(BytesIO(zbytes)) as zf:
        csv_name = next(n for n in zf.namelist() if n.lower().endswith(".csv"))
        raw = zf.read(csv_name).decode("utf-8", errors="replace")

    rows = []
    for line in raw.splitlines():
        parts = [p.strip() for p in line.split(",") if p.strip()]
        if len(parts) < 7 or not parts[0].isdigit() or len(parts[0]) != 6:
            continue
        ym = int(parts[0])
        y, m = ym // 100, ym % 100
        rows.append({
            "month": pd.Timestamp(year=y, month=m, day=1),
            "MKT_RF": float(parts[1]) / 100.0,
            "SMB": float(parts[2]) / 100.0,
            "HML": float(parts[3]) / 100.0,
            "RMW": float(parts[4]) / 100.0,
            "CMA": float(parts[5]) / 100.0,
            "RF": float(parts[6]) / 100.0,
        })

    ff = pd.DataFrame(rows).sort_values("month")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    ff.to_csv(out_path, index=False)
    mask = (ff["month"] >= pd.Timestamp(start)) & (ff["month"] <= pd.Timestamp(end))
    return ff.loc[mask].copy()


def fetch_macro_monthly(start: str, end: str, limiter: RateLimiter, refresh: bool = False) -> pd.DataFrame:
    out_path = CACHE_DIR / "macro_monthly.csv"
    if out_path.is_file() and not refresh:
        print(f"Using cached macro series: {out_path}")
        return pd.read_csv(out_path, parse_dates=["month"])

    syms = list(MACRO_TICKERS.keys())
    limiter.wait(f"macro batch ({', '.join(syms)})")
    raw = yf.download(
        syms,
        start=start,
        end=end,
        interval="1mo",
        auto_adjust=True,
        progress=False,
        group_by="column",
        threads=False,
    )
    if raw.empty:
        raise ValueError("No macro data returned from Yahoo")

    pieces: list[pd.DataFrame] = []
    for sym, col in MACRO_TICKERS.items():
        if len(syms) == 1:
            s = _flatten_yf_columns(raw)["Close"]
        else:
            if sym not in raw.columns.get_level_values(1):
                continue
            s = raw.xs(sym, axis=1, level=1)["Close"]
        df = s.rename(col).reset_index()
        df = df.rename(columns={"Date": "month"})
        df["month"] = pd.to_datetime(df["month"]).dt.to_period("M").dt.to_timestamp()
        pieces.append(df)

    macro = pieces[0]
    for p in pieces[1:]:
        macro = macro.merge(p, on="month", how="outer")
    macro = macro.sort_values("month")

    if "oil_price" in macro.columns:
        macro["oil_ret"] = macro["oil_price"].pct_change(fill_method=None)
    if "gold_price" in macro.columns:
        macro["gold_ret"] = macro["gold_price"].pct_change(fill_method=None)
    if "silver_price" in macro.columns:
        macro["silver_ret"] = macro["silver_price"].pct_change(fill_method=None)
    if "gbpusd" in macro.columns:
        macro["gbpusd_ret"] = macro["gbpusd"].pct_change(fill_method=None)
    if "us_10y_yield" in macro.columns:
        macro["us_10y_yield_chg"] = macro["us_10y_yield"].diff()

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    macro.to_csv(out_path, index=False)
    return macro


def _year_end_prices(daily: pd.DataFrame) -> pd.DataFrame:
    d = daily.copy()
    d["year"] = d["date"].dt.year
    return d.sort_values("date").groupby(["Ticker", "year"], as_index=False).agg(close=("close", "last"))


def fetch_fundamentals(
    tickers: list[str],
    daily: pd.DataFrame,
    limiter: RateLimiter,
    refresh: bool = False,
) -> pd.DataFrame:
    out_path = CACHE_DIR / "fundamentals.csv"
    meta = _load_metadata()
    if out_path.is_file() and not refresh and _cache_ok(meta, tickers, meta.get("start", ""), meta.get("end", "")):
        cached = pd.read_csv(out_path)
        if set(tickers).issubset(set(cached["Ticker"].unique())):
            print(f"Using cached fundamentals: {out_path}")
            return cached

    yprices = _year_end_prices(daily)
    rows: list[dict[str, Any]] = []

    for t in tickers:
        limiter.wait(f"{t} income statement")
        tk = yf.Ticker(t)
        inc = tk.get_income_stmt(freq="yearly")
        limiter.wait(f"{t} balance sheet")
        bal = tk.get_balance_sheet(freq="yearly")
        limiter.wait(f"{t} info")
        info = tk.info or {}
        sector = info.get("sector") or info.get("industry") or "Unknown"
        shares = info.get("sharesOutstanding")

        if inc is None or inc.empty:
            print(f"  warning: no income statement for {t}")
            continue

        def _row(metric: str, frame: pd.DataFrame) -> dict[int, float]:
            if frame is None or frame.empty or metric not in frame.index:
                return {}
            s = frame.loc[metric]
            out: dict[int, float] = {}
            for col in s.index:
                ts = pd.Timestamp(col)
                if pd.notna(s[col]):
                    out[ts.year] = float(s[col])
            return out

        rev = _row("TotalRevenue", inc)
        ni = _row("NetIncome", inc)
        eq = _row("StockholdersEquity", bal)
        debt = _row("TotalDebt", bal)
        cur_assets = _row("CurrentAssets", bal)
        cur_liab = _row("CurrentLiabilities", bal)

        years = sorted(set(rev.keys()) | set(ni.keys()))
        prev_rev = prev_ni = None
        for y in years:
            price_row = yprices[(yprices["Ticker"] == t) & (yprices["year"] == y)]
            price = float(price_row["close"].iloc[0]) if not price_row.empty else np.nan
            equity = eq.get(y, np.nan)
            net_inc = ni.get(y, np.nan)
            revenue = rev.get(y, np.nan)
            total_debt = debt.get(y, np.nan)
            roe = net_inc / equity if equity and np.isfinite(equity) and equity != 0 else np.nan
            mcap = price * shares if price and shares else info.get("marketCap")
            pe = price / (net_inc / shares) if net_inc and shares and net_inc > 0 else np.nan
            pb = mcap / equity if mcap and equity and equity > 0 else np.nan
            de = total_debt / equity if equity and total_debt and equity != 0 else np.nan
            rev_g = (revenue - prev_rev) / abs(prev_rev) if prev_rev and revenue else np.nan
            earn_g = (net_inc - prev_ni) / abs(prev_ni) if prev_ni and net_inc else np.nan
            pm = net_inc / revenue if revenue and net_inc else np.nan
            cl = cur_liab.get(y)
            cr = cur_assets.get(y, np.nan) / cl if cl else np.nan

            rows.append({
                "Symbol": t,
                "Ticker": t,
                "Sector": sector,
                "Fiscal_Year": y,
                "Price": price,
                "Price/Earnings": pe,
                "Price/Book": pb,
                "ROE": roe,
                "Debt/Equity": de,
                "Revenue Growth": rev_g,
                "Earnings Growth": earn_g,
                "Market Cap": mcap,
                "Profit Margin": pm,
                "Current Ratio": cr,
            })
            prev_rev, prev_ni = revenue, net_inc

    fund = pd.DataFrame(rows)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    fund.to_csv(out_path, index=False)
    return fund


def _merge_fundamentals_monthly(monthly: pd.DataFrame, fund: pd.DataFrame, lag_months: int) -> pd.DataFrame:
    if fund.empty or "Fiscal_Year" not in fund.columns:
        out = monthly.copy()
        for c in FUND_COLS:
            out[c] = np.nan
        return out

    f = fund.rename(columns={
        "Price/Earnings": "pe_ratio",
        "Price/Book": "pb_ratio",
        "ROE": "roe",
        "Debt/Equity": "debt_equity",
        "Revenue Growth": "revenue_growth",
        "Earnings Growth": "earnings_growth",
        "Market Cap": "market_cap",
        "Profit Margin": "profit_margin",
        "Current Ratio": "current_ratio",
    }).copy()
    f["fiscal_year"] = f["Fiscal_Year"].astype(int)
    cols = [
        "pe_ratio", "pb_ratio", "roe", "debt_equity", "revenue_growth", "earnings_growth",
        "market_cap", "profit_margin", "current_ratio",
    ]
    cols = [c for c in cols if c in f.columns]

    f["available_from"] = pd.to_datetime((f["fiscal_year"] + 1).astype(str) + "-01-01") + pd.DateOffset(months=lag_months)
    m = monthly.copy()
    merged = m.merge(f[["Ticker", "fiscal_year", "available_from"] + cols], on="Ticker", how="left")
    usable = merged["month"] >= merged["available_from"]
    merged = merged.loc[usable].sort_values(["Ticker", "month", "fiscal_year"])
    merged = merged.groupby(["Ticker", "month"], as_index=False).last()
    out = m.merge(merged[["Ticker", "month"] + cols], on=["Ticker", "month"], how="left")
    for c in cols:
        if c in out.columns:
            out[c] = out.groupby("Ticker")[c].ffill()
    if "market_cap" in out.columns:
        out["log_market_cap"] = np.log1p(out["market_cap"].clip(lower=0))
    return out


def _to_monthly(daily: pd.DataFrame, ff5: pd.DataFrame) -> pd.DataFrame:
    d = daily.copy()
    d["month"] = d["date"].dt.to_period("M").dt.to_timestamp()
    agg_cols = {c: (c, "last") for c in TECH_COLS if c in d.columns}
    monthly = (
        d.sort_values("date")
        .groupby(["Ticker", "month"], as_index=False)
        .agg(close=("close", "last"), volume=("volume", "sum"), amihud_illiquidity=("amihud_daily", "mean"), **agg_cols)
    )
    monthly["monthly_return"] = monthly.groupby("Ticker")["close"].pct_change(fill_method=None)
    rf_map = ff5.set_index("month")["RF"]
    monthly["RF"] = monthly["month"].map(rf_map).fillna(RISK_FREE_FALLBACK)
    monthly["target_excess_return"] = monthly.groupby("Ticker")["monthly_return"].shift(-1) - monthly["RF"]
    monthly["lagged_return_1m"] = monthly.groupby("Ticker")["monthly_return"].shift(1)
    monthly["lagged_return_3m"] = monthly.groupby("Ticker")["monthly_return"].shift(3)
    monthly["lagged_vol_30d"] = monthly.groupby("Ticker")["volatility_30d"].shift(1)
    monthly["year"] = monthly["month"].dt.year
    return monthly


def _add_sector_dummies(monthly: pd.DataFrame, fund: pd.DataFrame) -> pd.DataFrame:
    sectors = fund.groupby("Ticker")["Sector"].first().to_dict() if "Sector" in fund.columns else {}
    out = monthly.copy()
    out["sector"] = out["Ticker"].map(sectors).fillna("Unknown")
    dummies = pd.get_dummies(out["sector"], prefix="sector", dtype=float)
    return pd.concat([out, dummies], axis=1)


def _fundamentals_complete(tickers: list[str]) -> bool:
    path = CACHE_DIR / "fundamentals.csv"
    if not path.is_file():
        return False
    cached = pd.read_csv(path)
    if "Ticker" not in cached.columns:
        return False
    return set(tickers).issubset(set(cached["Ticker"].astype(str).str.upper()))


def build_yahoo_panel(
    tickers: list[str] | None = None,
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
    refresh: bool = False,
) -> pd.DataFrame:
    tickers = [t.upper() for t in (tickers or DEFAULT_TICKERS)]
    panel_path = CACHE_DIR / "panel_monthly.csv"
    meta = _load_metadata()

    if (
        panel_path.is_file()
        and not refresh
        and _cache_ok(meta, tickers, start, end)
        and _fundamentals_complete(tickers)
    ):
        print(f"Using cached monthly panel: {panel_path}")
        panel = pd.read_csv(panel_path, parse_dates=["month"])
        return panel[panel["Ticker"].isin(tickers)].copy()

    limiter = RateLimiter()
    print(f"Fetching Yahoo panel for {tickers} | {start} -> {end}")
    daily = fetch_daily_ohlcv(tickers, start, end, limiter, refresh=refresh)
    ff5 = fetch_ff5_monthly(start, end, refresh=refresh)
    macro = fetch_macro_monthly(start, end, limiter, refresh=refresh)
    fund = fetch_fundamentals(tickers, daily, limiter, refresh=refresh)

    monthly = _to_monthly(daily, ff5)
    monthly = monthly.merge(ff5.drop(columns=["RF"], errors="ignore"), on="month", how="left")
    monthly = _merge_fundamentals_monthly(monthly, fund, FUNDAMENTALS_LAG_MONTHS)
    monthly = monthly.merge(macro, on="month", how="left")
    monthly = _add_sector_dummies(monthly, fund)

    if "volume_trend" in monthly.columns and "monthly_return" in monthly.columns:
        z = monthly.groupby("Ticker")["volume_trend"].transform(lambda s: (s - s.mean()) / (s.std() + 1e-9))
        monthly["sentiment_proxy"] = z * np.sign(monthly["monthly_return"].fillna(0))

    monthly = monthly.sort_values(["Ticker", "month"])
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    monthly.to_csv(panel_path, index=False)
    _save_metadata(tickers, start, end, limiter)
    print(f"Saved panel -> {panel_path} ({len(monthly)} rows)")
    return monthly


def yahoo_feature_columns(include_alt: bool = False, panel: pd.DataFrame | None = None) -> list[str]:
    cols = list(YAHOO_FEATURE_COLS)
    if panel is not None:
        sector_cols = [c for c in panel.columns if c.startswith("sector_")]
        cols = [c for c in cols if c in panel.columns] + sector_cols
    if include_alt:
        cols = cols + ["sentiment_proxy"]
    return list(dict.fromkeys(cols))


def split_yahoo_panel(
    panel: pd.DataFrame,
    train_end: str = "2020-12-31",
    test_start: str = "2021-01-01",
    test_end: str = "2025-12-31",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    m = pd.to_datetime(panel["month"])
    train = panel.loc[m <= pd.Timestamp(train_end)].copy()
    test = panel.loc[(m >= pd.Timestamp(test_start)) & (m <= pd.Timestamp(test_end))].copy()
    return train, test


def build_sequences(
    df: pd.DataFrame,
    feature_cols: list[str],
    seq_len: int,
    target_col: str = "target_excess_return",
) -> tuple[np.ndarray | None, np.ndarray | None, list[str], list[tuple[str, str]]]:
    cols = [c for c in feature_cols if c in df.columns]
    parts_x, parts_y, parts_key = [], [], []
    for ticker, g in df.sort_values("month").groupby("Ticker"):
        g = g.dropna(subset=[target_col])
        if len(g) <= seq_len or not cols:
            continue
        X = g[cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).values.astype(np.float32)
        y = g[target_col].values.astype(np.float32)
        months = pd.to_datetime(g["month"]).dt.strftime("%Y-%m-%d").values
        for i in range(seq_len, len(g)):
            parts_x.append(X[i - seq_len : i])
            parts_y.append(y[i])
            parts_key.append((str(ticker), months[i]))
    if not parts_x:
        return None, None, cols, []
    return np.stack(parts_x), np.array(parts_y), cols, parts_key


def _load_metadata() -> dict:
    path = CACHE_DIR / "fetch_metadata.json"
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def _save_metadata(tickers: list[str], start: str, end: str, limiter: RateLimiter) -> None:
    meta = {
        "tickers": sorted(tickers),
        "start": start,
        "end": end,
        "fetched_at": pd.Timestamp.utcnow().isoformat(),
        "api_calls_last_run": len(limiter._window),
        "note": "IV4 Bloomberg BSI not on Yahoo; sentiment_proxy used instead.",
        "uk_macro": "not_available_on_yahoo",
    }
    (CACHE_DIR / "fetch_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def _cache_ok(meta: dict, tickers: list[str], start: str, end: str) -> bool:
    if not meta:
        return False
    return (
        sorted(meta.get("tickers", [])) == sorted(tickers)
        and meta.get("start") == start
        and meta.get("end") == end
    )

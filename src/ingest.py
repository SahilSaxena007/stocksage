# src/ingest.py
import pandas as pd, numpy as np
from datetime import datetime
from pathlib import Path

def _try_download(ticker: str, start: str):
    """Fetch data from Yahoo Finance."""
    try:
        import yfinance as yf
        df = yf.download(ticker, start=start, auto_adjust=True, progress=False)
        if df is None or df.empty: return None
        df = df.rename(columns=str.lower).reset_index()
        df["ticker"] = ticker
        return df
    except Exception:
        return None

def _make_synthetic(ticker: str, start="2015-01-01", n_days=1500, seed=7):
    """Offline fallback (generates GBM-like synthetic series)."""
    rng = np.random.default_rng(seed + sum(ord(c) for c in ticker))
    dates = pd.bdate_range(start=start, periods=n_days)
    mu, sigma = 0.0003, 0.02
    rets = rng.normal(mu, sigma, len(dates))
    price = 100 * (1 + pd.Series(rets, index=dates)).cumprod()
    df = pd.DataFrame({
        "date": dates,
        "open": price.shift(1, fill_value=price.iloc[0]),
        "high": price * (1 + rng.normal(0.001, 0.005, len(dates)).clip(-0.02, 0.02)),
        "low":  price * (1 + rng.normal(-0.001, 0.005, len(dates)).clip(-0.02, 0.02)),
        "close": price,
        "volume": rng.integers(5e5, 5e6, len(dates)),
        "ticker": ticker
    })
    return df

def load_prices(ticker: str, start="2012-01-01", cache_dir="data/raw", allow_synthetic=True):
    cache = Path(cache_dir) / f"{ticker}.csv"
    if cache.exists():
        return pd.read_csv(cache, parse_dates=["date"])
    df = _try_download(ticker, start)
    if df is None and allow_synthetic:
        df = _make_synthetic(ticker, start)
    elif df is None:
        raise RuntimeError(f"No data for {ticker}")
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache, index=False)
    return df

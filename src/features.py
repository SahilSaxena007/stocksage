import pandas as pd, numpy as np
from ta.momentum import RSIIndicator
from ta.trend import MACD, SMAIndicator
from ta.volatility import AverageTrueRange

def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("date").reset_index(drop=True)
    df["ret1"] = df["close"].pct_change()
    for w in (5,10,20):
        df[f"ret{w}"] = df["close"].pct_change(w)
        df[f"vol{w}"] = df["ret1"].rolling(w).std()
    for w in (10,20,50,200):
        df[f"sma{w}"] = SMAIndicator(df["close"], w).sma_indicator()
        df[f"sma{w}_ratio"] = df["close"] / df[f"sma{w}"]
    df["rsi14"] = RSIIndicator(df["close"], 14).rsi()
    df["macd"] = MACD(df["close"]).macd()
    df["atr14"] = AverageTrueRange(df["high"], df["low"], df["close"], 14).average_true_range()
    return df.dropna().reset_index(drop=True)

def build_feature_matrix(stock_df: pd.DataFrame, spy_df: pd.DataFrame) -> pd.DataFrame:
    """
    Combine stock and S&P500 features into one aligned matrix.
    """
    s = add_technical_features(stock_df)
    m = add_technical_features(spy_df)
    full = pd.merge(
        s,
        m[["date", "close", "ret1"]],
        on="date",
        how="inner",
        suffixes=("", "_spy")
    )
    full["cov60"]  = full["ret1"].rolling(60).cov(full["ret1_spy"])
    full["var60"]  = full["ret1_spy"].rolling(60).var()
    full["beta60"] = full["cov60"] / full["var60"]
    return full.dropna().reset_index(drop=True)

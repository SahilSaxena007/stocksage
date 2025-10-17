import os
import json
import pickle
import numpy as np
import pandas as pd
import yfinance as yf
from ta.momentum import RSIIndicator
from ta.trend import MACD, SMAIndicator
from ta.volatility import AverageTrueRange
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

TICKERS = [
    "AAPL","MSFT","AMZN","GOOGL","META","NVDA",
    "JPM","GS","MS","BAC","C","WFC",
    "XOM","CVX","KO","MCD","NKE","UNH","BABA","TSLA"
]

def fetch_yf(ticker):
    df = yf.download(ticker, period="2y", interval="1d", progress=False, auto_adjust=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join(str(c) for c in tup if c).strip().lower() for tup in df.columns.values]
    else:
        df.columns = [str(c).lower() for c in df.columns]
    df = df.reset_index()
    df.rename(columns={df.columns[0]: "date"}, inplace=True)
    for col in ["open", "high", "low", "close", "volume"]:
        matches = [c for c in df.columns if col in c]
        if matches:
            df[col] = df[matches[0]]
    return df[["date","open","high","low","close","volume"]].dropna()

def engineer(df):
    df["ret1"] = df["close"].pct_change()
    for w in (5,10,20):
        df[f"ret{w}"] = df["close"].pct_change(w)
        df[f"vol{w}"] = df["ret1"].rolling(w).std()
    for w in (10,20,50,200):
        sma = SMAIndicator(df["close"], w).sma_indicator()
        df[f"sma{w}"] = sma
        df[f"sma{w}_ratio"] = df["close"]/sma
    df["rsi14"] = RSIIndicator(df["close"],14).rsi()
    df["macd"] = MACD(df["close"]).macd()
    df["atr14"] = AverageTrueRange(df["high"],df["low"],df["close"],14).average_true_range()
    df["ret21"] = df["close"].pct_change(21)
    df = df.dropna().reset_index(drop=True)
    return df

def train_one(ticker):
    print(f"Training {ticker}...")
    df = engineer(fetch_yf(ticker))
    df["target_cls"] = (df["ret21"] > 0).astype(int)
    X = df.drop(columns=["date","ret21","target_cls"])
    y_cls = df["target_cls"]
    y_reg = df["ret21"]

    pipe_cls = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000))
    ])
    pipe_reg = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LinearRegression())
    ])

    pipe_cls.fit(X, y_cls)
    pipe_reg.fit(X, y_reg)

    # ✅ artifacts directory is now outside src
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    artifacts_dir = os.path.join(base_dir, "artifacts", "models", ticker)
    os.makedirs(artifacts_dir, exist_ok=True)

    pickle.dump(pipe_cls, open(os.path.join(artifacts_dir, "cls.pkl"), "wb"))
    pickle.dump(pipe_reg, open(os.path.join(artifacts_dir, "reg.pkl"), "wb"))
    json.dump(list(X.columns), open(os.path.join(artifacts_dir, "features.json"), "w"))

for t in TICKERS:
    try:
        train_one(t)
    except Exception as e:
        print(f"❌ {t}: {e}")

import streamlit as st
import pandas as pd
import numpy as np
import json, os, pickle
from datetime import datetime, timedelta
import yfinance as yf
from ta.momentum import RSIIndicator
from ta.trend import MACD, SMAIndicator
from ta.volatility import AverageTrueRange

# -------------------------------------------------------------------
#   PAGE CONFIG
# -------------------------------------------------------------------
st.set_page_config(
    page_title="StockSage Dashboard",
    page_icon="📊",
    layout="wide"
)
st.markdown(
    "<h1 style='text-align:center;color:#0a66c2;'>📊 StockSage</h1>",
    unsafe_allow_html=True
)

# -------------------------------------------------------------------
#   COMPANY SELECTION
# -------------------------------------------------------------------
available_models = sorted([
    d for d in os.listdir("artifacts/models")
    if os.path.isdir(f"artifacts/models/{d}")
])
if not available_models:
    st.error("⚠️ No trained models found under artifacts/models/")
    st.stop()

ticker = st.sidebar.selectbox("📈 Select Company / Ticker", available_models, index=0)
model_dir = f"artifacts/models/{ticker}"

# -------------------------------------------------------------------
#   LOAD MODELS
# -------------------------------------------------------------------
try:
    cls = pickle.load(open(f"{model_dir}/cls.pkl", "rb"))
    reg = pickle.load(open(f"{model_dir}/reg.pkl", "rb"))
    FEATURES = json.load(open(f"{model_dir}/features.json"))
except Exception as e:
    st.error(f"Could not load model artifacts for {ticker}: {e}")
    st.stop()

metrics = {}
if os.path.exists(f"{model_dir}/metrics.json"):
    try:
        metrics = json.load(open(f"{model_dir}/metrics.json"))
    except:
        pass

# -------------------------------------------------------------------
#   YFINANCE FETCH
# -------------------------------------------------------------------
@st.cache_data(ttl=3600)
def get_data(symbol):
    df = yf.download(symbol, period="2y", interval="1d", auto_adjust=False, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join(str(c) for c in tup if c).strip().lower() for tup in df.columns.values]
    else:
        df.columns = [str(c).lower() for c in df.columns]
    df = df.reset_index()
    df.rename(columns={df.columns[0]:"date"}, inplace=True)
    for c in ["open","high","low","close","volume"]:
        matches = [x for x in df.columns if c in x]
        if matches:
            df[c] = df[matches[0]]
    return df[["date","open","high","low","close","volume"]].dropna()

try:
    df_stock = get_data(ticker)
    df_spy = get_data("SPY")
except Exception as e:
    st.error(f"❌ Data fetch failed: {e}")
    st.stop()

if df_stock.empty or df_spy.empty:
    st.error("⚠️ No data retrieved. Check ticker symbol or internet connection.")
    st.stop()

# -------------------------------------------------------------------
#   FEATURE ENGINEERING
# -------------------------------------------------------------------
def compute_features(df_stock, df_spy):
    df_stock = df_stock.copy(); df_spy = df_spy.copy()
    df_stock["ret1"] = df_stock["close"].pct_change()
    for w in (5,10,20):
        df_stock[f"ret{w}"] = df_stock["close"].pct_change(w)
        df_stock[f"vol{w}"] = df_stock["ret1"].rolling(w).std()
    for w in (10,20,50,200):
        sma = SMAIndicator(df_stock["close"], w).sma_indicator()
        df_stock[f"sma{w}"] = sma
        df_stock[f"sma{w}_ratio"] = df_stock["close"]/sma
    df_stock["rsi14"] = RSIIndicator(df_stock["close"],14).rsi()
    df_stock["macd"] = MACD(df_stock["close"]).macd()
    df_stock["atr14"] = AverageTrueRange(
        df_stock["high"],df_stock["low"],df_stock["close"],14
    ).average_true_range()
    df_stock = df_stock.dropna().reset_index(drop=True)
    df_spy["ret1"] = df_spy["close"].pct_change()
    merged = pd.merge(df_stock, df_spy[["date","close","ret1"]],
                      on="date", how="inner", suffixes=("", "_spy"))
    merged["cov60"] = merged["ret1"].rolling(60).cov(merged["ret1_spy"])
    merged["var60"] = merged["ret1_spy"].rolling(60).var()
    merged["beta60"] = merged["cov60"]/merged["var60"]
    merged = merged.dropna().reset_index(drop=True)
    return merged

df_feat = compute_features(df_stock, df_spy)

if df_feat.empty:
    st.error("⚠️ Not enough data to compute features. Try increasing period or reducing window sizes.")
    st.stop()

latest = df_feat.iloc[-1]
X = pd.DataFrame([latest])[FEATURES]

# -------------------------------------------------------------------
#   PREDICTION
# -------------------------------------------------------------------
p_up = float(cls.predict_proba(X)[0,1])
mag = float(reg.predict(X)[0])
exp_move = (1 if p_up>=0.5 else -1)*mag

# -------------------------------------------------------------------
#   DASHBOARD HEADLINE
# -------------------------------------------------------------------
col1, col2, col3, col4 = st.columns(4)
col1.metric("Ticker", ticker)
col2.metric("Direction", "📈 UP" if p_up>=0.5 else "📉 DOWN")
col3.metric("Confidence", f"{p_up*100:.1f}%")
col4.metric("Expected 21-Day Move", f"{exp_move*100:.2f}%")

if metrics:
    acc = metrics.get("accuracy")
    auc = metrics.get("auc")
    if acc is not None and auc is not None:
        st.markdown(
            f"<p style='text-align:center;color:gray;'>Model Accuracy: {acc:.2f} &nbsp; • &nbsp; AUC: {auc:.2f}</p>",
            unsafe_allow_html=True
        )

# -------------------------------------------------------------------
#   INTERACTIVE CHART
# -------------------------------------------------------------------
st.markdown("---")
st.subheader(f"📊 {ticker} — Interactive Price Trend with SMA20 & SMA50")

# Add SMA columns for chart
df_stock["SMA20"] = SMAIndicator(df_stock["close"], 20).sma_indicator()
df_stock["SMA50"] = SMAIndicator(df_stock["close"], 50).sma_indicator()

# Rename for chart clarity
df_chart = df_stock[["date", "close", "SMA20", "SMA50"]].set_index("date")

st.line_chart(df_chart, height=400, width='stretch')


# -------------------------------------------------------------------
#   DETAILS
# -------------------------------------------------------------------
with st.expander("🔍 Show Computed Feature Snapshot"):
    st.dataframe(pd.DataFrame(latest).T)

st.caption("📘 Research demonstration — not financial advice. © 2025 StockSage")

import pandas as pd, numpy as np, os, json, argparse, pickle
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score, roc_auc_score, mean_absolute_error

# local imports
from ingest import load_prices
from features import build_feature_matrix
from labeling import make_labels


def train_one(ticker="AAPL", start="2012-01-01", benchmark="SPY", horizon=21):
    """Full training pipeline for one ticker."""
    print(f"\n▶️ Training {ticker} vs {benchmark} horizon={horizon} days")

    # --- 1. Load data ---
    stock_df = load_prices(ticker, start)
    spy_df   = load_prices(benchmark, start)

    # --- 2. Build features & labels ---
    df = build_feature_matrix(stock_df, spy_df)
    df = make_labels(df, horizon_days=horizon)

    FEATURES = [
        c for c in df.columns
        if c not in ["date","ticker","stock_ret_fwd","spy_ret_fwd","excess_ret","y_cls","y_reg"]
    ]
    X = df[FEATURES]; y_cls = df["y_cls"]; y_reg = df["y_reg"]

    # --- 3. Classification model (direction) ---
    cls = Pipeline([
        ("sc", StandardScaler()),
        ("lr", LogisticRegression(
            penalty="l1", C=1.0, solver="liblinear", class_weight="balanced"))
    ])
    cls.fit(X, y_cls)
    acc = accuracy_score(y_cls, cls.predict(X))
    auc = roc_auc_score(y_cls, cls.predict_proba(X)[:,1])

    # --- 4. Regression model (magnitude) ---
    reg = Pipeline([
        ("sc", StandardScaler()),
        ("rf", RandomForestRegressor(
            n_estimators=300, random_state=0, n_jobs=-1))
    ])
    reg.fit(X, y_reg)
    mae = mean_absolute_error(y_reg, reg.predict(X))

    # --- 5. Save artifacts ---
    model_dir = f"artifacts/models/{ticker}"
    os.makedirs(model_dir, exist_ok=True)
    with open(f"{model_dir}/cls.pkl","wb") as f: pickle.dump(cls,f)
    with open(f"{model_dir}/reg.pkl","wb") as f: pickle.dump(reg,f)
    with open(f"{model_dir}/features.json","w") as f: json.dump(FEATURES,f, indent=2)

    report = {
        "ticker": ticker,
        "train_rows": len(df),
        "horizon_days": horizon,
        "timestamp": datetime.utcnow().isoformat(),
        "metrics": {"acc":acc,"auc":auc,"mae":mae}
    }
    os.makedirs("artifacts/reports", exist_ok=True)
    with open(f"artifacts/reports/{ticker}_report.json","w") as f: json.dump(report,f, indent=2)
    print(f"[{ticker}] acc={acc:.3f} auc={auc:.3f} mae={mae:.4f}")
    print(f"Models saved to {model_dir}")
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", required=True)
    ap.add_argument("--start", default="2012-01-01")
    args = ap.parse_args()
    train_one(args.ticker, start=args.start)

import pandas as pd

def make_labels(df: pd.DataFrame, horizon_days=21) -> pd.DataFrame:
    """
    Create classification and regression labels for excess returns.
    """
    df = df.copy()
    df["stock_ret_fwd"] = df["close"].shift(-horizon_days)/df["close"] - 1
    df["spy_ret_fwd"]   = df["close_spy"].shift(-horizon_days)/df["close_spy"] - 1
    df["excess_ret"]    = df["stock_ret_fwd"] - df["spy_ret_fwd"]
    df["y_cls"] = (df["excess_ret"] > 0).astype(int)
    df["y_reg"] = df["excess_ret"].abs()
    return df.dropna().reset_index(drop=True)

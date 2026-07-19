import pandas as pd
import numpy as np

def make_features(df, y_col, X_cols, lags=(1, 2, 24, 25, 48), rolling_windows=(24, 168)):
    df = df.copy()

    # Autoregressive lags (mirror your ARIMA orders: p=1, seasonal P=1 at m=24)
    for lag in lags:
        df[f'{y_col}_lag{lag}'] = df[y_col].shift(lag)

    # Rolling stats — captures level/trend info that lags alone miss
    for w in rolling_windows:
        df[f'{y_col}_rollmean{w}'] = df[y_col].shift(1).rolling(w).mean()
        df[f'{y_col}_rollstd{w}'] = df[y_col].shift(1).rolling(w).std()

    # Seasonal differencing as a feature — since you confirmed D=1 mattered
    df[f'{y_col}_diff24'] = df[y_col].diff(24)

    # Calendar features — since m=24 suggests hourly data with daily cycle
    df['hour'] = df['date'].dt.hour
    df['dayofweek'] = df['date'].dt.dayofweek
    df['is_weekend'] = (df['date'].dt.dayofweek >= 5).astype(int)

    # X's carry over unchanged — already exogenous/known ahead of time
    feature_cols = (
        [f'{y_col}_lag{l}' for l in lags]
        + [f'{y_col}_rollmean{w}' for w in rolling_windows]
        + [f'{y_col}_rollstd{w}' for w in rolling_windows]
        + [f'{y_col}_diff24', 'hour', 'dayofweek', 'is_weekend']
        + list(X_cols)
    )
    return df, feature_cols
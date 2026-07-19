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

import numpy as np

def rolling_forecast_lgb(model, history_df, X_test, y_test, y_col, feature_cols, lags, rolling_windows, horizon=24):
    history = history_df.copy()
    all_preds = []

    for day_start in range(0, len(X_test), horizon):
        X_day = X_test.iloc[day_start:day_start+horizon]
        y_day = y_test.iloc[day_start:day_start+horizon]

        day_preds = []
        day_history = history.copy()

        # within-day: still recursive, since you only have real y up to the start of the day
        for i in range(len(X_day)):
            next_row = X_day.iloc[[i]].copy()
            next_row[y_col] = np.nan
            day_history = pd.concat([day_history, next_row])

            feat_df, _ = make_features(day_history, y_col=y_col, X_cols=X_day.columns,
                                         lags=lags, rolling_windows=rolling_windows)
            x_input = feat_df.iloc[[-1]][feature_cols]
            y_hat = model.predict(x_input)[0]

            day_preds.append(y_hat)
            day_history.iloc[-1, day_history.columns.get_loc(y_col)] = y_hat

        all_preds.extend(day_preds)

        # NOW feed the real y_day into the master history before next day
        real_day = X_day.copy()
        real_day[y_col] = y_day.values
        history = pd.concat([history, real_day])

    return pd.Series(all_preds, index=X_test.index)

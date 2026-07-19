from statsmodels.tsa.arima.model import ARIMA
from config import order, seasonal_order, arima_feature_columns
import pandas as pd

def fit_arima(y, X):
    X = X[arima_feature_columns]  # Select only the specified feature columns
    model = ARIMA(y, X, order, seasonal_order)
    return model.fit()

def predict_arima(model, y_test, X_test):
    X_test = X_test[arima_feature_columns]  # Select only the specified feature columns
    horizon = 24  # one day, given m=24

    arimax_preds = []
    current_results = model  # your already-fitted model

    for day_start in range(0, len(y_test), horizon):
        X_day = X_test.iloc[day_start:day_start+horizon]
        y_day = y_test.iloc[day_start:day_start+horizon]

        fc = current_results.get_forecast(steps=len(X_day), exog=X_day)
        arimax_preds.append(fc.predicted_mean)

        # feed the REAL y_day back in, updating state (not re-optimizing coefficients)
        current_results = current_results.apply(endog=y_day, exog=X_day)

    arimax_preds = pd.concat(arimax_preds)
    return arimax_preds
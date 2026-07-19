from statsmodels.tsa.arima.model import ARIMA

def fit_arima(y, X, order, seasonal_order):
    model = ARIMA(y, X, order, seasonal_order)
    return model.fit()
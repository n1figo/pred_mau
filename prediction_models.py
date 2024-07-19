import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

def numeric_prediction(current_mau, growth_rate=0.15):
    """
    Simple numeric prediction based on current MAU and fixed growth rate.
    """
    return current_mau * (1 + growth_rate)

def regression_prediction(historical_data):
    """
    Linear regression prediction based on historical data.
    """
    X = np.array(range(len(historical_data))).reshape(-1, 1)
    y = np.array(historical_data)
    model = LinearRegression()
    model.fit(X, y)
    next_month = len(historical_data)
    return model.predict([[next_month]])[0]

def ml_prediction(historical_data, features):
    """
    Machine learning prediction using polynomial regression.
    This is a simplified version and should be replaced with a proper ML model in production.
    """
    X = np.array(range(len(historical_data))).reshape(-1, 1)
    y = np.array(historical_data)
    
    poly_features = PolynomialFeatures(degree=2)
    X_poly = poly_features.fit_transform(X)
    
    model = LinearRegression()
    model.fit(X_poly, y)
    
    next_month = len(historical_data)
    X_next = poly_features.transform([[next_month]])
    return model.predict(X_next)[0]

def get_predictions(current_mau, historical_data, features):
    numeric_pred = numeric_prediction(current_mau)
    regression_pred = regression_prediction(historical_data)
    ml_pred = ml_prediction(historical_data, features)
    
    return {
        'numeric': numeric_pred,
        'regression': regression_pred,
        'machine_learning': ml_pred
    }
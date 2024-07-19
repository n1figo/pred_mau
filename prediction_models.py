import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from datetime import datetime, timedelta
import calendar

def get_remaining_days(date):
    month_end = date.replace(day=calendar.monthrange(date.year, date.month)[1])
    days = (month_end - date).days + 1
    weekdays = sum(1 for d in range(days) if (date + timedelta(days=d)).weekday() < 5)
    weekends = days - weekdays
    return weekdays, weekends

def calculate_daily_average(mau, day_of_month):
    return mau / day_of_month

def numeric_prediction(current_mau, current_date, prev_month_data, login_increase, visitor_increase):
    # Calculate remaining days for current and previous month
    current_weekdays, current_weekends = get_remaining_days(current_date)
    prev_month = current_date.replace(day=1) - timedelta(days=1)
    prev_month_date = prev_month.replace(day=current_date.day)
    prev_weekdays, prev_weekends = get_remaining_days(prev_month_date)

    # Calculate average daily increase
    avg_login_increase = login_increase / 30  # Assuming 30 days per month
    avg_visitor_increase = visitor_increase / 30

    # Determine average weekday and weekend customers for previous month
    prev_weekday_avg = prev_month_data['weekday_avg']
    prev_weekend_avg = prev_month_data['weekend_avg']

    # Apply averages to current month
    expected_increase = (current_weekdays * prev_weekday_avg + current_weekends * prev_weekend_avg) * \
                        (avg_login_increase + avg_visitor_increase)

    # Calculate growth factor
    current_daily_avg = calculate_daily_average(current_mau, current_date.day)
    prev_daily_avg = calculate_daily_average(prev_month_data['mau'], current_date.day)
    growth_factor = current_daily_avg / prev_daily_avg

    # Apply growth factor to prediction
    predicted_mau = current_mau + (expected_increase * growth_factor)

    return predicted_mau

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
    current_date = datetime.now()
    prev_month_data = {
        'mau': historical_data[-1],
        'weekday_avg': features.get('prev_weekday_avg', 10000),  # Use provided value or default
        'weekend_avg': features.get('prev_weekend_avg', 8000)    # Use provided value or default
    }
    login_increase = features['login_customers'] - features.get('prev_login_customers', historical_data[-2])
    visitor_increase = features['unique_visitors'] - features.get('prev_unique_visitors', historical_data[-3])

    numeric_pred = numeric_prediction(current_mau, current_date, prev_month_data, login_increase, visitor_increase)
    regression_pred = regression_prediction(historical_data)
    ml_pred = ml_prediction(historical_data, features)
    
    return {
        'numeric': numeric_pred,
        'regression': regression_pred,
        'machine_learning': ml_pred
    }
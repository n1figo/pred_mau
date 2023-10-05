import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

# CSV 파일 읽기

data = pd.read_csv('your_file_name.csv')

# 데이터 분리

x = data[['daily_visitors', 'total_monthly_visitors', 'days_left_to_month_end', 'weekday', 'remaining_weekdays', 'remaining_weekends']].values
y = data['mau'].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# 모델 정의

models = {
'Linear Regression': LinearRegression(),
'Decision Tree': DecisionTreeRegressor(),
'Random Forest': RandomForestRegressor(),
'Support Vector Machines': SVR(kernel='linear')
}

# 모델 학습 및 성능 평가

best_model = None
best_score = float('inf')

for name, model in models.items():
model.fit(x_train, y_train)
scores = -cross_val_score(model, x_train, y_train, cv=5, scoring='neg_mean_squared_error').mean()


if scores < best_score:
    best_score = scores
    best_model = model

print(f"{name} MSE: {scores}")






# 가장 오차가 적은 모델로 예측

predictions = best_model.predict(x_test)
print(f"Predictions with {best_model.**class**.**name**}: {predictions}")

mse = mean_squared_error(y_test, predictions)
print(f"Test MSE with best model ({best_model.**class**.**name**}): {mse}")
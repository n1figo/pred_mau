import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def generate_daily_data(start_date, end_date, start_value, end_value):
    date_range = pd.date_range(start=start_date, end=end_date)
    values = np.linspace(start_value, end_value, len(date_range), dtype=int)
    return list(date_range), list(values)

# 월별 데이터
monthly_data = {
    'date': ['2024-01-31', '2024-02-29', '2024-03-31', '2024-04-30', '2024-05-31', '2024-06-30'],
    'login_customers': [769912, 698529, 716138, 699452, 757722, 709116],
    'unique_visitors': [124686, 131059, 145582, 143221, 140186, 144679],
    'mau': [894598, 829588, 861720, 842653, 877908, 853795]
}

# 6월 데이터 생성
june_start = datetime(2024, 6, 1)
june_end = datetime(2024, 6, 30)
june_dates, june_login = generate_daily_data(june_start, june_end, 38933, 709116)
_, june_visitors = generate_daily_data(june_start, june_end, 8786, 144679)
_, june_mau = generate_daily_data(june_start, june_end, 47719, 853795)

# 7월 데이터
july_data = {
    'date': [
        '2024-07-01', '2024-07-02', '2024-07-03', '2024-07-04', '2024-07-05', '2024-07-06',
        '2024-07-07', '2024-07-08', '2024-07-09', '2024-07-10', '2024-07-11', '2024-07-12',
        '2024-07-13', '2024-07-14', '2024-07-15', '2024-07-16', '2024-07-17', '2024-07-18', '2024-07-19'
    ],
    'login_customers': [
        68632, 109521, 144953, 181490, 214173, 240834,
        248994, 279994, 308828, 341395, 369304, 394307,
        409646, 422479, 451211, 475342, 498096, 521012, 541012
    ],
    'unique_visitors': [
        14260, 23817, 32381, 41414, 49028, 54072,
        58624, 65118, 71396, 78364, 84074, 89386,
        93212, 96539, 102134, 106976, 111525, 115900, 120100
    ],
    'mau': [
        82892, 133332, 177334, 222904, 263201, 287621,
        307618, 345112, 380224, 419759, 453378, 483693,
        502858, 519017, 553345, 582318, 609621, 609621, 609621
    ]
}

# 모든 데이터 합치기
all_dates = monthly_data['date'][:-1] + june_dates + july_data['date']
all_login = monthly_data['login_customers'][:-1] + june_login + july_data['login_customers']
all_visitors = monthly_data['unique_visitors'][:-1] + june_visitors + july_data['unique_visitors']
all_mau = monthly_data['mau'][:-1] + june_mau + july_data['mau']

# DataFrame 생성
df = pd.DataFrame({
    'date': all_dates,
    'login_customers': all_login,
    'unique_visitors': all_visitors,
    'mau': all_mau
})

# 날짜 열을 datetime 타입으로 변환
df['date'] = pd.to_datetime(df['date'])

# data 폴더가 없으면 생성
if not os.path.exists('data'):
    os.makedirs('data')

# 엑셀 파일로 저장 (data 폴더에)
df.to_excel('data/mau_data.xlsx', index=False)

print("data/mau_data.xlsx 파일이 생성되었습니다.")
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import streamlit as st

def load_data():
    # 실제로는 데이터베이스나 파일에서 로드해야 합니다.
    # 여기서는 예시 데이터를 사용합니다.
    data = {
        'date': pd.date_range(start='2024-06-01', end='2024-06-30'),
        'login_customers': np.cumsum(np.random.randint(1000, 5000, 30)),
        'unique_visitors': np.cumsum(np.random.randint(500, 2000, 30)),
        'MAU': np.cumsum(np.random.randint(1500, 7000, 30))
    }
    return pd.DataFrame(data)

def predict_mau(current_date, login_customers, login_customers_mom, 
                unique_visitors, unique_visitors_mom, 
                mau, mau_mom, prev_month_data):
    
    # 남은 날짜 계산
    month_end = pd.Timestamp(current_date).to_period('M').to_timestamp('M')
    remaining_days = (month_end - pd.Timestamp(current_date)).days + 1
    
    # 남은 평일/주말 수 계산
    remaining_weekdays = np.busday_count(current_date, month_end.date()) + 1
    remaining_weekends = remaining_days - remaining_weekdays
    
    # 전월 동일 기간 평균 계산
    prev_month_same_period = prev_month_data[prev_month_data['date'].dt.day >= current_date.day]
    prev_weekday_avg = prev_month_same_period[prev_month_same_period['date'].dt.dayofweek < 5]['MAU'].diff().mean()
    prev_weekend_avg = prev_month_same_period[prev_month_same_period['date'].dt.dayofweek >= 5]['MAU'].diff().mean()
    
    # 현재 추세에 따른 배수 계산
    if mau_mom > 1:
        multiplier = 1.1  # 성장 중이면 10% 증가
    else:
        multiplier = 0.9  # 감소 중이면 10% 감소
    
    # 예상 증가량 계산
    expected_increase = (prev_weekday_avg * remaining_weekdays + prev_weekend_avg * remaining_weekends) * multiplier
    
    # 최종 MAU 예측
    predicted_mau = mau + expected_increase
    
    return predicted_mau

# Streamlit 앱
st.title('MAU 예측 대시보드')

# 사용자 입력
current_date = st.date_input('현재 날짜', datetime.now())
login_customers = st.number_input('로그인 고객수 (누적)', min_value=0)
login_customers_mom = st.number_input('로그인 고객수 전월비', min_value=0.0)
unique_visitors = st.number_input('방문자 고유 ID (누적)', min_value=0)
unique_visitors_mom = st.number_input('방문자 고유 ID 전월비', min_value=0.0)
mau = st.number_input('MAU (누적)', min_value=0)
mau_mom = st.number_input('MAU 전월비', min_value=0.0)

# 데이터 로드
prev_month_data = load_data()

if st.button('예측'):
    predicted_mau = predict_mau(current_date, login_customers, login_customers_mom,
                                unique_visitors, unique_visitors_mom,
                                mau, mau_mom, prev_month_data)
    st.success(f'예상 월말 MAU: {predicted_mau:,.0f}')

# 그래프 표시
st.subheader('전월 MAU 추이')
st.line_chart(prev_month_data.set_index('date')['MAU'])

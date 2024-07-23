import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import cv2
from PIL import Image
import pytesseract
from prediction_models import get_predictions

# 데이터 로드 함수
@st.cache_data
def load_data():
    df = pd.read_excel('data/mau_data.xlsx')
    df['date'] = pd.to_datetime(df['date'])
    return df

# 이미지에서 텍스트 추출 함수
def extract_text_from_image(image):
    img_array = np.array(image)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    text = pytesseract.image_to_string(gray)
    return text

# 이미지 회전 함수
def rotate_image(image, angle):
    img_array = np.array(image)
    height, width = img_array.shape[:2]
    center = (width / 2, height / 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(img_array, rotation_matrix, (width, height))
    return Image.fromarray(rotated_image)

# 예측 함수
def predict_mau(mau, login_customers, unique_visitors, mau_mom, df):
    historical_data = df['mau'].tolist()[-6:]  # 최근 6개월 데이터 사용
    features = {
        'login_customers': login_customers,
        'unique_visitors': unique_visitors,
        'mau_mom': mau_mom,
        'prev_login_customers': df['login_customers'].iloc[-2],
        'prev_unique_visitors': df['unique_visitors'].iloc[-2],
        'prev_weekday_avg': df['mau'].iloc[-30:-2][df['date'].dt.dayofweek < 5].mean(),
        'prev_weekend_avg': df['mau'].iloc[-30:-2][df['date'].dt.dayofweek >= 5].mean(),
    }
    return get_predictions(mau, historical_data, features)

# 최근 연월 데이터 표시 함수
def show_recent_data(data):
    st.subheader("최근 연월 데이터")
    recent_data = data.sort_values('date', ascending=True).head(12)  # 최근 12개월 데이터
    recent_data['년월'] = recent_data['date'].dt.strftime('%Y-%m')
    recent_data_display = recent_data[['년월', 'mau', 'login_customers', 'unique_visitors']]
    recent_data_display.columns = ['년월', 'MAU', '로그인 고객수', '방문자 고유 ID']
    st.table(recent_data_display)

# 페이지 설정
st.set_page_config(page_title="MAU 예측 대시보드", layout="wide")

# CSS를 사용하여 입력 필드의 스타일 조정
st.markdown("""
<style>
    .stNumberInput input {
        -webkit-appearance: none !important;
        -moz-appearance: textfield !important;
        appearance: textfield !important;
    }
    .stNumberInput input::-webkit-outer-spin-button,
    .stNumberInput input::-webkit-inner-spin-button {
        -webkit-appearance: none !important;
        margin: 0 !important;
    }
</style>
""", unsafe_allow_html=True)

# 제목
st.title("MAU 예측 대시보드")

# 데이터 로드
df = load_data()

# 최근 연월 데이터 표시
show_recent_data(df)

# 사이드바: 날짜 선택
selected_date = st.sidebar.date_input("날짜 선택", df['date'].max())

# 이미지 업로드 섹션
st.subheader("최신 MAU 데이터 업로드")
uploaded_file = st.file_uploader("MAU 데이터 이미지 업로드", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="원본 이미지", use_column_width=True)
    
    angle = st.slider("이미지 회전 각도", min_value=-180, max_value=180, value=0, step=1)
    rotated_image = rotate_image(image, angle)
    st.image(rotated_image, caption="회전된 이미지", use_column_width=True)
    
    if st.button("데이터 추출"):
        extracted_text = extract_text_from_image(rotated_image)
        st.text("추출된 텍스트:")
        st.text(extracted_text)
        
        # TODO: 추출된 텍스트를 파싱하고 데이터를 업데이트하는 로직 추가
        st.success("데이터가 성공적으로 추출되었습니다.")

# 선택된 날짜의 데이터 가져오기
selected_data = df[df['date'] == pd.to_datetime(selected_date)]

if not selected_data.empty:
    current_data = selected_data.iloc[0]
    
    # 메인 섹션을 두 컬럼으로 나눕니다
    left_column, right_column = st.columns(2)

    with left_column:
        st.subheader("현재 데이터 입력")
        login_customers = st.number_input("로그인 고객수 (누적)", min_value=0, value=int(current_data['login_customers']), format="%d")
        unique_visitors = st.number_input("방문자 고유 ID (누적)", min_value=0, value=int(current_data['unique_visitors']), format="%d")
        mau = st.number_input("MAU (누적)", min_value=0, value=int(current_data['mau']), format="%d")
        mau_mom = st.number_input("MAU 전월비", min_value=0.0, value=float(df['mau'].pct_change().iloc[-1]), format="%f")

        # 예측 실행
        predictions = predict_mau(mau, login_customers, unique_visitors, mau_mom, df)
        
        st.success(f"예측된 월말 MAU:")
        st.write(f"숫자 기반 모델: {predictions['numeric']:,.0f}")
        st.write(f"회귀 모델: {predictions['regression']:,.0f}")
        st.write(f"머신러닝 모델: {predictions['machine_learning']:,.0f}")

    with right_column:
        st.subheader("MAU 추세 및 예측")
        
        fig = go.Figure()

        # 실제 데이터 (파란색 선)
        fig.add_trace(go.Scatter(x=df['date'], y=df['mau'], mode='lines+markers', name='실제 MAU', line=dict(color='blue')))

        # 예측 데이터 준비
        last_date = df['date'].max()
        future_dates = [last_date + timedelta(days=30*i) for i in range(1, 4)]  # 3개월 예측
        
        # 예측 모델별 선 그리기
        for model, color in zip(['numeric', 'regression', 'machine_learning'], ['red', 'orange', 'green']):
            fig.add_trace(go.Scatter(x=future_dates, y=[df['mau'].iloc[-1]] + [predictions[model]] * 2, 
                                     mode='lines', name=f'{model} 예측', line=dict(color=color, dash='dash')))

        fig.update_layout(title='MAU 추세 및 예측',
                          xaxis_title='날짜',
                          yaxis_title='MAU',
                          height=500,
                          legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
        
        st.plotly_chart(fig, use_container_width=True)

# 추가 정보 섹션
st.subheader("추가 정보")
st.info("""
이 대시보드는 현재 입력된 데이터를 기반으로 월말 MAU를 예측합니다.
3가지 예측 모델을 사용합니다:
1. 숫자 기반 모델: 현재 MAU와 고정 성장률을 기반으로 한 단순 예측
2. 회귀 모델: 과거 데이터를 사용한 선형 회귀 예측
3. 머신러닝 모델: 다항 회귀를 사용한 예측 (실제 환경에서는 더 복잡한 모델로 대체 가능)

실제 결과는 다양한 외부 요인에 따라 달라질 수 있습니다.
""")

# 일별 MAU 데이터 표시 및 수기 입력
st.subheader("일별 MAU 데이터")

# 선택된 날짜의 데이터 표시 또는 새 데이터 입력
st.write(f"선택된 날짜: {selected_date}")
cols = st.columns(4)
with cols[0]:
    st.write("날짜")
    st.write(selected_date)
with cols[1]:
    new_mau = st.number_input("MAU", value=int(current_data['mau']) if not selected_data.empty else 0, key="mau_edit", format="%d")
with cols[2]:
    new_login = st.number_input("로그인 고객수", value=int(current_data['login_customers']) if not selected_data.empty else 0, key="login_edit", format="%d")
with cols[3]:
    new_visitors = st.number_input("방문자 고유 ID", value=int(current_data['unique_visitors']) if not selected_data.empty else 0, key="visitors_edit", format="%d")

if st.button("데이터 저장/수정"):
    if selected_data.empty:
        # 새 데이터 추가
        new_data = pd.DataFrame({
            'date': [pd.to_datetime(selected_date)],
            'mau': [new_mau],
            'login_customers': [new_login],
            'unique_visitors': [new_visitors]
        })
        df = pd.concat([df, new_data], ignore_index=True)
        df = df.sort_values('date').reset_index(drop=True)
        st.success("새 데이터가 추가되었습니다.")
    else:
        # 기존 데이터 수정
        df.loc[df['date'] == pd.to_datetime(selected_date), 'mau'] = new_mau
        df.loc[df['date'] == pd.to_datetime(selected_date), 'login_customers'] = new_login
        df.loc[df['date'] == pd.to_datetime(selected_date), 'unique_visitors'] = new_visitors
        st.success("데이터가 수정되었습니다.")

    # 데이터 저장
    df.to_excel('data/mau_data_updated.xlsx', index=False)
    st.success("데이터가 성공적으로 저장되었습니다.")

    # 최근 연월 데이터 표 업데이트
    show_recent_data(df)

# 전체 데이터 표시
if st.checkbox("전체 데이터 보기"):
    st.write(df)
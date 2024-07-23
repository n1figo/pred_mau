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

# 페이지 설정
st.set_page_config(page_title="MAU 예측 대시보드", layout="wide")

# 제목
st.title("MAU 예측 대시보드")

# 데이터 로드
df = load_data()

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

# 일별 MAU 데이터 표시 및 수기 입력
st.subheader("일별 MAU 데이터")

# 데이터 편집을 위한 데이터프레임 생성
editable_df = df[['date', 'mau', 'login_customers', 'unique_visitors']].copy()
editable_df['date'] = editable_df['date'].dt.date  # datetime을 date로 변환

# 데이터 편집 기능
edited_df = st.data_editor(
    editable_df,
    column_config={
        "date": st.column_config.DateColumn("Date", disabled=True),
        "mau": st.column_config.NumberColumn("MAU"),
        "login_customers": st.column_config.NumberColumn("로그인 고객수"),
        "unique_visitors": st.column_config.NumberColumn("방문자 고유 ID"),
    },
    use_container_width=True,
    num_rows="dynamic"
)

# 변경된 데이터 처리
if not edited_df.equals(editable_df):
    df.update(edited_df)
    st.success("데이터가 업데이트되었습니다.")

# 메인 섹션을 두 컬럼으로 나눕니다
left_column, right_column = st.columns(2)

with left_column:
    st.subheader("현재 데이터 입력")
    current_data = df[df['date'] == pd.to_datetime(selected_date)].iloc[0]
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
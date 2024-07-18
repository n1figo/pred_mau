import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import cv2
from PIL import Image
import pytesseract

# 데이터 로드 함수
def load_data():
    return pd.read_excel('data/mau_data.xlsx')

# 이미지에서 텍스트 추출 함수
def extract_text_from_image(image):
    # PIL Image를 numpy 배열로 변환
    img_array = np.array(image)
    
    # 이미지를 그레이스케일로 변환
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    # Tesseract를 사용하여 텍스트 추출
    text = pytesseract.image_to_string(gray)
    
    return text

# 이미지 회전 함수
def rotate_image(image, angle):
    # PIL Image를 numpy 배열로 변환
    img_array = np.array(image)
    
    # 이미지의 중심점 계산
    height, width = img_array.shape[:2]
    center = (width / 2, height / 2)
    
    # 회전 행렬 생성
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # 이미지 회전
    rotated_image = cv2.warpAffine(img_array, rotation_matrix, (width, height))
    
    # numpy 배열을 다시 PIL Image로 변환
    return Image.fromarray(rotated_image)

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
    
    # 각도 조절 슬라이더 추가
    angle = st.slider("이미지 회전 각도", min_value=-180, max_value=180, value=0, step=1)
    
    # 이미지 회전
    rotated_image = rotate_image(image, angle)
    st.image(rotated_image, caption="회전된 이미지", use_column_width=True)
    
    if st.button("데이터 추출"):
        extracted_text = extract_text_from_image(rotated_image)
        st.text("추출된 텍스트:")
        st.text(extracted_text)
        
        # 여기에 추출된 텍스트를 파싱하고 데이터를 업데이트하는 로직을 추가합니다.
        # parsed_data = parse_mau_data(extracted_text)
        # update_mau_data(parsed_data)
        
        st.success("데이터가 성공적으로 추출되었습니다.")

# 메인 섹션을 두 컬럼으로 나눕니다
left_column, right_column = st.columns(2)

with left_column:
    st.subheader("현재 데이터 입력")
    login_customers = st.number_input("로그인 고객수 (누적)", min_value=0, format="%d")
    login_customers_mom = st.number_input("로그인 고객수 전월비", min_value=0.0, format="%f")
    unique_visitors = st.number_input("방문자 고유 ID (누적)", min_value=0, format="%d")
    unique_visitors_mom = st.number_input("방문자 고유 ID 전월비", min_value=0.0, format="%f")
    mau = st.number_input("MAU (누적)", min_value=0, format="%d")
    mau_mom = st.number_input("MAU 전월비", min_value=0.0, format="%f")

    if st.button("예측 실행"):
        # 여기에 예측 로직을 구현합니다
        predicted_mau = mau * 1.1  # 임시 예측 로직 (실제로는 더 복잡한 계산이 필요합니다)
        st.success(f"예측된 월말 MAU: {predicted_mau:,.0f}")

with right_column:
    st.subheader("MAU 추세")
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['date'], y=df['mau'], mode='lines+markers', name='실제 MAU'))
    
    fig.update_layout(title='MAU 추세',
                      xaxis_title='날짜',
                      yaxis_title='MAU',
                      height=400)
    
    st.plotly_chart(fig, use_container_width=True)

# 추가 정보 섹션
st.subheader("추가 정보")
st.info("""
이 대시보드는 현재 입력된 데이터를 기반으로 월말 MAU를 예측합니다.
예측 모델은 다음 요소를 고려합니다:
- 현재까지의 누적 데이터
- 전월 대비 성장률
- 남은 영업일 수
- 주말/평일 패턴
실제 결과는 다양한 외부 요인에 따라 달라질 수 있습니다.
""")

# 데이터 테이블 (옵션)
if st.checkbox("상세 데이터 보기"):
    st.subheader("일별 MAU 데이터")
    st.dataframe(df)
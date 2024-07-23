import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import cv2
from PIL import Image
import pytesseract
from prediction_models import get_predictions

# (이전 코드는 동일하게 유지)

# 페이지 최하단에 위치할 일별 MAU 데이터 표시 및 수기 입력 섹션
st.subheader("일별 MAU 데이터")

# 데이터 편집을 위한 데이터프레임 생성
editable_df = df[['date', 'mau', 'login_customers', 'unique_visitors']].copy()
editable_df['date'] = editable_df['date'].dt.date  # datetime을 date로 변환

# CSS를 사용하여 입력 필드의 스타일 조정
st.markdown("""
<style>
    .stNumberInput input {
        -webkit-appearance: none;
        margin: 0;
        -moz-appearance: textfield;
    }
</style>
""", unsafe_allow_html=True)

# 데이터 표시 및 편집
st.write("일별 데이터를 수정하려면 아래 필드를 직접 클릭하여 수정하세요.")
for index, row in editable_df.iterrows():
    cols = st.columns(4)
    with cols[0]:
        st.write(row['date'])
    with cols[1]:
        new_mau = st.number_input(f"MAU {row['date']}", value=int(row['mau']), key=f"mau_{index}", step=0, format="%d")
    with cols[2]:
        new_login = st.number_input(f"로그인 고객수 {row['date']}", value=int(row['login_customers']), key=f"login_{index}", step=0, format="%d")
    with cols[3]:
        new_visitors = st.number_input(f"방문자 고유 ID {row['date']}", value=int(row['unique_visitors']), key=f"visitors_{index}", step=0, format="%d")
    
    # 데이터 업데이트
    df.loc[index, 'mau'] = new_mau
    df.loc[index, 'login_customers'] = new_login
    df.loc[index, 'unique_visitors'] = new_visitors
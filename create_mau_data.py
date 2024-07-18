import pandas as pd
from datetime import datetime, timedelta

# 이미지에서 추출한 데이터
data = {
    'date': ['2024-01-31', '2024-02-29', '2024-03-31', '2024-04-30', '2024-05-31', '2024-06-30'],
    'login_customers': [769912, 698529, 716138, 699452, 737722, 709116],
    'unique_visitors': [124686, 131059, 145582, 143221, 140186, 144679],
    'mau': [894598, 829588, 861720, 842653, 877908, 853795]
}

# DataFrame 생성
df = pd.DataFrame(data)

# 날짜를 datetime 형식으로 변환
df['date'] = pd.to_datetime(df['date'])

# 6월 데이터 상세화 (일별 데이터 생성)
june_data = []
june_start = datetime(2024, 6, 1)
june_end = datetime(2024, 6, 30)
current_date = june_start

while current_date <= june_end:
    june_data.append({
        'date': current_date,
        'login_customers': None,
        'unique_visitors': None,
        'mau': None
    })
    current_date += timedelta(days=1)

# 6월 1일과 30일 데이터 추가
june_data[0]['login_customers'] = 38933
june_data[0]['unique_visitors'] = 8786
june_data[0]['mau'] = 47719

june_data[-1]['login_customers'] = 709116
june_data[-1]['unique_visitors'] = 144679
june_data[-1]['mau'] = 853795

# 6월 데이터를 DataFrame에 추가
june_df = pd.DataFrame(june_data)
df = pd.concat([df, june_df]).reset_index(drop=True)

# 날짜순으로 정렬
df = df.sort_values('date')

# 엑셀 파일로 저장
df.to_excel('data/mau_data.xlsx', index=False)

print("mau_data.xlsx file has been created in the data folder.")
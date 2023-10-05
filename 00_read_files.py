from calendar import monthrange
import pandas as pd


# 엑셀 파일 읽기
df1 = pd.read_excel('D:/github/pred_mau/input/raw_20231005.xlsx', sheet_name='Sheet1', engine='openpyxl')
df2 = pd.read_excel('D:/github/pred_mau/input/raw_20231005.xlsx', sheet_name='Sheet2', engine='openpyxl')
df3 = pd.read_excel('D:/github/pred_mau/input/raw_20231005.xlsx', sheet_name='Sheet3', engine='openpyxl')


# 각 데이터프레임에서 5번째부터 8번째 컬럼만 추출
df1_selected = df1.iloc[:, 6:13]
df2_selected = df2.iloc[:, 6:13]
df3_selected = df3.iloc[:, 6:13]

# 결과 확인
print(df1_selected.head())
print(df2_selected.head())
print(df3_selected.head())

# df1_selected.to_csv('D:/github/pred_mau/input/out.csv', encoding='cp949')



# 주중(1)과 주말(0)을 나타내는 새로운 컬럼 생성

def make_weekdays_weekends(df):
    df2 = df.copy()

    df2.columns = ['연월일','요일','로그인고객수','전일비','방문자고유ID접속자수','합계','전일비2']

    df2['주중_주말'] = df2['요일'].apply(lambda x: 0 if x in ['토', '일'] else 1)

    # 월말까지 잔여일수 계산
    df2['연월일_'] = pd.to_datetime(df2['연월일'], format='%Y%m%d')  # 날짜 형식으로 변환
    df2['월말까지_잔여일수'] = df2['연월일_'].apply(lambda x: monthrange(x.year, x.month)[1] - x.day)

    return df2

# 잔여 평일수와 주말수 계산
def calculate_remaining_weekdays_and_weekends(row):
    total_days = row['월말까지_잔여일수']
    date = row['연월일_']
    weekdays = 0
    weekends = 0
    
    for i in range(1, total_days + 1):
        next_day = date + pd.Timedelta(days=i)
        if next_day.weekday() < 5:  # 주중
            weekdays += 1
        else:  # 주말
            weekends += 1
    
    return pd.Series([weekdays, weekends])



# 반영
df1_selected_week = make_weekdays_weekends(df1_selected)
# df1_selected_week.to_csv('D:/github/pred_mau/input/tmp.csv', encoding='cp949')

df1_selected_week[['잔여_평일수', '잔여_주말수']] = df1_selected_week.apply(calculate_remaining_weekdays_and_weekends, axis=1)

# 결과 확인
print(df1_selected.head())

df1_selected_week.to_csv('D:/github/pred_mau/input/tmp.csv', encoding='cp949')
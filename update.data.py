import pandas as pd
import numpy as np
from datetime import datetime

def fetch_new_data():
    # 실제 데이터 소스에서 데이터를 가져오는 로직 구현 필요
    today = datetime.now().strftime('%Y-%m-%d')
    new_data = pd.DataFrame({
        'date': [today],
        'mau': [100000 + np.random.randint(-1000, 1000)]
    })
    return new_data

def update_excel(new_data):
    try:
        df = pd.read_excel('data/mau_data.xlsx')
    except FileNotFoundError:
        df = pd.DataFrame(columns=['date', 'mau'])
    
    df = pd.concat([df, new_data], ignore_index=True)
    df = df.drop_duplicates(subset=['date'], keep='last')
    df = df.sort_values('date')
    
    df.to_excel('data/mau_data.xlsx', index=False)

if __name__ == "__main__":
    new_data = fetch_new_data()
    update_excel(new_data)
    print("MAU data updated successfully.")
#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install webdriver_manager')


# In[92]:


from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import time
import pandas as pd
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from datetime import timedelta
import bisect
import re


# # Data Gathering

# In[101]:


## 법률 이슈 크롤링

# Chrome 드라이버 자동 설정
service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service)

# Argos 페이지 열기
url = "https://argos.nanet.go.kr/main/fusionanalysis/lawIssueGuest.do"
driver.get(url)

try:
    time.sleep(3)

    for page_num in range(1, 101):
        law_elements = driver.find_elements(By.CSS_SELECTOR, "nav#nav_law_list a")
        law_list = []
        
        for law_element in law_elements:
            law_id = law_element.get_attribute("id")
            law_name = law_element.text
            law_list.append({"id": law_id, "name": law_name})

        for law in law_list:
            try:
                law_id = law["id"]
                law_name = law["name"]
                script = f"setSelctLawIssList('{law_id}', '00000000', '{law_name}', 'L');"
                driver.execute_script(script)
                time.sleep(1)  

                download_button = WebDriverWait(driver, 30).until(
                    EC.element_to_be_clickable((By.XPATH, "//button[contains(@class, 'down') and text()='다운로드']"))
                )
                download_button.click()
                time.sleep(1)  
            except Exception as e:
                print(f"{law_name} (ID: {law_id}) Download Error: {e}")

        try:
            next_button = driver.find_element(By.ID, "nextPage")
            if next_button.is_enabled():
                next_button.click()
                time.sleep(2)  
            else:
                print("마지막 페이지")
                break
        except Exception as e:
            print(f"Page Error: {e}")
            break

finally:
    # 작업 완료 후 드라이버 종료
    driver.quit()


# In[102]:


# 엑셀 파일들이 있는 디렉토리 경로를 지정합니다.
directory = 'C:/Users/als31/Downloads'

# 병합할 모든 데이터프레임을 저장할 리스트
all_data = []

# 영문 칼럼명으로 변경하는 함수 (예시로 기본 변환 이름을 지정)
def rename_columns(columns):
    column_mappings = {
        "번호": "law_issue_num",
        "일자": "law_issue_date",
        "이슈구분": "law_issue_type",
        "법률이슈 제목": "law_issue_title"
    }
    return [column_mappings.get(col, col) for col in columns]

# 디렉토리 내 모든 엑셀 파일을 읽어와 병합
for filename in os.listdir(directory):
    if filename.endswith(".xlsx"):
        file_path = os.path.join(directory, filename)
        
        # 파일명에서 '법률이슈_' 제거
        law_name = os.path.splitext(filename)[0].replace("법률이슈_", "")
        
        # 엑셀 파일 읽기 (첫 줄을 헤더로 사용)
        df = pd.read_excel(file_path, header=1)  # 첫 번째 줄을 칼럼명으로 사용
        
        # 칼럼명을 영문으로 변경
        df.columns = rename_columns(df.columns)
        
        # 'law' 열 추가
        df.insert(loc=0, column='law', value = law_name)
        
        # 리스트에 데이터프레임 추가
        all_data.append(df)

# 모든 데이터를 하나의 데이터프레임으로 병합
issue = pd.concat(all_data, ignore_index=True)

# 병합된 데이터를 CSV 파일로 저장
issue.to_csv('C:/Users/als31/Downloads/law_issue.csv', index=False, encoding='utf-8-sig')
issue


# # Data Merging

# In[ ]:


# 토탈데이터
df = pd.read_csv('merged_v1_news_parl_issue.csv')


# In[ ]:


# Step 1: 불용어 제거 함수 정의
stopwords = ['법', '에 관한', '일부개정법률안', '특별법', '대한민국', '제정법', '일부개정', '등']
def remove_stopwords(text, stopwords):
    pattern = '|'.join(stopwords)
    return re.sub(pattern, '', text)

# Step 2: 불용어 제거한 필드를 생성하여 issue 데이터 전처리
issue['filtered_law'] = issue['law'].apply(lambda x: remove_stopwords(str(x[:-1]), stopwords))  # 마지막 글자 제외
issue['law_issue_date'] = pd.to_datetime(issue['law_issue_date'], errors='coerce')

# Step 3: law_issue_type별 칼럼을 df에 추가 (초기값 결측치로 설정)
law_issue_types = issue['law_issue_type'].unique()
for issue_type in law_issue_types:
    df[f'배경_{issue_type}'] = None  # None으로 초기화

# '배경_법명' 열을 추가하여 법률이슈가 찾아진 경우 law값을 저장
df['배경_법명'] = None

# Step 4: df의 각 행에 대해 조건에 맞는 법률 이슈를 찾아 처리
for i, df_row in tqdm(df.iterrows(), total=len(df), desc="Join process"):
    title = remove_stopwords(str(df_row['title']), stopwords)  # 불용어 제거 후 비교
    df_date = pd.to_datetime(df_row['date'], errors='coerce')

    # Step 4.1: title에 완전히 포함되는 law를 가진 issue 찾기
    matching_issues = issue[issue['filtered_law'].apply(lambda x: x in title)]

    if not matching_issues.empty:
        # Step 4.2: 2005년 이후 & df_date 이전인 이슈로 필터링
        relevant_issues = matching_issues[
            (matching_issues['law_issue_date'] >= '2005-01-01') &
            (matching_issues['law_issue_date'] < df_date)
            ]

        if not relevant_issues.empty:
            # Step 4.3: law_issue_type별로 갯수를 세어 각 칼럼에 저장
            counts = relevant_issues['law_issue_type'].value_counts()
            df.at[i, '배경_법명'] = relevant_issues.iloc[0]['law']
            for issue_type in law_issue_types:
                df.at[i, f'배경_{issue_type}'] = counts.get(issue_type, 0)
        else:
            # Step 4.4: 필터링 이후 모든 타입의 이슈가 없을 경우 0으로 설정
            df.at[i, '배경_법명'] = matching_issues.iloc[0]['law']
            for issue_type in law_issue_types:
                df.at[i, f'배경_{issue_type}'] = 0
    else:
        # Step 4.5: 매칭된 이슈가 없으면 결측치 유지 (배경_법명과 배경_타입 칼럼들)
        df.at[i, '배경_법명'] = None
        for issue_type in law_issue_types:
            df.at[i, f'배경_{issue_type}'] = None

merged_law_standard = df


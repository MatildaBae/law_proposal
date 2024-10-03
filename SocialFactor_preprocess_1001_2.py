#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import json
import pandas as pd
from tqdm import tqdm

# 기본 경로 설정
base_dir = 'TL1'  # 'TL1' 폴더 경로

# 결과를 저장할 빈 리스트
data = []

# TL 폴더 내의 모든 카테고리 폴더 탐색
for field in os.listdir(base_dir):  # 카테고리 폴더 탐색
    field_path = os.path.join(base_dir, field)
    
    if os.path.isdir(field_path):  # 폴더인지 확인
        # 카테고리 폴더 내의 JSON 파일 목록
        json_files = [f for f in os.listdir(field_path) if f.endswith('.json')]
        
        # 해당 카테고리 폴더 내의 JSON 파일 읽기 (tqdm 적용)
        for json_file in tqdm(json_files, desc=f"Processing JSON files in {field}"):
            json_path = os.path.join(field_path, json_file)
            
            with open(json_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
                
                # "named_entity" 항목 추출
                for entity in json_data.get('named_entity', []):
                    for title in entity.get('title', []):
                        sentence = title.get('sentence', '')  # 문장 추출

                        # "board", "write_date", "source_site"를 named_entity 내에서 추출
                        board = entity.get('board', '')  # "board"는 entity 내에 존재
                        write_date = entity.get('write_date', '')  # "write_date"는 entity 내에 존재
                        source_site = entity.get('source_site', '')  # "source_site"는 entity 내에 존재

                        # 데이터를 리스트에 저장
                        data.append({
                            'sentence': sentence,
                            'board': board,
                            'write_date': write_date,
                            'source_site': source_site,
                            'field': field  # 상위 폴더 이름을 'field'로 저장
                        })

# 데이터프레임으로 변환
train = pd.DataFrame(data)


# In[ ]:


# train 데이터프레임을 news_title_train.csv로 저장
train.to_csv('news_title_train.csv', index=False, encoding='utf-8')


# In[ ]:


# 테스트 폴더이름 모르겠음 !!

import os
import json
import pandas as pd
from tqdm import tqdm

# 기본 경로 설정
base_dir = 'VL1'  # 'VL1' 폴더 경로

# 결과를 저장할 빈 리스트
data = []

# TL 폴더 내의 모든 카테고리 폴더 탐색
for field in os.listdir(base_dir):  # 카테고리 폴더 탐색
    field_path = os.path.join(base_dir, field)
    
    if os.path.isdir(field_path):  # 폴더인지 확인
        # 카테고리 폴더 내의 JSON 파일 목록
        json_files = [f for f in os.listdir(field_path) if f.endswith('.json')]
        
        # 해당 카테고리 폴더 내의 JSON 파일 읽기 (tqdm 적용)
        for json_file in tqdm(json_files, desc=f"Processing JSON files in {field}"):
            json_path = os.path.join(field_path, json_file)
            
            with open(json_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
                
                # "named_entity" 항목 추출
                for entity in json_data.get('named_entity', []):
                    for title in entity.get('title', []):
                        sentence = title.get('sentence', '')  # 문장 추출

                        # "board", "write_date", "source_site"를 named_entity 내에서 추출
                        board = entity.get('board', '')  # "board"는 entity 내에 존재
                        write_date = entity.get('write_date', '')  # "write_date"는 entity 내에 존재
                        source_site = entity.get('source_site', '')  # "source_site"는 entity 내에 존재

                        # 데이터를 리스트에 저장
                        data.append({
                            'sentence': sentence,
                            'board': board,
                            'write_date': write_date,
                            'source_site': source_site,
                            'field': field  # 상위 폴더 이름을 'field'로 저장
                        })

# 데이터프레임으로 변환
test = pd.DataFrame(data)


# In[ ]:


# train 데이터프레임을 news_title_train.csv로 저장
test.to_csv('news_title_test.csv', index=False, encoding='utf-8')


# ### 직접적 요인 빈도수/기간 추출

# In[1]:


import pandas as pd
from tqdm import tqdm
from datetime import datetime, timedelta


# In[2]:


law = pd.read_csv('data/law_summary_train.csv', index_col=0)
law


# In[3]:


law = law[['id', 'field', 'date', 'terminology', 'disposal']]
law


# In[4]:


news = pd.read_csv('data/news_title_cleaned_train.csv')
news


# ### 빈도 확률 계산

# In[5]:


# 날짜 형식 변환
news['write_date'] = pd.to_datetime(news['write_date'])
law['date'] = pd.to_datetime(law['date'])

# tqdm을 통해 빈도 계산 작업 모니터링
def calculate_keyword_frequency(terminology, filtered_news):
    keywords = terminology.split(', ')
    keyword_count = 0
    
    for sentence in filtered_news['sentence']:
        for keyword in keywords:
            if keyword in sentence:
                keyword_count += 1
                
    return keyword_count

# 빈도 확률 계산
law['frequency_probability'] = 0.0

for idx, row in tqdm(law.iterrows(), total=law.shape[0], desc="Processing law keywords"):
    # 각 법률안의 date에서 1년 전까지의 뉴스만 필터링
    start_date = row['date'] - timedelta(days=365)
    end_date = row['date']
    
    # 해당 기간의 뉴스 데이터 필터링
    filtered_news = news[(news['write_date'] >= start_date) & (news['write_date'] <= end_date)]
    
    # 해당 기간의 뉴스 개수
    total_news_count = len(filtered_news)
    
    # terminology에 포함된 키워드 빈도 계산
    keyword_count = calculate_keyword_frequency(row['terminology'], filtered_news)
    
    # 빈도 확률 계산
    if total_news_count > 0:
        law.at[idx, 'frequency_probability'] = keyword_count / total_news_count
    else:
        law.at[idx, 'frequency_probability'] = 0

# 총 6시간


# In[6]:


law.to_csv('data/law_with_direct_freq.csv', index=False)


# In[9]:


law


# In[17]:


import matplotlib.pyplot as plt

# 히스토그램을 그리기 위한 데이터
frequency_data = law['frequency_probability']

# 히스토그램 그리기
plt.figure(figsize=(8, 6))
plt.hist(frequency_data, bins=10000, color='skyblue', edgecolor='black')
plt.title('Distribution of Frequency Probability')
plt.xlabel('Frequency Probability')
plt.ylabel('Frequency')
plt.xlim(0, 0.02)  # x축 범위를 0 ~ 0.03로 제한
plt.grid(True)

# 히스토그램 표시
plt.show()


# #### 1년치 데이터 슬라이싱

# In[10]:


# 뉴스 데이터: 2014-01-12부터 1년
news_start_date = pd.to_datetime('2014-01-12')
news_end_date = news_start_date + timedelta(days=365)
filtered_news = news[(news['write_date'] >= news_start_date) & (news['write_date'] <= news_end_date)]

# 법률 데이터: 2015-01-12부터 1년
law_start_date = pd.to_datetime('2015-01-12')
law_end_date = law_start_date + timedelta(days=365)
filtered_law = law[(law['date'] >= law_start_date) & (law['date'] <= law_end_date)]

# tqdm을 통해 빈도 계산 작업 모니터링
def calculate_keyword_frequency(terminology, filtered_news):
    keywords = terminology.split(', ')
    keyword_count = 0
    
    for sentence in filtered_news['sentence']:
        for keyword in keywords:
            if keyword in sentence:
                keyword_count += 1
                
    return keyword_count

# 빈도 확률 계산
filtered_law['frequency_probability'] = 0.0

for idx, row in tqdm(filtered_law.iterrows(), total=filtered_law.shape[0], desc="Processing law keywords"):
    # 각 법률안의 date에서 1년 전까지의 뉴스만 필터링
    start_date = row['date'] - timedelta(days=365)
    end_date = row['date']
    
    # 해당 기간의 뉴스 데이터 필터링
    recent_news = filtered_news[(filtered_news['write_date'] >= start_date) & (filtered_news['write_date'] <= end_date)]
    
    # 해당 기간의 뉴스 개수
    total_news_count = len(recent_news)
    
    # terminology에 포함된 키워드 빈도 계산
    keyword_count = calculate_keyword_frequency(row['terminology'], recent_news)
    
    # 빈도 확률 계산
    if total_news_count > 0:
        filtered_law.at[idx, 'frequency_probability'] = keyword_count / total_news_count
    else:
        filtered_law.at[idx, 'frequency_probability'] = 0


# In[11]:


filtered_law


# In[21]:


import matplotlib.pyplot as plt

# 히스토그램을 그리기 위한 데이터
frequency_data = filtered_law['frequency_probability']

# 히스토그램 그리기
plt.figure(figsize=(8, 6))
plt.hist(frequency_data, bins=100, color='skyblue', edgecolor='black')
plt.title('Distribution of Frequency Probability')
plt.xlabel('Frequency Probability')
plt.ylabel('Frequency')
plt.xlim(0, 0.008)  # x축 범위를 0 ~ 0.001로 제한
plt.grid(True)

# 히스토그램 표시
plt.show()


# ### 최근 날짜 차이(전체)

# In[7]:


# 날짜 형식 변환
# news['write_date'] = pd.to_datetime(news['write_date'])
# law['date'] = pd.to_datetime(law['date'])

# 뉴스 데이터를 날짜순으로 정렬 (여기에 tqdm 사용하지 않음)
news = news.sort_values(by='write_date')


# In[8]:


import pandas as pd
from tqdm import tqdm
from datetime import timedelta
import bisect
import numpy as np

# 실시간으로 중앙값을 구하는 함수
def calculate_median_gap(terminology, law_date, news_df):
    keywords = terminology.split(', ')
    if not keywords:
        return None

    # 날짜 차이를 저장할 리스트
    date_differences = []

    # 이미 처리된 날짜는 건너뛰도록 관리
    processed_dates = set()

    # 날짜 순으로 정렬된 뉴스 데이터에서 키워드를 찾는 과정
    for idx, row in news_df.iterrows():
        news_date = row['write_date']

        # 뉴스 날짜가 처리된 적이 있다면 건너뜀
        if news_date in processed_dates:
            continue

        # 해당 뉴스 문장에서 키워드가 포함되는지 확인
        for keyword in keywords:
            if keyword in row['sentence']:
                # 날짜 차이를 계산하고 리스트에 추가
                date_diff = (law_date - news_date).days
                if 0 < date_diff <= 365:  # 1년 이내만 고려
                    date_differences.append(date_diff)
                
                # 해당 날짜는 처리된 것으로 기록
                processed_dates.add(news_date)
                # 그 날짜에 대한 다른 뉴스는 무시하고 다음 날짜로 넘어감
                break

    # 리스트가 비어있으면 None을 반환
    if not date_differences:
        return None

    # 중앙값 계산
    return np.median(date_differences)

# 법률안에 대해 날짜 차이 중앙값 계산
law['median_date_difference'] = None

for idx, row in tqdm(law.iterrows(), total=law.shape[0], desc="Processing law keywords"):
    # 각 법률안의 날짜에서 1년 전까지의 뉴스만 필터링
    start_date = row['date'] - timedelta(days=365)
    end_date = row['date']
    
    # 해당 기간의 뉴스 데이터 필터링
    filtered_news = news[(news['write_date'] >= start_date) & (news['write_date'] <= end_date)]
    
    # 키워드와 뉴스 날짜 차이 중앙값을 계산
    median_diff = calculate_median_gap(row['terminology'], row['date'], filtered_news)
    
    # 결과 저장
    law.at[idx, 'median_date_difference'] = median_diff

# 결과 확인
# print(law[['id', 'terminology', 'median_date_difference']])


# In[ ]:


law.to_csv('data/law_with_direct_recent.csv', index=False)


# ### 최근 날짜 차이(우선 1년치만)

# In[18]:


# 뉴스 데이터: 2014-01-12부터 1년
news_start_date = pd.to_datetime('2014-01-12')
news_end_date = news_start_date + timedelta(days=365)
filtered_news = news[(news['write_date'] >= news_start_date) & (news['write_date'] <= news_end_date)]

# 법률 데이터: 2015-01-12부터 1년
law_start_date = pd.to_datetime('2015-01-12')
law_end_date = law_start_date + timedelta(days=365)
filtered_law = law[(law['date'] >= law_start_date) & (law['date'] <= law_end_date)]

# 키워드가 등장한 뉴스와 법률안 날짜 간의 차이를 계산하는 함수
def calculate_date_differences(terminology, law_date, news_df):
    keywords = terminology.split(', ')
    if not keywords:
        return None

    # 날짜 간격을 저장할 리스트
    date_differences = []

    # 키워드에 대한 각 뉴스에서의 날짜 간격을 계산
    for keyword in keywords:
        keyword = keyword.strip()  # 키워드 앞뒤 공백 제거
        if keyword:
            # 해당 키워드가 포함된 뉴스 기사 추출
            matching_news = news_df[news_df['sentence'].str.contains(keyword, na=False)]

            # 뉴스 기사별로 날짜 간격 계산
            for news_date in matching_news['write_date']:
                date_diff = (law_date - news_date).days  # 날짜 간격을 일 단위로 계산
                if 0 < date_diff < 365:  # 1년 이내 뉴스만 포함
                    date_differences.append(date_diff)

    return date_differences

# 각 법률안의 키워드가 뉴스에서 얼마나 최근에 나왔는지 계산
filtered_law['date_differences'] = filtered_law.apply(
    lambda row: calculate_date_differences(row['terminology'], row['date'], filtered_news), axis=1
)


# In[19]:


filtered_law


# In[20]:


import numpy as np
import statistics

# mode 계산 중 빈 리스트는 NaN으로 처리
filtered_law['date_diff_mod'] = filtered_law['date_differences'].apply(
    lambda x: np.nan if not x else statistics.mode(x)
)

filtered_law


# In[21]:


# NaN 값을 제외한 'date_diff_mod' 데이터
filtered_data_mode = filtered_law['date_diff_mod'].dropna()

# 히스토그램 그리기
plt.figure(figsize=(8, 6))
plt.hist(filtered_data_mode, bins=20, color='skyblue', edgecolor='black')
plt.title('Distribution of Mode of Date Differences')
plt.xlabel('Mode of Date Differences (days)')
plt.ylabel('Frequency')
plt.grid(True)

# 히스토그램 표시
plt.show()


# In[ ]:





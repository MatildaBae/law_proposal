#!/usr/bin/env python
# coding: utf-8

# In[7]:


import os
import json
import pandas as pd
import matplotlib.pyplot as plt


# ### 인터넷 기사 train 데이터 정제

# In[ ]:


# 폴더 경로 설정
root_dir = 'data/TL1'

# 추출할 데이터를 담을 리스트
data = []

# 각 json 파일을 열어서 필요한 데이터를 추출하는 함수
def extract_json_data(json_file_path, field_name):
    with open(json_file_path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
        # 각 named_entity에서 필요한 데이터를 추출
        for entity in json_data['named_entity']:
            # sentence 합치기
            sentences = ' '.join([content['sentence'] for content in entity['content']])
            source_site = entity['source_site']
            write_date = entity['write_date']
            # 데이터를 리스트에 추가, field_name도 포함
            data.append([sentences, source_site, write_date, field_name])

# 폴더 내 모든 json 파일을 반복하면서 데이터를 추출
for root, dirs, files in os.walk(root_dir):
    for dir_name in dirs:
        field_dir = os.path.join(root, dir_name)
        for file_name in os.listdir(field_dir):
            if file_name.endswith('.json'):
                json_file_path = os.path.join(field_dir, file_name)
                # field_name은 폴더 이름으로 설정
                extract_json_data(json_file_path, dir_name)

# 데이터프레임 생성 (field 컬럼 추가)
news_int_train = pd.DataFrame(data, columns=['sentences', 'source_site', 'write_date', 'field'])


# In[ ]:


# 데이터프레임을 CSV 파일로 저장
news_int_train.to_csv('data/preprocessed/news_int_train.csv', index=False, encoding='utf-8-sig')


# In[ ]:


# 'write_date' 컬럼을 날짜 형식으로 변환
news_int_train['write_date'] = pd.to_datetime(news_int_train['write_date'], format='%Y%m%d')

# 일별 기사 수 계산
news_counts_daily = news_int_train['write_date'].value_counts().sort_index()

# 가장 먼저 시작한 날짜 확인
earliest_date = news_int_train['write_date'].min()

# 바그래프 그리기
plt.figure(figsize=(12,6))
plt.bar(news_counts_daily.index, news_counts_daily.values)
plt.xlabel('Date')
plt.ylabel('Number of Internet News')
plt.title('Number of Internet News by Date')
plt.xticks(rotation=45)
plt.tight_layout()

# 그래프 출력
plt.show()

# 가장 먼저 시작한 날짜 출력
print(f'The earliest date is: {earliest_date}')


# ### 인터넷 기사 test 데이터 정제

# In[ ]:


# 폴더 경로 설정
root_dir = 'data/TL1'

# 추출할 데이터를 담을 리스트
data = []

# 각 json 파일을 열어서 필요한 데이터를 추출하는 함수
def extract_json_data(json_file_path, field_name):
    with open(json_file_path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
        # 각 named_entity에서 필요한 데이터를 추출
        for entity in json_data['named_entity']:
            # sentence 합치기
            sentences = ' '.join([content['sentence'] for content in entity['content']])
            source_site = entity['source_site']
            write_date = entity['write_date']
            # 데이터를 리스트에 추가, field_name도 포함
            data.append([sentences, source_site, write_date, field_name])

# 폴더 내 모든 json 파일을 반복하면서 데이터를 추출
for root, dirs, files in os.walk(root_dir):
    for dir_name in dirs:
        field_dir = os.path.join(root, dir_name)
        for file_name in os.listdir(field_dir):
            if file_name.endswith('.json'):
                json_file_path = os.path.join(field_dir, file_name)
                # field_name은 폴더 이름으로 설정
                extract_json_data(json_file_path, dir_name)

# 데이터프레임 생성 (field 컬럼 추가)
news_int_test = pd.DataFrame(data, columns=['sentences', 'source_site', 'write_date', 'field'])


# In[ ]:


# 데이터프레임을 CSV 파일로 저장
news_int_test.to_csv('data/preprocessed/news_int_test.csv', index=False, encoding='utf-8-sig')


# In[ ]:


# 'write_date' 컬럼을 날짜 형식으로 변환
news_int_train['write_date'] = pd.to_datetime(news_int_train['write_date'], format='%Y%m%d')

# 일별 기사 수 계산
news_counts_daily = news_int_train['write_date'].value_counts().sort_index()

# 가장 먼저 시작한 날짜 확인
earliest_date = news_int_train['write_date'].min()

# 바그래프 그리기
plt.figure(figsize=(12,6))
plt.bar(news_counts_daily.index, news_counts_daily.values)
plt.xlabel('Date')
plt.ylabel('Number of Internet News')
plt.title('Number of Internet News by Date')
plt.xticks(rotation=45)
plt.tight_layout()

# 그래프 출력
plt.show()

# 가장 먼저 시작한 날짜 출력
print(f'The earliest date is: {earliest_date}')


# # 하연악 여기까지만 돌려주며는 돼 <:3

# #### '경제' 폴더만 보자

# In[11]:


# '경제' 폴더 경로 설정
root_dir = 'data/경제'

# 추출할 데이터를 담을 리스트
data = []

# 각 json 파일을 열어서 필요한 데이터를 추출하는 함수
def extract_json_data(json_file_path, field_name):
    with open(json_file_path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
        # 각 named_entity에서 필요한 데이터를 추출
        for entity in json_data['named_entity']:
            # sentence 합치기
            sentences = ' '.join([content['sentence'] for content in entity['content']])
            source_site = entity['source_site']
            write_date = entity['write_date']
            # 데이터를 리스트에 추가, field_name도 포함
            data.append([sentences, source_site, write_date, field_name])

# '경제' 폴더 내 모든 json 파일을 반복하면서 데이터를 추출
for root, dirs, files in os.walk(root_dir):
    for file_name in files:
        if file_name.endswith('.json'):
            json_file_path = os.path.join(root, file_name)
            # '경제' 폴더만이므로 field_name은 '경제'로 고정
            extract_json_data(json_file_path, '경제')

# 데이터프레임 생성 (field 컬럼 추가)
df = pd.DataFrame(data, columns=['sentences', 'source_site', 'write_date', 'field'])


# In[12]:


df


# In[8]:


# 'write_date' 컬럼을 날짜 형식으로 변환
df['write_date'] = pd.to_datetime(df['write_date'], format='%Y-%m-%d')

# 일별 기사 수 계산
news_counts_daily = df['write_date'].value_counts().sort_index()

# 가장 먼저 시작한 날짜 확인
earliest_date = df['write_date'].min()

# 바그래프 그리기
plt.figure(figsize=(12,6))
plt.bar(news_counts_daily.index, news_counts_daily.values)
plt.xlabel('Date')
plt.ylabel('Number of Internet News')
plt.title('Number of Internet News by Date')
plt.xticks(rotation=45)
plt.tight_layout()

# 그래프 출력
plt.show()

# 가장 먼저 시작한 날짜 출력
print(f'The earliest date is: {earliest_date}')


# In[ ]:





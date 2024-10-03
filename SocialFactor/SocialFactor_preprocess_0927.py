#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd

import os
import json

import matplotlib.pyplot as plt


# ### 법률안보고서 데이터 정제

# In[2]:


law = pd.read_csv('data/law_summary_train.csv')
law.head()


# In[6]:


law = law[['id', 'date', 'terminology', 'title', 'committee', 'field', 'disposal']]
law.head()


# In[12]:


law = law[law['disposal']!='임기만료폐기']


# In[13]:


law


# In[15]:


# 'date' 컬럼을 일별로 변환
law['date'] = pd.to_datetime(law['date'])

# 일별 법안 수 계산
law_counts_daily = law['date'].value_counts().sort_index()

# 바그래프 그리기 (일별 법안 수)
plt.figure(figsize=(12,6))
plt.bar(law_counts_daily.index, law_counts_daily.values)
plt.xlabel('Date')
plt.ylabel('Number of Laws')
plt.title('Number of Laws by Date')
plt.xticks(rotation=45)
plt.tight_layout()

# 그래프 출력
plt.show()


# In[21]:


# 데이터프레임을 CSV 파일로 저장
law.to_csv('data/preprocessed/law.csv', index=False, encoding='utf-8-sig')


# ### 뉴스 스크립트 트레인 데이터 정제

# In[8]:


root_dir = 'data/news_script_train'

# 추출할 데이터를 담을 리스트
data = []

# 첫 번째 json 파일만 열기 위한 함수
def extract_json_data(json_file_path):
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        # 필요한 값 추출
        script_id = data['script']['id']
        press_field = data['script']['press_field']
        press_date = data['script']['press_date']
        keyword = data['script']['keyword']
        return script_id, press_field, press_date, keyword

# 폴더 내 폴더를 반복하면서 데이터 추출
for root, dirs, files in os.walk(root_dir):
    for dir_name in dirs:
        sub_dir = os.path.join(root, dir_name)
        for sub_root, sub_dirs, sub_files in os.walk(sub_dir):
            for file_name in sub_files:
                if file_name.endswith('.json'):
                    json_file_path = os.path.join(sub_root, file_name)
                    # 첫 번째 json 파일만 처리하고 나가기
                    script_id, press_field, press_date, keyword = extract_json_data(json_file_path)
                    data.append([script_id, press_field, press_date, keyword])
                    break

# 데이터프레임 생성
news_script = pd.DataFrame(data, columns=['id', 'press_field', 'press_date', 'keyword'])


# In[9]:


news_script_train


# In[22]:


# 데이터프레임을 CSV 파일로 저장
news_script.to_csv('data/preprocessed/news_script_train.csv', index=False, encoding='utf-8-sig')


# In[28]:


# 'press_date' 컬럼을 날짜 형식으로 변환
news_script['press_date'] = pd.to_datetime(news_script['press_date'], format='%Y%m%d')

# 일별 기사 수 계산
news_counts_daily = news_script['press_date'].value_counts().sort_index()

# 가장 먼저 시작한 날짜 확인
earliest_date = news_script['press_date'].min()

# 바그래프 그리기
plt.figure(figsize=(12,6))
plt.bar(news_counts_daily.index, news_counts_daily.values)
plt.xlabel('Date')
plt.ylabel('Number of News Scripts')
plt.title('Number of News Scripts by Date')
plt.xticks(rotation=45)
plt.tight_layout()

# 그래프 출력
plt.show()

# 가장 먼저 시작한 날짜 출력
print(f'The earliest date is: {earliest_date}')


# #### 문제점
# 1. 법률안은 2015년부터인데, 뉴스가 2018년부터 밖에 없음(이전 시점을 언제까지로 제한?)
# 2. 일자별로 고르지 않음
# 
# #### Test data 다시 합쳐볼까?

# In[17]:


root_dir = 'data/news_script_test'

# 추출할 데이터를 담을 리스트
data = []

# 첫 번째 json 파일만 열기 위한 함수
def extract_json_data(json_file_path):
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        # 필요한 값 추출
        script_id = data['script']['id']
        press_field = data['script']['press_field']
        press_date = data['script']['press_date']
        keyword = data['script']['keyword']
        return script_id, press_field, press_date, keyword

# 폴더 내 폴더를 반복하면서 데이터 추출
for root, dirs, files in os.walk(root_dir):
    for dir_name in dirs:
        sub_dir = os.path.join(root, dir_name)
        for sub_root, sub_dirs, sub_files in os.walk(sub_dir):
            for file_name in sub_files:
                if file_name.endswith('.json'):
                    json_file_path = os.path.join(sub_root, file_name)
                    # 첫 번째 json 파일만 처리하고 나가기
                    script_id, press_field, press_date, keyword = extract_json_data(json_file_path)
                    data.append([script_id, press_field, press_date, keyword])
                    break

# 데이터프레임 생성
news_script_rest = pd.DataFrame(data, columns=['id', 'press_field', 'press_date', 'keyword'])


# In[23]:


# 데이터프레임을 CSV 파일로 저장
news_script_rest.to_csv('data/preprocessed/news_script_test.csv', index=False, encoding='utf-8-sig')


# In[29]:


# 'press_date' 컬럼을 날짜 형식으로 변환
news_script_rest['press_date'] = pd.to_datetime(news_script_rest['press_date'], format='%Y%m%d')

# 일별 기사 수 계산
news_counts_daily = news_script_rest['press_date'].value_counts().sort_index()

# 가장 먼저 시작한 날짜 확인
earliest_date = news_script_rest['press_date'].min()

# 바그래프 그리기
plt.figure(figsize=(12,6))
plt.bar(news_counts_daily.index, news_counts_daily.values)
plt.xlabel('Date')
plt.ylabel('Number of News Scripts')
plt.title('Number of News Scripts by Date')
plt.xticks(rotation=45)
plt.tight_layout()

# 그래프 출력
plt.show()

# 가장 먼저 시작한 날짜 출력
print(f'The earliest date is: {earliest_date}')


# #### 여전히 문제...
# - 고르지 않음

# ### 뉴스 기사 데이터 정제

# In[19]:


# 폴더 경로 설정
root_dir = 'data/news_art_train'

# 추출할 데이터를 담을 리스트
data = []

# 각 json 파일을 열어서 필요한 데이터를 추출하는 함수
def extract_json_data(json_file_path):
    with open(json_file_path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
        for item in json_data['data']:
            doc_id = item['doc_id']
            doc_published = item['doc_published']
            doc_title = item['doc_title']
            doc_source = item['doc_source']
            doc_code = item['doc_class']['code']
            # 첫 번째 문단의 context만 추출
            context = item['paragraphs'][0]['context'] if item['paragraphs'] else None
            # 리스트에 데이터 추가
            data.append([doc_id, doc_published, doc_title, doc_source, doc_code, context])

# 폴더 내의 모든 json 파일을 반복하면서 데이터를 추출
for root, dirs, files in os.walk(root_dir):
    for file_name in files:
        if file_name.endswith('.json'):
            json_file_path = os.path.join(root, file_name)
            extract_json_data(json_file_path)

# 데이터프레임 생성
news_art_train = pd.DataFrame(data, columns=['doc_id', 'doc_published', 'doc_title', 'doc_source', 'code', 'context'])


# In[24]:


# 데이터프레임을 CSV 파일로 저장
news_art_train.to_csv('data/preprocessed/news_art_train.csv', index=False, encoding='utf-8-sig')


# In[20]:


news_art_train


# In[26]:


# 'doc_published' 컬럼을 날짜 형식으로 변환
news_art_train['doc_published'] = pd.to_datetime(news_art_train['doc_published'], format='%Y%m%d')

# 일별 뉴스 기사 수 계산
news_counts_daily = news_art_train['doc_published'].value_counts().sort_index()

# 가장 먼저 시작한 날짜 확인
earliest_date = news_art_train['doc_published'].min()

# 바그래프 그리기
plt.figure(figsize=(12,6))
plt.bar(news_counts_daily.index, news_counts_daily.values)
plt.xlabel('Date')
plt.ylabel('Number of News Articles')
plt.title('Number of News Articles by Date')
plt.xticks(rotation=45)
plt.tight_layout()

# 그래프 출력
plt.show()

# 가장 먼저 시작한 날짜 출력
print(f'The earliest date is: {earliest_date}')


# In[25]:


# Just in case...
# 폴더 경로 설정
root_dir = 'data/news_art_test'

# 추출할 데이터를 담을 리스트
data = []

# 각 json 파일을 열어서 필요한 데이터를 추출하는 함수
def extract_json_data(json_file_path):
    with open(json_file_path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
        for item in json_data['data']:
            doc_id = item['doc_id']
            doc_published = item['doc_published']
            doc_title = item['doc_title']
            doc_source = item['doc_source']
            doc_code = item['doc_class']['code']
            # 첫 번째 문단의 context만 추출
            context = item['paragraphs'][0]['context'] if item['paragraphs'] else None
            # 리스트에 데이터 추가
            data.append([doc_id, doc_published, doc_title, doc_source, doc_code, context])

# 폴더 내의 모든 json 파일을 반복하면서 데이터를 추출
for root, dirs, files in os.walk(root_dir):
    for file_name in files:
        if file_name.endswith('.json'):
            json_file_path = os.path.join(root, file_name)
            extract_json_data(json_file_path)

# 데이터프레임 생성
news_art_test = pd.DataFrame(data, columns=['doc_id', 'doc_published', 'doc_title', 'doc_source', 'code', 'context'])

# 데이터프레임을 CSV 파일로 저장
news_art_test.to_csv('data/preprocessed/news_art_test.csv', index=False, encoding='utf-8-sig')


# In[27]:


# 'doc_published' 컬럼을 날짜 형식으로 변환
news_art_test['doc_published'] = pd.to_datetime(news_art_test['doc_published'], format='%Y%m%d')

# 일별 뉴스 기사 수 계산
news_counts_daily = news_art_test['doc_published'].value_counts().sort_index()

# 가장 먼저 시작한 날짜 확인
earliest_date = news_art_test['doc_published'].min()

# 바그래프 그리기
plt.figure(figsize=(12,6))
plt.bar(news_counts_daily.index, news_counts_daily.values)
plt.xlabel('Date')
plt.ylabel('Number of News Articles')
plt.title('Number of News Articles by Date')
plt.xticks(rotation=45)
plt.tight_layout()

# 그래프 출력
plt.show()

# 가장 먼저 시작한 날짜 출력
print(f'The earliest date is: {earliest_date}')


# In[ ]:





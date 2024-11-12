#!/usr/bin/env python
# coding: utf-8

# In[10]:


import pandas as pd
from tqdm import tqdm
from datetime import timedelta
import bisect
import numpy as np
import re
import matplotlib.pyplot as plt


# In[1]:


import pandas as pd

t1 = pd.read_csv('data/assembly_bills_with_proposal_text1.csv')
t2 = pd.read_csv('data/assembly_bills_with_proposal_text2.csv')


# In[6]:


law = pd.concat([t1, t2], ignore_index=True)
law


# In[7]:


law['member_list'][0]


# In[16]:


law.to_csv('data/assembly_bills_with_proposal_text.csv')


# In[8]:


temp = law[0:5]


# In[20]:


temp


# In[28]:


# 텍스트에서 '발의의원 명단' 이후 텍스트를 추출하고 가공하는 함수
def extract_proposers(text):
    # '발의의원 명단' 이후 텍스트 추출
    match = re.search(r'발의의원 명단(.*)', text, re.DOTALL)
    if match:
        proposers_text = match.group(1)

        # 텍스트를 줄바꿈 기준으로 나누고 불필요한 부분을 제거
        proposers = proposers_text.split('\n')

        # 의원 이름과 당명을 추출하는 정규식 패턴
        pattern = r'([가-힣]+)\(([^)]+)\)'  # 한글 이름과 괄호 안의 당명

        proposers_list = []
        for proposer in proposers:
            # 각 항목에서 정규식 패턴으로 이름과 당명 추출
            match = re.search(pattern, proposer)
            if match:
                name = match.group(1)  # 이름
                party = match.group(2)  # 당명
                # 한자와 `/` 제거
                cleaned_party = re.sub(r'\/.*', '', party)  # `/` 이후와 한자 제거
                proposers_list.append(f"{name}({cleaned_party})")

        # 결과를 콤마로 구분된 리스트 형식으로 반환
        return ', '.join(proposers_list)
    return ""

# 'member_list' URL에서 텍스트 긁어오기
def get_member_list_text(url):
    try:
        # 요청 보내기
        response = requests.get(url)
        response.raise_for_status()  # HTTP 오류 발생 시 예외를 발생시킴
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # 텍스트 추출
        text = soup.get_text()
        
        # 발의의원 명단 텍스트 추출
        proposers_list = extract_proposers(text)
        return proposers_list
    except Exception as e:
        print(f"Error with URL {url}: {e}")
        return ""


# In[31]:


# 'member_list' 컬럼에 있는 URL로 텍스트 긁어오기
law.loc[:, 'member_list_text'] = None  # 새로운 컬럼 추가 (SettingWithCopyWarning 해결)
for idx, row in tqdm(law.iterrows(), total=law.shape[0]):
    url = row['member_list']
    if isinstance(url, str) and url.startswith('http'):
        law.loc[idx, 'member_list_text'] = get_member_list_text(url)


# In[32]:


law


# In[33]:


law.to_csv('data/assembly_bills_with_proposal_ppl.csv')


# ## 1-1. 직접적 요인 빈도수/기간 추출

# In[2]:


import pandas as pd
from tqdm import tqdm
from datetime import datetime, timedelta


# In[3]:


law = pd.read_csv('data/law_summary_train.csv', index_col=0)
law


# In[5]:


law['direct_freq'] = freq['frequency_probability']
law


# In[6]:


# 시작일과 종료일 설정
start_date = '2015-01-12'
end_date = '2022-07-01'  # 2022-07-01 전날까지 포함하려면 종료일을 2022-07-01로 설정하고 미만(<) 연산 사용

# 조건에 맞는 데이터 필터링
law = law[
    (law['date'] >= start_date) & (law['date'] < end_date)
]

law


# In[12]:


law.to_csv('법률안검토보고서_직접빈도까지.csv', index=False)


# In[32]:


law = law.drop(columns=['median_date_difference'])
law


# In[33]:


law_min = law[0:20000]
law_min


# In[36]:


law_jong = law[20000:]
law_jong


# In[35]:


law_min.to_csv('minseok_ping.csv',index=False)


# In[37]:


law_jong.to_csv('jongrak_ping.csv',index=False)


# In[39]:


law.to_csv('hayeon_ping.csv', index=False)


# ## 1-2. 직접적 최근 등장 기간

# In[7]:


law['date'] = pd.to_datetime(law['date'])


# In[8]:


news = pd.read_csv('data/news_title_cleaned_train.csv')
news = news.sort_values(by='write_date')
news = news.reset_index()
news['write_date'] = pd.to_datetime(news['write_date'])
news


# In[21]:


# 중앙 날짜와의 차이를 계산하는 함수
def calculate_median_gap(terminology, law_date, filtered_news):
    keywords = terminology.split(', ')
    if not keywords:
        return None

    # 키워드를 정규 표현식으로 인식하지 않게 처리
    keyword_pattern = '|'.join([re.escape(keyword) for keyword in keywords])

    # 키워드가 포함된 뉴스만 필터링
    matching_news = filtered_news[filtered_news['sentence'].str.contains(keyword_pattern, na=False)]

    if matching_news.empty:
        return None

    # 날짜 차이를 계산
    date_differences = (law_date - matching_news['write_date']).dt.days

    # 1년 이내의 차이만 고려
    date_differences = date_differences[(date_differences > 0) & (date_differences <= 365)]

    if date_differences.empty:
        return None

    # 중앙값 계산
    return np.median(date_differences)

# tqdm을 사용하여 진행 상황을 모니터링
law['median_date_difference'] = None

for idx, row in tqdm(law.iterrows(), total=law.shape[0], desc="Processing law keywords"):
    # 각 법률안의 date에서 1년 전까지의 뉴스만 필터링
    start_date = row['date'] - timedelta(days=365)
    end_date = row['date']
    
    # 해당 기간의 뉴스 데이터 필터링 (필터링된 뉴스만 사용)
    filtered_news = news[(news['write_date'] >= start_date) & (news['write_date'] <= end_date)]
    
    # 키워드와 뉴스 날짜 차이 중앙값을 계산
    median_diff = calculate_median_gap(row['terminology'], row['date'], filtered_news)
    
    # 결과 저장
    law.at[idx, 'median_date_difference'] = median_diff



# ## 2-1. 간접적 빈도

# In[24]:


# 키워드 빈도 계산 함수 수정
def calculate_keyword_frequency(keywords, filtered_news):
    if not keywords:
        return 0

    # 키워드를 정규식 패턴으로 생성
    # 각 키워드를 단어 경계로 감싸 정확한 단어만 매칭하도록 설정할 수 있음 (옵션)
    # pattern = '|'.join(r'\b{}\b'.format(re.escape(keyword)) for keyword in keywords)
    pattern = '|'.join(map(re.escape, keywords))
    
    # 문장에서 키워드가 포함된 문장의 수를 계산
    keyword_count = filtered_news['sentence'].str.contains(pattern, regex=True).sum()
    
    return keyword_count

# 빈도 확률 계산
law['indirect_freq'] = 0.0

for idx, row in tqdm(law.iterrows(), total=law.shape[0], desc="Processing keywords"):
    # 각 항목의 date에서 1년 전까지의 뉴스만 필터링
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
        law.at[idx, 'indirect_freq'] = keyword_count / total_news_count
    else:
        law.at[idx, 'indirect_freq'] = 0


# In[25]:


law


# In[ ]:





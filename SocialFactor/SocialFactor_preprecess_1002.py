#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
from tqdm import tqdm
from datetime import timedelta
import bisect
import numpy as np
import re
import matplotlib.pyplot as plt


# ## 1. 직접적 빈도 확률

# In[3]:


law = pd.read_csv('data/law_with_direct_freq.csv')
law


# In[9]:


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


# In[8]:


law_selected = law[law['disposal']!='임기만료폐기']
law_selected = law_selected.reset_index(drop=True)
law_selected


# In[14]:


import matplotlib.pyplot as plt

# 히스토그램을 그리기 위한 데이터
frequency_data = law_selected['frequency_probability']

# 히스토그램 그리기
plt.figure(figsize=(8, 6))
plt.hist(frequency_data, bins=2000, color='skyblue', edgecolor='black')
plt.title('Distribution of Frequency Probability of Selected LAw')
plt.xlabel('Frequency Probability')
plt.ylabel('Frequency')
plt.xlim(0, 0.02)  # x축 범위를 0 ~ 0.01로 제한
plt.grid(True)

# 히스토그램 표시
plt.show()


# #### EDA 결과
# - 임기만료폐기를 제외한 결과, 과거에 나왔던 빈도 확률이 확연히 줄어듦

# In[15]:


law_selected = pd.read_csv('data/law_selected_with_direct_all', index_col=0)
law_selected


# In[33]:


# 시작일과 종료일 설정
start_date = '2015-01-12'
end_date = '2022-07-01'  # 2022-07-01 전날까지 포함하려면 종료일을 2022-07-01로 설정하고 미만(<) 연산 사용

# 조건에 맞는 데이터 필터링
law_selected = law_selected[
    (law_selected['date'] >= start_date) & (law_selected['date'] < end_date)
]

law_selected


# In[18]:


# 최소 날짜와 최대 날짜 추출
min_date = law_selected['date'].min()
max_date = law_selected['date'].max()

# 날짜 범위 출력
print(f"데이터의 날짜 범위는 {min_date.date()}부터 {max_date.date()}까지입니다.")



# In[34]:


law_selected = law_selected.reset_index()
law_selected.to_csv('data/law_selected_with_direct_all.csv')


# In[35]:


import pandas as pd

# 구간 설정 및 레이블
bins = [0, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1]  # 구간 설정
labels = ['0-0.0001', '0.0001-0.0005', '0.0005-0.001', '0.001-0.005', '0.005-0.01', '0.01-0.05', '0.05-0.1', '0.1-0.5', '0.5-1']

# frequency_probability에 구간을 적용
law_selected['freq_bin'] = pd.cut(law_selected['frequency_probability'], bins=bins, labels=labels, include_lowest=True)

# 각 구간별 데이터 수 계산
freq_counts = law_selected['freq_bin'].value_counts().sort_index()

# 결과 출력
print(freq_counts)


# In[36]:


# frequency_probability 값이 0인 개수 계산
zero_count = (law_selected['frequency_probability'] == 0).sum()

# 결과 출력
print(f"Frequency probability equal to 0: {zero_count}")


# In[38]:


import matplotlib.pyplot as plt

# frequency_probability 값이 0이 아닌 데이터만 선택
filtered_data = law_selected[law_selected['frequency_probability'] != 0]['frequency_probability']

# 히스토그램 그리기
plt.figure(figsize=(8, 6))
plt.hist(filtered_data, bins=2000, color='skyblue', edgecolor='black')
plt.title('Distribution of Frequency Probability (excluding 0)')
plt.xlabel('Frequency Probability')
plt.ylabel('Count')
plt.xlim(0, 0.01)  # x축 범위를 0 ~ 0.01로 제한

plt.grid(True)

# 히스토그램 표시
plt.show()


# In[39]:


import matplotlib.pyplot as plt

# frequency_probability 값이 0.0001 미만인 값들을 제외
filtered_data = law_selected[law_selected['frequency_probability'] >= 0.0001]['frequency_probability']

# 히스토그램 그리기
plt.figure(figsize=(8, 6))
plt.hist(filtered_data, bins=2000, color='skyblue', edgecolor='black')
plt.title('Distribution of Frequency Probability (excluding 0-0.0001)')
plt.xlabel('Frequency Probability')
plt.ylabel('Count')
plt.xlim(0, 0.01)  # x축 범위를 0 ~ 0.01로 제한

plt.grid(True)

# 히스토그램 표시
plt.show()


# ## 2. 직접적 최근 등장 기간

# In[18]:


law_selected['date'] = pd.to_datetime(law_selected['date'])


# In[7]:


law_selected = pd.read_csv('data/law_selected_with_direct_freq.csv')
law_selected['date'] = pd.to_datetime(law_selected['date'])
law_selected


# In[4]:


news = pd.read_csv('data/news_title_cleaned_train.csv')
news = news.sort_values(by='write_date')
news = news.reset_index()
news['write_date'] = pd.to_datetime(news['write_date'])
news


# In[24]:


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
law_selected['median_date_difference'] = None

for idx, row in tqdm(law_selected.iterrows(), total=law_selected.shape[0], desc="Processing law keywords"):
    # 각 법률안의 date에서 1년 전까지의 뉴스만 필터링
    start_date = row['date'] - timedelta(days=365)
    end_date = row['date']
    
    # 해당 기간의 뉴스 데이터 필터링 (필터링된 뉴스만 사용)
    filtered_news = news[(news['write_date'] >= start_date) & (news['write_date'] <= end_date)]
    
    # 키워드와 뉴스 날짜 차이 중앙값을 계산
    median_diff = calculate_median_gap(row['terminology'], row['date'], filtered_news)
    
    # 결과 저장
    law_selected.at[idx, 'median_date_difference'] = median_diff



# In[26]:


law_selected


# In[27]:


import matplotlib.pyplot as plt

# first_date_difference와 last_date_difference 칼럼 삭제
law_selected = law_selected.drop(columns=['first_date_difference', 'last_date_difference'])
law_selected


# In[28]:


law_selected.to_csv('data/law_selected_with_direct_all')


# In[29]:


# 'median_date_difference' 칼럼에서 None 값 세기
none_count = law_selected['median_date_difference'].isna().sum()

# None 값 출력
print(f"None values in 'median_date_difference': {none_count}")

# 16240개 중 3365개


# In[30]:


# None 값을 제외한 'median_date_difference' 값만 선택
filtered_data = law_selected['median_date_difference'].dropna()

# 히스토그램 그리기
plt.figure(figsize=(8, 6))
plt.hist(filtered_data, bins=50, color='skyblue', edgecolor='black')
plt.title('Distribution of Median Date Difference of Selected Law')
plt.xlabel('Median Date Difference (days)')
plt.ylabel('Frequency')
plt.grid(True)

# 히스토그램 표시
plt.show()


# #### 1년 슬라이싱 샘플

# In[14]:


# 뉴스 데이터: 2014-01-12부터 1년
news_start_date = pd.to_datetime('2014-01-12')
news_end_date = news_start_date + timedelta(days=365)
fil_news = news[(news['write_date'] >= news_start_date) & (news['write_date'] <= news_end_date)]

# 법률 데이터: 2015-01-12부터 1년
law_start_date = pd.to_datetime('2015-01-12')
law_end_date = law_start_date + timedelta(days=365)
fil_law = law_selected[(law_selected['date'] >= law_start_date) & (law_selected['date'] <= law_end_date)]

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
fil_law['median_date_difference'] = None

for idx, row in tqdm(fil_law.iterrows(), total=fil_law.shape[0], desc="Processing law keywords"):
    # 각 법률안의 date에서 1년 전까지의 뉴스만 필터링
    start_date = row['date'] - timedelta(days=365)
    end_date = row['date']
    
    # 해당 기간의 뉴스 데이터 필터링 (필터링된 뉴스만 사용)
    filtered_news = fil_news[(fil_news['write_date'] >= start_date) & (fil_news['write_date'] <= end_date)]
    
    # 키워드와 뉴스 날짜 차이 중앙값을 계산
    median_diff = calculate_median_gap(row['terminology'], row['date'], filtered_news)
    
    # 결과 저장
    fil_law.at[idx, 'median_date_difference'] = median_diff

# 결과 확인
print(fil_law[['id', 'terminology', 'median_date_difference']])


# In[16]:


fil_law


# In[23]:


import matplotlib.pyplot as plt

# NaN 값을 제외한 'median_date_difference' 값만 선택
frequency_data = fil_law['median_date_difference'].dropna()

# 히스토그램 그리기
plt.figure(figsize=(8, 6))
plt.hist(frequency_data, bins=2000, color='skyblue', edgecolor='black')
plt.title('Distribution of Median Date Difference of Selected Law')
plt.xlabel('Median Date Difference (days)')
plt.ylabel('Frequency')
plt.xlim(0, 365)  # x축 범위를 0 ~ 0.02로 제한
plt.grid(True)

# 히스토그램 표시
plt.show()


# ## 3. 간접적 빈도 확률

# In[5]:


review = pd.read_csv('data/review.csv', index_col=0)
review


# In[7]:


review_selected = review[review['result']!='임기만료폐기']


# In[19]:


# 최소 날짜와 최대 날짜 추출
min_date = review_selected['date'].min()
max_date = review_selected['date'].max()

# 날짜 범위 출력
print(f"데이터의 날짜 범위는 {min_date.date()}부터 {max_date.date()}까지입니다.")



# In[8]:


import ast

# 문자열을 딕셔너리로 변환하는 함수 정의
def str_to_dict(x):
    if isinstance(x, str):
        try:
            return ast.literal_eval(x)
        except:
            # 변환에 실패한 경우 빈 딕셔너리 반환 또는 다른 처리
            return {}
    elif isinstance(x, dict):
        return x
    else:
        # 다른 타입의 경우에도 빈 딕셔너리 반환 또는 다른 처리
        return {}
# 문자열을 딕셔너리로 변환하여 'token_freq' 열 갱신
review_selected['token_freq'] = review_selected['token_freq'].apply(str_to_dict)

# 단어들만 추출하여 'terminology' 열 생성
review_selected['terminology'] = review_selected['token_freq'].apply(lambda x: list(x.keys()))


# In[12]:


# 날짜 형식으로 변환하기 위해 pandas 모듈 임포트
import pandas as pd

# 'date' 열을 datetime 형식으로 변환
review_selected['date'] = pd.to_datetime(review_selected['date'])

# 시작일과 종료일 설정
start_date = '2015-01-12'
end_date = '2022-07-01'  # 2022-07-01 전날까지 포함하려면 종료일을 2022-07-01로 설정하고 미만(<) 연산 사용

# 조건에 맞는 데이터 필터링
review_selected = review_selected[
    (review_selected['date'] >= start_date) & (review_selected['date'] < end_date)
]


# In[13]:


review_selected


# In[21]:


news = pd.read_csv('data/news_title_cleaned_train.csv')
news


# In[23]:


# import pandas as pd
# from tqdm import tqdm
# from datetime import timedelta
# import re

# 날짜 형식 변환
news['write_date'] = pd.to_datetime(news['write_date'])
review_selected['date'] = pd.to_datetime(review_selected['date'])

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
review_selected['frequency_probability'] = 0.0

for idx, row in tqdm(review_selected.iterrows(), total=review_selected.shape[0], desc="Processing keywords"):
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
        review_selected.at[idx, 'frequency_probability'] = keyword_count / total_news_count
    else:
        review_selected.at[idx, 'frequency_probability'] = 0


# In[24]:


review_selected.to_csv('data/law_with_indirect_freq.csv', index=False)


# In[25]:


review_selected


# In[26]:


import matplotlib.pyplot as plt

# 히스토그램을 그리기 위한 데이터
frequency_data = review_selected['frequency_probability']

# 히스토그램 그리기
plt.figure(figsize=(8, 6))
plt.hist(frequency_data, bins=2000, color='skyblue', edgecolor='black')
plt.title('Distribution of Frequency Probability of Indirect Law')
plt.xlabel('Frequency Probability')
plt.ylabel('Frequency')
plt.xlim(0, 1)  # x축 범위를 0 ~ 0.01로 제한
plt.grid(True)

# 히스토그램 표시
plt.show()


# ### Merge

# In[58]:


review = pd.read_csv('data/law_with_indirect_freq.csv')
# review = review[['text', 'frequency_probability']]
review


# In[60]:


# 중복된 text 값 제거 (첫 번째 행만 남김)
review_uni = review.drop_duplicates(subset='text', keep='first')
review_uni


# In[61]:


review_final = pd.read_csv('data/review_final.csv', index_col=0)
# review_final = review_final[['id','text']]
review_final


# In[62]:


# 중복된 text 값 제거 (첫 번째 행만 남김)
review_final_uni = review_final.drop_duplicates(subset='text', keep='first')
review_final_uni


# In[63]:


# Merging DataFrames on 'text'
merged_df = pd.merge(review_uni, review_final_uni, on='text', how='inner')
merged_df


# In[64]:


merged_df.rename(columns={
    'frequency_probability': 'indirect_freq'
}, inplace=True)
merged_df


# In[54]:


law = pd.read_csv('data/law_selected_with_direct_all.csv', index_col=0)
law


# In[94]:


fin1 = merged_df.drop_duplicates(subset=['id', 'terminology_x'])
fin1 = fin1[['id', 'date_x', 'terminology_y', 'indirect_freq', 'law_x', 'text', 'result_x']]
fin1.rename(columns={
    'terminology_y': 'terminology',
    'law_x':'field',
    'date_x':'date',
    'result_x' : 'result'
}, inplace=True)
fin1


# In[95]:


fin2 = law.drop_duplicates(subset=['id', 'terminology'])
fin2.rename(columns={
    'frequency_probability': 'direct_freq'
}, inplace=True)
fin2 = fin2[['id', 'field', 'date', 'terminology', 'direct_freq', 'median_date_difference']]
fin2


# In[107]:


# Inner merge on id, date, and terminology
final = pd.merge(fin1, fin2, on=['id','date','terminology'], how='inner')
final


# In[98]:


final.to_csv('data/social_law.csv', index=False)


# In[ ]:





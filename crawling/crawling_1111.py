#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from tqdm import tqdm
from datetime import timedelta
import re


# In[1]:


import pandas as pd

d1 = pd.read_csv('data/law_min_res.csv')
d2 = pd.read_csv('data/law_jong_res.csv')

law = pd.concat([d1, d2], ignore_index=True)
law


# In[2]:


ha = pd.read_csv('data/law_ha_res.csv')
ha


# In[3]:


law['indirect_freq'] = ha['indirect_freq']
law


# In[4]:


# 특정 칼럼 이름 변경
law.rename(columns={"direct_freq": "시의성_직접_빈도", "median_date_difference": "시의성_직접_기간", "indirect_freq":"시의성_간접_빈도"}, inplace=True)

law


# ## 시각화

# In[6]:


# 시의성_직접_빈도

import matplotlib.pyplot as plt

# 히스토그램을 그리기 위한 데이터
frequency_data = law['시의성_직접_빈도']

# 히스토그램 그리기
plt.figure(figsize=(8, 6))
plt.hist(frequency_data, bins=10000, color='skyblue', edgecolor='black')
plt.title('Distribution of Frequency Probability')
plt.xlabel('Frequency Probability')
plt.ylabel('Frequency')
plt.xlim(0, 0.02)
plt.grid(True)

# 히스토그램 표시
plt.show()


# In[7]:


# None 값을 제외한 'median_date_difference' 값만 선택
filtered_data = law['시의성_직접_기간'].dropna()

# 히스토그램 그리기
plt.figure(figsize=(8, 6))
plt.hist(filtered_data, bins=50, color='skyblue', edgecolor='black')
plt.title('Distribution of Median Date Difference of Selected Law')
plt.xlabel('Median Date Difference (days)')
plt.ylabel('Frequency')
plt.grid(True)

# 히스토그램 표시
plt.show()


# In[46]:


law_with_term['시의성_간접_빈도_2']


# In[47]:


import matplotlib.pyplot as plt

# 히스토그램을 그리기 위한 데이터
frequency_data = law_with_term['시의성_간접_빈도_2']

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


# In[33]:


review = pd.read_csv('data/review.csv', index_col=0)
review


# In[34]:


review = review[['text', 'token_freq']].drop_duplicates()
review


# In[37]:


# review의 text와 law의 paragraph가 매칭되는 경우만 필터링하여 '종락_단어'를 law에 추가
law_with_term = law.merge(review[['text', 'terminology']], left_on="paragraph", right_on="text", how="left")

# terminology 칼럼을 '종락_단어'로 이름 변경
law_with_term.rename(columns={"terminology": "종락_단어"}, inplace=True)

# text 칼럼 삭제 (optional)
law_with_term.drop(columns=["text"], inplace=True)

law_with_term


# In[46]:


law = law_with_term[['date','bill_id','title', 'terminology_y']]
law.to_csv('data/간접_도전.csv', index=False)


# In[44]:


law_with_term.to_csv('data/시의성_데이터.csv', index=False)


# In[41]:


len(law_with_term['bill_id'].unique())


# In[2]:


news = pd.read_csv('data/news_title_cleaned_train.csv')
news = news.sort_values(by='write_date')
news = news.reset_index()
news['write_date'] = pd.to_datetime(news['write_date'])
news


# In[54]:


law['terminology_x'][0]


# In[4]:


# 날짜 형식 변환
news['write_date'] = pd.to_datetime(news['write_date'])
law_with_term['date'] = pd.to_datetime(law_with_term['date'])

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
law_with_term['시의성_간접_빈도_2'] = 0.0

for idx, row in tqdm(law_with_term.iterrows(), total=law_with_term.shape[0], desc="Processing keywords"):
    # 각 항목의 date에서 1년 전까지의 뉴스만 필터링
    start_date = row['date'] - timedelta(days=365)
    end_date = row['date']
    
    # 해당 기간의 뉴스 데이터 필터링
    filtered_news = news[(news['write_date'] >= start_date) & (news['write_date'] <= end_date)]
    
    # 해당 기간의 뉴스 개수
    total_news_count = len(filtered_news)
    
    # terminology에 포함된 키워드 빈도 계산
    keyword_count = calculate_keyword_frequency(row['terminology_y'], filtered_news)
    
    # 빈도 확률 계산
    if total_news_count > 0:
        law_with_term.at[idx, '시의성_간접_빈도_2'] = keyword_count / total_news_count
    else:
        law_with_term.at[idx, '시의성_간접_빈도_2'] = 0


# In[5]:


law_with_term


# In[51]:


law = pd.read_csv('data/시의성_데이터.csv')
law['bill_id'].nunique()


# ## Merge

# In[18]:


ha1 = pd.read_csv('data/law_sub_par_variables.csv')
ha1


# In[40]:


ha2 = pd.read_csv('data/하연_감정분석.csv')
ha2


# In[42]:


ha1['average_sentiment'] = ha2['average_sentiment']
ha1


# In[43]:


ha1.to_csv('data/하연_전송.csv', index=False)


# In[6]:


law4 = law_with_term


# In[24]:


law4 = law4[['date', 'bill_id', 'title', 'terminology_y', '시의성_간접_빈도_2']]
law4


# In[25]:


law3 = pd.read_csv('data/law3_result.csv', index_col = 0)
law1 = pd.read_csv('data/law1_result.csv', index_col = 0)


# In[10]:


law2 = pd.read_csv('data/시의의의성.csv')


# In[26]:


law = pd.concat([law1, law2, law3, law4], ignore_index=True)
law


# In[27]:


law['bill_id'].nunique()


# In[36]:


law_ind


# In[41]:


law_with_term = pd.read_csv('data/시의성_데이터.csv')
law_with_term


# In[45]:


law_with_term.info()


# In[31]:


law_with_term['시의성_간접_빈도_2'] = law_ind['시의성_간접_빈도_2']


# In[35]:


law_with_term.to_csv('law_with_news.csv', index=False)


# In[40]:


law_with_term


# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# ## 0. 데이터 불러오기

# In[71]:


import pandas as pd
import numpy as np


# In[72]:


main = pd.read_csv('/Users/hayoun/Desktop/K-DS/minute_data_preprocessed/main_minute_ordered.csv', index_col = 0)
parliament_wrap = pd.read_csv('/Users/hayoun/Desktop/K-DS/minute_data_preprocessed/all_minute_wrapped.csv', index_col = 0)
sub_wrap = pd.read_csv('/Users/hayoun/Desktop/K-DS/minute_data_preprocessed/sub_minute_wrapped.csv', index_col = 0)
law_df = pd.read_csv('/Users/hayoun/Desktop/K-DS/rawdata/law_train_filtered.csv', index_col = 0) # 임기만료 폐기 제외; 날짜 범위: 2015-01-12 00:00:00 ~ 2023-07-27 00:00:00


# In[73]:


law_parliament_df = pd.read_csv('/Users/hayoun/Desktop/K-DS/law_parliament_df.csv')
law_sub_parliament_df = pd.read_csv('/Users/hayoun/Desktop/K-DS/law_sub_parliament_df.csv')

law_parliament_df_unique = pd.read_csv('/Users/hayoun/Desktop/K-DS/law_parliament_df_unique.csv')
law_sub_parliament_df_unique = pd.read_csv('/Users/hayoun/Desktop/K-DS/law_sub_parliament_df_unique.csv')


# ### 0. 데이터 형태 검토

# In[74]:


law_sub_parliament_df_unique['bill_id'].nunique()


# In[75]:


# law_parliament_df.head()
law_sub_parliament_df.info()


# In[5]:


law_df = law_df.sort_values(by = ['id', 'bill_id'])


# In[23]:


# # 'bill_id', 'date', 'id', 'title'가 동일한 경우 중복 제거
# law_parliament_df_unique = law_parliament_df.drop_duplicates(subset=['bill_id', 'date', 'id', 'title']).reset_index(drop = True)

# law_parliament_df_unique.info()
# # 결과 확인
# law_parliament_df_unique.head()

# 'bill_id', 'date', 'id', 'title'가 동일한 경우 중복 제거
law_sub_parliament_df_unique = law_sub_parliament_df.drop_duplicates(subset=['bill_id', 'date', 'id', 'title']).reset_index(drop = True)

law_sub_parliament_df_unique.info()
# 결과 확인
law_sub_parliament_df_unique.head()


# In[80]:


law_sub_parliament_df_unique['directly_related'].value_counts()


# In[ ]:


# law_parliament_df_unique.to_csv('/Users/hayoun/Desktop/K-DS/law_parliament_df_unique.csv', index = False)
# law_sub_parliament_df_unique.to_csv('/Users/hayoun/Desktop/K-DS/law_sub_parliament_df_unique.csv', index = False)


# ## 1. 유사도 점수

# ### 1-1. law_df: bill_id 등 기준으로 병합

# In[14]:


law_df.info()


# In[17]:


# 공동 키를 기준으로 `ext_summary` 병합
merged_law_df_full = (
    law_df.groupby(['id', 'bill_id', 'title', 'date'], group_keys=False)
    .apply(lambda x: pd.Series({
        'committee': x['committee'].iloc[0],
        'field': x['field'].iloc[0],
        'enactment': x['enactment'].iloc[0],
        'amendment': x['amendment'].iloc[0],
        'proposer': x['proposer'].iloc[0],
        'advisor': x['advisor'].iloc[0],
        'paragraph_merged': ' '.join(x['paragraph']),
        'ext_summary_merged': ' '.join(x['ext_summary']),
        'gen_summary_merged': ' '.join(x['gen_summary']),
        'terminology_merged': ' '.join(x['terminology']),
        'disposal': x['disposal'].iloc[0]
    }))
    .reset_index()
)

# 필요한 컬럼들만 선택하여 `merged_law_df` 완성
merged_law_df_full.head()


# In[ ]:


# merged_law_df_full.to_csv('/Users/hayoun/Desktop/K-DS/merged_law_df_full.csv', index = False)


# ### 1-2. 유사도 분석

# In[19]:


law_sub_parliament_df_unique.head()


# In[27]:


sub_wrap.head()


# In[61]:


sub_wrap['context_full'].iloc[0]


# In[62]:


import ast

# 문자열을 리스트로 변환하는 함수 정의
def convert_text_to_list(text):
    try:
        # ast.literal_eval을 사용하여 문자열을 리스트로 변환
        return ast.literal_eval(text)
    except:
        # 변환에 실패할 경우 빈 리스트 반환
        return []

# context_full 칼럼을 리스트 형식으로 변환
sub_wrap['context_full'] = sub_wrap['context_full'].apply(convert_text_to_list)

# 결과 확인
sub_wrap['context_full'].head()


# In[48]:


# 문자열을 리스트로 변경
import ast

# 문자열을 리스트로 변환하는 함수 정의
def convert_to_list(text):
    try:
        # ast.literal_eval을 사용해 문자열을 실제 리스트로 변환
        return ast.literal_eval(text)
    except:
        # 변환에 실패할 경우 빈 리스트 반환
        return []

# related_agenda_index 칼럼을 리스트 형식으로 변환
law_sub_parliament_df_unique['related_agenda_index'] = law_sub_parliament_df_unique['related_agenda_index'].apply(convert_to_list)

# 결과 확인
law_sub_parliament_df_unique['related_agenda_index'].head()


# In[63]:


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# 유사도 점수를 계산하는 함수 정의
def calculate_similarity(text1, text2):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([text1, text2])
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    return similarity[0][0]

# 결과를 저장할 빈 리스트 생성
similarity_data = []

# law_sub_parliament_df_unique의 각 행을 순회
for _, row in law_sub_parliament_df_unique.iterrows():
    # 각 행의 ['bill_id', 'date', 'id', 'title']과 같은 merged_law_df 행 찾기
    merged_row = merged_law_df[
        (merged_law_df['bill_id'] == row['bill_id']) &
        (merged_law_df['date'] == row['date']) &
        (merged_law_df['id'] == row['id']) &
        (merged_law_df['title'] == row['title'])
    ]

    if not merged_row.empty:
        # ext_summary_merged 텍스트 가져오기
        ext_summary_text = merged_row.iloc[0]['ext_summary_merged']
        
        # 각 행의 related_agenda_index 리스트에서 agenda_index와 같은 sub_wrap의 행 접근
        similarity_scores = []
        
        for agenda_index in row['related_agenda_index']:
            # sub_wrap에서 관련된 agenda_index와 같은 행 찾기
            sub_wrap_row = sub_wrap[sub_wrap['agenda_index'] == agenda_index]
            
            if not sub_wrap_row.empty:
                # 'context_full' 리스트의 텍스트 합치기
                context_text = ' '.join(sub_wrap_row.iloc[0]['context_full'])
                
                # ext_summary_merged와 context_full 텍스트의 유사도 계산
                similarity_score = calculate_similarity(ext_summary_text, context_text)
                
                # 유사도 점수를 리스트에 추가
                similarity_scores.append(similarity_score)

        # similarity_score_list와 평균 similarity_score 계산
        similarity_score_list = similarity_scores
        similarity_score = np.mean(similarity_scores) if similarity_scores else 0
        
        # 결과를 저장
        similarity_data.append({
            'bill_id': row['bill_id'],
            'date': row['date'],
            'id': row['id'],
            'title': row['title'],
            'directly_related': row['directly_related'],
            'related_agenda_index': row['related_agenda_index'],
            'unique_agendas': row['unique_agendas'],
            'similarity_score_list': similarity_score_list,
            'similarity_score': similarity_score
        })

# 결과를 데이터프레임으로 변환
law_sub_par_variables = pd.DataFrame(similarity_data)

# 결과 확인
law_sub_par_variables.head()


# In[64]:


import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# Set the font to a font that supports Korean
plt.rcParams['font.family'] = 'AppleGothic'  # or 'Malgun Gothic' if that's installed

# Optionally, resolve the minus sign issue
plt.rcParams['axes.unicode_minus'] = False


# In[65]:


# 기초 통계량 계산
similarity_score_range = (law_sub_par_variables['similarity_score'].min(), law_sub_par_variables['similarity_score'].max())
similarity_score_mean = law_sub_par_variables['similarity_score'].mean()
similarity_score_distribution = law_sub_par_variables['similarity_score'].describe()

# 분포 그래프 생성
plt.figure(figsize=(10, 6))
plt.hist(law_sub_par_variables['similarity_score'], bins=20, edgecolor='black', alpha=0.7)
plt.title('Distribution of Similarity Scores')
plt.xlabel('Similarity Score')
plt.ylabel('Frequency')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

similarity_score_range, similarity_score_mean, similarity_score_distribution


# In[66]:


from sklearn.preprocessing import MinMaxScaler

# MinMaxScaler 초기화 (0~1 스케일)
scaler = MinMaxScaler(feature_range=(0, 1))

# similarity_score 칼럼 스케일링 적용
law_sub_par_variables['scaled_similarity_score'] = scaler.fit_transform(
    law_sub_par_variables[['similarity_score']]
)

# 스케일링 결과 확인
law_sub_par_variables[['similarity_score', 'scaled_similarity_score']].describe()


# In[70]:


law_sub_par_variables.head()


# In[ ]:


# law_sub_par_variables.to_csv('/Users/hayoun/Desktop/K-DS/law_sub_par_variables.csv', index = False)


# ## 2. 질의응답 셋의 개수(기각)

# In[ ]:


# 'num_QA_sets' 컬럼 추가
law_sub_parliament_df_unique['num_QA_sets'] = law_sub_parliament_df_unique['related_agenda_index'].apply(len)
law_sub_parliament_df_unique.head()


# In[31]:


law_sub_parliament_df_unique['num_QA_sets'].unique()


# In[ ]:





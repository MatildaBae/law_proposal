#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
from tqdm import tqdm
from datetime import timedelta
import bisect
import numpy as np
import re
import matplotlib.pyplot as plt


# In[56]:


law = pd.read_csv('data/social_law.csv')
law


# In[8]:


law.rename(columns={
    'field_x': 'field',
    'median_date_difference' : 'direct_median'
}, inplace=True)

law


# In[9]:


# Creating non_shown column as a categorical factor
law['non_shown'] = pd.Categorical((law['direct_freq'] == 0).astype(int).map({0: '0', 1: '1'}))
law


# In[12]:


law = law[['id', 'date', 'field', 'terminology', 'text', 'direct_freq', 'direct_median', 'indirect_freq', 'non_shown', 'result']]


# In[13]:


law


# In[16]:


law.to_csv('data/social_law.csv', index=False)


# ## 1. 직접적 빈도 확률

# In[15]:


import matplotlib.pyplot as plt

# 히스토그램을 그리기 위한 데이터
frequency_data = law['direct_freq']

# 히스토그램 그리기
plt.figure(figsize=(8, 6))
plt.hist(frequency_data, bins=2000, color='skyblue', edgecolor='black')
plt.title('Distribution of Frequency Probability of Law')
plt.xlabel('Frequency Probability')
plt.ylabel('Frequency')
plt.xlim(0, 0.02)  # x축 범위를 0 ~ 0.01로 제한
plt.grid(True)

# 히스토그램 표시
plt.show()


# In[34]:


# frequency_probability 값이 0이 아닌 데이터만 선택
filtered_data = law[law['direct_freq'] != 0]['direct_freq']

# 히스토그램 그리기
plt.figure(figsize=(8, 6))
plt.hist(filtered_data, bins=2000, color='wheat', edgecolor='black')
plt.title('Distribution of Frequency Probability (excluding 0)')
plt.xlabel('Frequency Probability')
plt.ylabel('Count')
plt.xlim(0, 0.01)  # x축 범위를 0 ~ 0.01로 제한

plt.grid(True)

# 히스토그램 표시
plt.show()


# ## 2. 직접적 기간 중앙값

# In[33]:


# None 값을 제외한 'median_date_difference' 값만 선택
filtered_data = law['direct_median'].dropna()

# 히스토그램 그리기
plt.figure(figsize=(8, 6))
plt.hist(filtered_data, bins=50, color='wheat', edgecolor='black')
plt.title('Distribution of Median Date Difference of Law')
plt.xlabel('Median Date Difference (days)')
plt.ylabel('Frequency')
plt.grid(True)

# 히스토그램 표시
plt.show()


# ## 3. 간접적 빈도 확률

# In[35]:


# 히스토그램을 그리기 위한 데이터
frequency_data = law['indirect_freq']

# 히스토그램 그리기
plt.figure(figsize=(8, 6))
plt.hist(frequency_data, bins=5000, color='wheat', edgecolor='black')
plt.title('Distribution of Frequency Probability of Indirect Law')
plt.xlabel('Frequency Probability')
plt.ylabel('Frequency')
plt.xlim(0, 1)  # x축 범위를 0 ~ 0.01로 제한
plt.grid(True)

# 히스토그램 표시
plt.show()


# ## Multi-Classification Analysis

# In[3]:


import matplotlib.font_manager as fm

# 한글 폰트 설정 (AppleGothic)
plt.rcParams['font.family'] = 'AppleGothic'

# 음수 표시를 위한 설정
plt.rcParams['axes.unicode_minus'] = False


# In[9]:


law = pd.read_csv('data/social_law.csv')
law


# In[37]:


law['result'].unique()


# In[63]:


law['result'].value_counts()


# ### 법안 처리 상태
# #### - 가결:
#     '수정가결': 수정된 내용으로 가결된 경우
#     '원안가결': 원안 그대로 가결된 경우
# #### - 폐기/부결:
#     '대안반영폐기': 대안이 반영되지 않고 폐기된 경우
#     '폐기': 법안이 폐기된 경우
#     '부결': 법안이 부결된 경우
# #### - 수정 요청:
#     '수정안반영폐기': 수정안이 반영되지 않고 폐기된 경우
#     '철회': 법안이 철회된 경우

# ### 3개 범주 분류

# In[5]:


category_map = {
    '수정가결': '가결',
    '원안가결': '가결',
    '대안반영폐기': '폐기/부결',
    '폐기': '폐기/부결',
    '부결': '폐기/부결',
    '수정안반영폐기': '수정 요청',
    '철회': '수정 요청'
}

# Replace values in the 'result' column based on the mapping
law['result'] = law['result'].map(category_map).fillna('기타')  # Fill with '기타' for any unmapped values
law


# In[41]:


law['result'].value_counts()


# ### 텍스트 변수 처리

# In[6]:


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import scipy.sparse as sp
import matplotlib.pyplot as plt
import seaborn as sns


# In[48]:


# Convert direct_median to numeric, handling NaN values
law['direct_median'] = pd.to_numeric(law['direct_median'], errors='coerce').fillna(0)

# Define target and features
X = law[['field', 'terminology', 'text', 'direct_freq', 'direct_median', 'indirect_freq', 'non_shown']]
y = law['result']

# Combine text fields for vectorization
X_text = X['field'] + ' ' + X['terminology'] + ' ' + X['text']

# TF-IDF Vectorization
vectorizer = TfidfVectorizer()
X_text_vectorized = vectorizer.fit_transform(X_text)

# Convert other features to DataFrame and ensure all are numeric
other_features = X[['direct_freq', 'direct_median', 'indirect_freq', 'non_shown']]

# Ensure the data types are correct (float)
other_features = other_features.astype(float)

# Combine text vectors with other features
X_combined = sp.hstack([X_text_vectorized, other_features])


# In[49]:


# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.2, random_state=42)

# Create and fit the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
print(classification_report(y_test, y_pred))


# In[52]:


# Confusion Matrix
cm = confusion_matrix(y_test, y_pred, labels=model.classes_)

# Plotting the Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=model.classes_,
            yticklabels=model.classes_)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()


# ### 가결/부결 2진 분류

# In[73]:


category_map = {
    '수정가결': '긍정적',
    '원안가결': '긍정적',
    '수정안반영폐기': '긍정적', 
    '대안반영폐기': '대안반영폐기',
    '부결': '부정적',
    '철회': '부정적', 
    '폐기': '부정적'
}

# Replace values in the 'result' column based on the mapping
law['result'] = law['result'].map(category_map).fillna('기타')  # Fill with '기타' for any unmapped values

law['result'].value_counts()


# In[74]:


# Convert direct_median to numeric, handling NaN values
law['direct_median'] = pd.to_numeric(law['direct_median'], errors='coerce').fillna(0)

# Define target and features
X = law[['field', 'terminology', 'text', 'direct_freq', 'direct_median', 'indirect_freq', 'non_shown']]
y = law['result']

# Combine text fields for vectorization
X_text = X['field'] + ' ' + X['terminology'] + ' ' + X['text']

# TF-IDF Vectorization
vectorizer = TfidfVectorizer()
X_text_vectorized = vectorizer.fit_transform(X_text)

# Convert other features to DataFrame and ensure all are numeric
other_features = X[['direct_freq', 'direct_median', 'indirect_freq', 'non_shown']]

# Ensure the data types are correct (float)
other_features = other_features.astype(float)

# Combine text vectors with other features
X_combined = sp.hstack([X_text_vectorized, other_features])


# In[75]:


# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.2, random_state=42)

# Create and fit the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
print(classification_report(y_test, y_pred))


# In[76]:


# Confusion Matrix
cm = confusion_matrix(y_test, y_pred, labels=model.classes_)

# Plotting the Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=model.classes_,
            yticklabels=model.classes_)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()


# ### 데이터 슬라이싱

# In[11]:


# Create a mask for '대안반영폐기' rows
mask = law['result'] == '대안반영폐기'

# Select only 2000 '대안반영폐기' rows, if available
limited_rows = law[mask].sample(n=2000, random_state=42) if mask.sum() > 2000 else law[mask]

# Concatenate the selected '대안반영폐기' rows with the rest of the DataFrame
law_df = pd.concat([limited_rows, law[~mask]])


# In[12]:


category_map = {
    '수정가결': '가결',
    '원안가결': '가결',
    '대안반영폐기': '폐기/부결',
    '폐기': '폐기/부결',
    '부결': '폐기/부결',
    '수정안반영폐기': '수정 요청',
    '철회': '수정 요청'
}

# Replace values in the 'result' column based on the mapping
law_df['result'] = law_df['result'].map(category_map).fillna('기타')  # Fill with '기타' for any unmapped values
law_df


# In[13]:


# Convert direct_median to numeric, handling NaN values
law_df['direct_median'] = pd.to_numeric(law_df['direct_median'], errors='coerce').fillna(0)

# Define target and features
X = law_df[['field', 'terminology', 'text', 'direct_freq', 'direct_median', 'indirect_freq', 'non_shown']]
y = law_df['result']

# Combine text fields for vectorization
X_text = X['field'] + ' ' + X['terminology'] + ' ' + X['text']

# TF-IDF Vectorization
vectorizer = TfidfVectorizer()
X_text_vectorized = vectorizer.fit_transform(X_text)

# Convert other features to DataFrame and ensure all are numeric
other_features = X[['direct_freq', 'direct_median', 'indirect_freq', 'non_shown']]

# Ensure the data types are correct (float)
other_features = other_features.astype(float)

# Combine text vectors with other features
X_combined = sp.hstack([X_text_vectorized, other_features])


# In[14]:


# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.2, random_state=42)

# Create and fit the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
print(classification_report(y_test, y_pred))


# In[22]:


# Confusion Matrix
cm = confusion_matrix(y_test, y_pred, labels=model.classes_)

# Plotting the Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='PuOr', 
            xticklabels=model.classes_,
            yticklabels=model.classes_)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()


# In[83]:


law_df['result'].value_counts()


# In[ ]:





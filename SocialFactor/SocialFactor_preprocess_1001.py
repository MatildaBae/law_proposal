#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

import re
import matplotlib.pyplot as plt

from tqdm import tqdm

from sklearn.feature_extraction.text import TfidfVectorizer


# In[3]:


# Define the file path
file_path = 'data/news_title_train1001.csv'

# Get the total number of rows (for tqdm progress bar)
total_rows = sum(1 for row in open(file_path, 'r')) - 1  # Minus 1 for header

# Set the chunk size
chunk_size = 10000

# Initialize an empty list to store the chunks
chunks = []

# Read the CSV file in chunks and show progress using tqdm
with tqdm(total=total_rows, desc="Loading CSV", unit="rows") as pbar:
    for chunk in pd.read_csv(file_path, chunksize=chunk_size):
        chunks.append(chunk)
        pbar.update(chunk_size)

# Concatenate all the chunks into a single DataFrame
news = pd.concat(chunks, ignore_index=True)

# Display the first few rows of the DataFrame
news

# 총 4분


# ## 전처리
# - '(서울=뉴스1) (이름) 기자 =' 반복되는 포멧 삭제
# - 언론사, 날짜별 고르게 데이터 수집되었는지 확인

# In[4]:


# 긴 컬럼 내용을 생략하지 않고 모두 표시
pd.set_option('display.max_colwidth', None)


# In[5]:


# 법률안 보고서 기간 기준 슬라이싱

news['write_date'] = pd.to_datetime(news['write_date'])

# Define the date range
start_date = '2014-01-12'
end_date = '2022-07-27'

# Filter the DataFrame to include only rows within the specified date range
news = news[(news['write_date'] >= start_date) & (news['write_date'] <= end_date)]

news


# In[6]:


# 날짜별 데이터 고르게 되어있는지 확인

news['write_date'].value_counts().sort_index().plot(kind='line')
plt.title('Write Date Distribution')
plt.show()

# 해보니 훅 떨어지는 날짜가 있음


# In[7]:


# 훅 떨어지는 날짜 확인 후 그외 부분 슬라이싱
# Count the number of news articles per date
news_count_by_date = news['write_date'].value_counts().sort_index()

# Find the first date where the count of news articles drops below 100
drop_below_100_date = news_count_by_date[news_count_by_date < 100].index[0]

# Display the first date where the count drops below 100
print(f"The first date where the news count drops below 100 is: {drop_below_100_date}")


# In[8]:


# Slice the data to include only articles before that date
news = news[news['write_date'] < drop_below_100_date]

# 그 뒤로 다시 그려봐
news['write_date'].value_counts().sort_index().plot(kind='line')
plt.title('Write Date Distribution')
plt.show()


# In[9]:


import matplotlib.font_manager as fm

# 한글 폰트 설정 (AppleGothic)
plt.rcParams['font.family'] = 'AppleGothic'

# 음수 표시를 위한 설정
plt.rcParams['axes.unicode_minus'] = False


# In[10]:


# 언론사별 데이터 고르게 되어있는지 확인

news['source_site'].value_counts().plot(kind='bar')
plt.title('Source Site Distribution')
plt.show()


# In[11]:


# 언론사별 필드는 고르게 되어있는가?

# Group by 'source_site' and 'field' and count occurrences
field_distribution = news.groupby(['source_site', 'field']).size().unstack(fill_value=0)

# Plot the distribution as a bar chart
field_distribution.plot(kind='bar', stacked=True, figsize=(8, 6))

# Add titles and labels
plt.title('Field Distribution by Source Site')
plt.xlabel('Source Site')
plt.ylabel('Number of Articles')
plt.legend(title='Field', bbox_to_anchor=(1.05, 1), loc='upper left')

# Show plot
plt.tight_layout()
plt.show()


# In[12]:


# 필드별 데이터 고르게 되어있는지 확인

news['field'].value_counts().plot(kind='bar')
plt.title('Field Distribution')
plt.show()


# In[13]:


# 필드별 데이터 고르게 되어있는지 확인

news['board'].value_counts().plot(kind='bar')
plt.title('Board Distribution')
plt.show()


# In[14]:


news['board'].value_counts()


# In[15]:


len(news['board'].value_counts())


# ### 스케일링 및 노멀라이제이션 결론 
# - 언론사별로 특정 필드만 엄청 많이 다루는 언론사가 없어서 굳이 샘플링 안해줘도 될듯
# - 필드를 똑같이 맞춰주면 어떤 필드가 많이 주목을 받는지 모르기 때문에, 해당 필드가 많이 언급된다고,
# - 노멀라이제이션 해주어서는 안될듯
# 
# ##### - 그런데 여전히 뉴스 개수의 상승세가 있으므로 단순히 빈도수..로 해야할지 혹은 빈도의 확률로 해야할지는 의문

# In[16]:


news.to_csv('data/news_title_sliced', index=False)
# 기간 슬라이싱만 된 데이터
# 5분 걸림


# ## TF-IDF 키워드 추출

# In[42]:


# Threshold를 지정하기 위해 어느정도의 키워드가 나오는지 확인하기 위해 처음 50개만 우선 적용
from sklearn.feature_extraction.text import TfidfVectorizer

# Sample the first 50 rows for testing
sample_news = news[:50]

# Initialize tqdm for pandas
tqdm.pandas()

# Initialize the TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer(max_df=0.85, max_features=10000, stop_words='english')

# Fit and transform the cleaned sentences with progress bar
tfidf_matrix = tfidf_vectorizer.fit_transform(tqdm(sample_news['sentences'], desc="TF-IDF Processing"))

# Get feature names (words)
feature_names = tfidf_vectorizer.get_feature_names_out()

# Function to extract keywords with TF-IDF scores above a certain threshold
def extract_keywords_by_threshold(row_idx, tfidf_matrix, feature_names, threshold=0.3):
    # Get the TF-IDF scores for the document
    tfidf_scores = tfidf_matrix[row_idx].T.todense()
    
    # Create a list of words with their TF-IDF scores
    word_scores = [(feature_names[i], tfidf_scores[i, 0]) for i in range(len(feature_names))]
    
    # Filter words by threshold
    filtered_word_scores = [(word, score) for word, score in word_scores if score >= threshold]
    
    # Return the filtered words
    return [word for word, score in filtered_word_scores]

# Apply the function to each row in the sample DataFrame and set threshold, with tqdm progress bar
threshold_value = 0.2
sample_news['filtered_keywords'] = [extract_keywords_by_threshold(i, tfidf_matrix, feature_names, threshold=threshold_value) for i in tqdm(range(tfidf_matrix.shape[0]), desc="Keyword Extraction")]


# In[41]:


# threshold 0.15
sample_news


# In[43]:


# threshold 0.2
sample_news


# ### Threshold 별로 샘플 행들 처리 시도

# In[46]:


import time
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
import pandas as pd

# 1만 행 샘플 데이터 생성
sample_news = news['sentences'][:10000]

# 1. 데이터 처리 시작 시간을 기록
start_time_total = time.time()

# 2. TF-IDF 벡터화를 시작하기 전 시간 기록
start_time_vectorization = time.time()

# Initialize the TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer(max_df=0.85, max_features=10000, stop_words='english')

# Fit and transform the cleaned sentences (샘플 데이터로 벡터화)
tfidf_matrix = tfidf_vectorizer.fit_transform(sample_news)

# 벡터화 종료 후 시간 측정
end_time_vectorization = time.time()
print(f"TF-IDF vectorization for 10,000 rows took {end_time_vectorization - start_time_vectorization} seconds.")

# 3. Feature names 추출 시간 측정
start_time_feature_names = time.time()

# Get feature names (words)
feature_names = tfidf_vectorizer.get_feature_names_out()

end_time_feature_names = time.time()
print(f"Feature names extraction took {end_time_feature_names - start_time_feature_names} seconds.")

# 4. Keyword extraction 시작 시간 기록
start_time_keyword_extraction = time.time()

# Function to extract keywords with TF-IDF scores above a certain threshold
def extract_keywords_by_threshold(row_idx, tfidf_matrix, feature_names, threshold=0.3):
    # Get the TF-IDF scores for the document
    tfidf_scores = tfidf_matrix[row_idx].T.todense()
    
    # Create a list of words with their TF-IDF scores
    word_scores = [(feature_names[i], tfidf_scores[i, 0]) for i in range(len(feature_names))]
    
    # Filter words by threshold
    filtered_word_scores = [(word, score) for word, score in word_scores if score >= threshold]
    
    # Return the filtered words
    return [word for word, score in filtered_word_scores]

# Apply the function to each row in the DataFrame with tqdm progress bar and set threshold (샘플 데이터로 적용)
threshold_value = 0.3
sample_news_filtered_keywords = [extract_keywords_by_threshold(i, tfidf_matrix, feature_names, threshold=threshold_value) for i in tqdm(range(tfidf_matrix.shape[0]), desc="Extracting Keywords")]

# Keyword extraction 종료 시간 측정
end_time_keyword_extraction = time.time()
print(f"Keyword extraction for 10,000 rows took {end_time_keyword_extraction - start_time_keyword_extraction} seconds.")

# 5. 전체 작업 종료 시간 측정
end_time_total = time.time()

print(f"Total processing time for 10,000 rows: {end_time_total - start_time_total} seconds.")


# In[47]:


# threshold 0.15로
# Apply the function to each row in the DataFrame with tqdm progress bar and set threshold (샘플 데이터로 적용)
threshold_value = 0.15
sample_news_filtered_keywords = [extract_keywords_by_threshold(i, tfidf_matrix, feature_names, threshold=threshold_value) for i in tqdm(range(tfidf_matrix.shape[0]), desc="Extracting Keywords")]

sample_news = news[:10000]
sample_news['keywords'] = sample_news_filtered_keywords

sample_news

sample_news.to_csv('data/tfidf_15_sample_news', index=False)


# In[44]:


# threshold 0.15

from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize tqdm for pandas
tqdm.pandas()

# Initialize the TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer(max_df=0.85, max_features=10000, stop_words='english')

# Fit and transform the cleaned sentences
tfidf_matrix = tfidf_vectorizer.fit_transform(news['sentences'])

# Get feature names (words)
feature_names = tfidf_vectorizer.get_feature_names_out()

# Function to extract keywords with TF-IDF scores above a certain threshold
def extract_keywords_by_threshold(row_idx, tfidf_matrix, feature_names, threshold=0.3):
    # Get the TF-IDF scores for the document
    tfidf_scores = tfidf_matrix[row_idx].T.todense()
    
    # Create a list of words with their TF-IDF scores
    word_scores = [(feature_names[i], tfidf_scores[i, 0]) for i in range(len(feature_names))]
    
    # Filter words by threshold
    filtered_word_scores = [(word, score) for word, score in word_scores if score >= threshold]
    
    # Return the filtered words
    return [word for word, score in filtered_word_scores]

# Apply the function to each row in the DataFrame with tqdm progress bar and set threshold
threshold_value = 0.15
news['filtered_keywords'] = [extract_keywords_by_threshold(i, tfidf_matrix, feature_names, threshold=threshold_value) for i in tqdm(range(tfidf_matrix.shape[0]), desc="Extracting Keywords")]

# 아니 24시간 걸린대


# In[ ]:


# Initialize tqdm for pandas
tqdm.pandas()

# Apply the function to each row in the DataFrame with tqdm progress bar and set threshold
threshold_value = 0.15
news['filtered_keywords'] = [extract_keywords_by_threshold(i, tfidf_matrix, feature_names, threshold=threshold_value) for i in tqdm(range(tfidf_matrix.shape[0]), desc="Extracting Keywords")]


# In[ ]:


# 저장
news.to_csv('data/tfidf_15_news', index=False)

# 리셋
news = news.drop(columns=['filtered_keywords'])


# ### 뉴스 타이틀 데이터로 tf-idf 시도

# In[15]:


news = pd.read_csv('data/news_title_sliced', header=0)
news


# In[20]:


news = pd.DataFrame(news)

news.isna().sum()


# In[21]:


# sentence 열에서 NaN 값을 가진 행을 제거
news = news.dropna(subset=['sentence'])


# In[27]:


news.to_csv('data/news_title_cleaned.csv', index=False)


# In[24]:


# threshold 0.15

from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize tqdm for pandas
tqdm.pandas()

# Initialize the TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer(max_df=0.85, max_features=10000, stop_words='english')

# Fit and transform the cleaned sentences
tfidf_matrix = tfidf_vectorizer.fit_transform(news['sentence'])

# Get feature names (words)
feature_names = tfidf_vectorizer.get_feature_names_out()

# Function to extract keywords with TF-IDF scores above a certain threshold
def extract_keywords_by_threshold(row_idx, tfidf_matrix, feature_names, threshold=0.3):
    # Get the TF-IDF scores for the document
    tfidf_scores = tfidf_matrix[row_idx].T.todense()
    
    # Create a list of words with their TF-IDF scores
    word_scores = [(feature_names[i], tfidf_scores[i, 0]) for i in range(len(feature_names))]
    
    # Filter words by threshold
    filtered_word_scores = [(word, score) for word, score in word_scores if score >= threshold]
    
    # Return the filtered words
    return [word for word, score in filtered_word_scores]

# Apply the function to each row in the DataFrame with tqdm progress bar and set threshold
threshold_value = 0.15
news['filtered_keywords'] = [extract_keywords_by_threshold(i, tfidf_matrix, feature_names, threshold=threshold_value) for i in tqdm(range(tfidf_matrix.shape[0]), desc="Extracting Keywords")]


# #### 나중에 test dataset 전처리 해야하면
# - 2014-01-12 - 2021-07-01 기간 슬라이싱
# - sentence nan 값 제거

# ## 벡터화 됐다고 쳐, 그리고 그 다음에 법률안과 엮는?
# - 간접적

# In[18]:


law = pd.read_csv('data/law_summary_train.csv', index_col=0)
law


# In[ ]:





# In[ ]:





# In[ ]:





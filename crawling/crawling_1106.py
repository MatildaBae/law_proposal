#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import requests
from bs4 import BeautifulSoup
# import pdfplumber
import os
from tqdm import tqdm

# CSV 파일 경로
csv_file = "data/assembly_bills.csv"

# CSV 파일 불러오기
df = pd.read_csv(csv_file)


# In[8]:


df1 = df.iloc[:30000]
df1


# In[7]:


df2 = df.iloc[30000:]
df2


# In[9]:


import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

# full_text를 가져온 후 세 번째 "제안이유 및 주요내용"부터 텍스트를 자르는 함수
def extract_proposal_text(detail_link):
    response = requests.get(detail_link)
    if response.status_code != 200:
        print(f"Failed to retrieve page at {detail_link}")
        return "페이지 로드 실패"

    # BeautifulSoup으로 HTML 파싱
    soup = BeautifulSoup(response.content, 'html.parser')

    # 페이지 전체 텍스트 추출
    full_text = soup.get_text(separator="\n", strip=True)

    # 세 번째 "제안이유 및 주요내용" 이후의 텍스트 추출
    try:
        # 첫 번째, 두 번째, 세 번째 "제안이유 및 주요내용" 위치 찾기
        first_idx = full_text.index("제안이유 및 주요내용")
        second_idx = full_text.index("제안이유 및 주요내용", first_idx + len("제안이유 및 주요내용"))
        third_idx = full_text.index("제안이유 및 주요내용", second_idx + len("제안이유 및 주요내용"))

        # 세 번째 "제안이유 및 주요내용" 이후부터 텍스트 추출 시작
        start_idx = third_idx + len("제안이유 및 주요내용")
        end_idx = full_text.find("감추기", start_idx)
        if end_idx == -1:  # "감추기"가 없을 경우 "+ 더보기"로 종료 지점 설정
            end_idx = full_text.find("+ 더보기", start_idx)

        # 지정된 구간의 텍스트 추출
        if end_idx != -1:
            proposal_text = full_text[start_idx:end_idx].strip()
        else:
            proposal_text = full_text[start_idx:].strip()  # 종료 지점이 없으면 끝까지 추출

        # 정제 작업
        proposal_text = proposal_text.replace("제안이유 및 주요내용", "")  # "제안이유 및 주요내용" 제거
        proposal_text = proposal_text.replace("\n", "")  # 줄바꿈 제거

        # 중복 텍스트 제거: 첫 문장을 찾아 이후의 중복 제거
        first_sentence = proposal_text.split(".")[0] + "."  # 첫 문장을 기준으로 찾기
        duplicate_start_idx = proposal_text.find(first_sentence, len(first_sentence))
        if duplicate_start_idx != -1:
            proposal_text = proposal_text[:duplicate_start_idx].strip()

    except ValueError:
        proposal_text = "제안이유 및 주요내용 섹션을 찾을 수 없습니다."

    return proposal_text

# 'detail_link' 열의 각 링크에서 "제안이유 및 주요내용" 섹션을 추출하여 새로운 열 'proposal_text'에 저장
tqdm.pandas()  # tqdm 활성화
df2['proposal_text'] = df2['detail_link'].progress_apply(lambda x: extract_proposal_text(x))

# 업데이트된 데이터프레임을 다시 CSV 파일로 저장
df2.to_csv("data/assembly_bills_with_proposal_text2.csv", index=False)

print("Proposal text extraction completed and saved to assembly_bills_with_proposal_text1.csv.")


# In[27]:


df2['proposal_text'][30000]


# In[16]:


law = pd.read_csv('data/review_final.csv')
law['text'][0]


# In[19]:


law


# In[15]:


pd.read_csv('data/social_law.csv')


# In[ ]:





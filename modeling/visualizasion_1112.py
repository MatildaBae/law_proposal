#!/usr/bin/env python
# coding: utf-8

# In[2]:


import matplotlib as mpl

mpl.rcParams['font.family'] = 'Malgun Gothic'  # 또는 'AppleGothic'
mpl.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지


# In[3]:


import pandas as pd
import matplotlib.pyplot as plt

# Data for the table
data = {
    "Feature": [
        "발의자_이념비율", "발의자_성비", "발의자_평균당선횟수", "발의자_평균의석수", 
        "시의성_직접_기간", "시의성_간접_빈도", "대표_발의자_당선횟수", "배경_의안",
        "시의성_직접_빈도", "배경_이슈", "배경_사건", "논의여부_0", "대표_발의자_성별_남", 
        "배경_회의록", "대표_발의자_여야_4.0", "배경_개정", "대표_발의자_의석수_19.0", 
        "대표_발의자_당선방식_비례대표", "대표_발의자_여야_2.0", "대표_발의자_의석수_122.0"
    ],
    "Importance": [
        672, 383, 301, 220, 159, 150, 146, 140, 132, 90, 69, 60, 52, 43, 37, 30, 27, 26, 25, 24
    ]
}

# Create DataFrame
importance_df = pd.DataFrame(data)

# Sort by importance for visualization
importance_df = importance_df.sort_values(by="Importance", ascending=True)

# Plot
plt.figure(figsize=(10, 8))
plt.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
plt.xlabel("Importance", fontsize=12)
plt.ylabel("Feature", fontsize=12)
plt.title("Top 20 Features by Importance", fontsize=14)
plt.tight_layout()
plt.show()


# In[ ]:





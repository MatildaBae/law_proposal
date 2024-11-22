#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import VotingClassifier
import joblib
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


import warnings
warnings.filterwarnings('ignore')


# In[8]:


import locale
locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')


# In[9]:


import os
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'


# In[28]:


import matplotlib.pyplot as plt
from matplotlib import rc
import platform

# Mac의 경우 AppleGothic 설정
if platform.system() == 'Darwin':  # MacOS
    rc('font', family='AppleGothic')
elif platform.system() == 'Windows':  # Windows
    rc('font', family='Malgun Gothic')  # 맑은 고딕
else:  # Linux
    rc('font', family='NanumGothic')  # 나눔고딕

# 음수 기호(-) 깨짐 방지
plt.rcParams['axes.unicode_minus'] = False


# In[3]:


# pandas 옵션 설정: 칼럼 이름 생략 방지
pd.set_option('display.max_columns', None)  # 모든 칼럼을 표시
pd.set_option('display.expand_frame_repr', False)  # 데이터프레임이 여러 줄에 걸쳐 표시되지 않도록 설정


# In[43]:


f1 = pd.read_csv('data/final_final.csv')
f1


# In[44]:


# 드랍할 열 인덱스 리스트
drop_columns = list(range(2, 4)) + list(range(5, 17)) + [20, 24] + list(range(32, 37)) + [43, 46, 48]

# 인덱스를 기준으로 열 이름 가져오기
columns_to_drop = f1.columns[drop_columns]

# 열 드랍
jw_f1 = f1.drop(columns=columns_to_drop)
# 여러 열 삭제
jw_f1 = jw_f1.drop(columns=['회의록_유사도_스케일링', '회의록_긍부정'])

# NaN 값을 -1로 채우기
jw_f1['시의성_직접_기간'] = jw_f1['시의성_직접_기간'].fillna(-1)

jw_f1['배경_개정'] = jw_f1['배경_개정'].fillna(0)
jw_f1['배경_사건'] = jw_f1['배경_사건'].fillna(0)
jw_f1['배경_의안'] = jw_f1['배경_의안'].fillna(0)
jw_f1['배경_이슈'] = jw_f1['배경_이슈'].fillna(0)
jw_f1['배경_회의록'] = jw_f1['배경_회의록'].fillna(0)


jw_f1 = jw_f1[jw_f1['정부/기타 발의'] == 0]

jw_f1 = jw_f1.drop(columns=['정부/기타 발의'])

jw_f1


# In[45]:


df = jw_f1
df


# In[10]:


# Load the saved model (for verification or future use)
loaded_lgbm_model = joblib.load('lgbm_model.pkl')
print("\nLightGBM model loaded successfully.")


# ### 균형 데이터

# In[46]:


# Step 1: Map disposal category and drop rows with NaN in 'disposal_category'
category_map = {
    '수정가결': '가결',
    '원안가결': '가결',
    '대안반영폐기': '폐기',
    '폐기': '폐기',
    '부결': '폐기',
    '수정안반영폐기': '폐기',
    '철회': '폐기',
    '임기만료폐기': '폐기' 
}

# Map disposal values using the category_map
df['disposal'] = df['disposal'].map(category_map)


# In[48]:


df.to_csv('data/model_input.csv')


# In[25]:


# Step 1: Balance the dataset by sampling 1000 rows each for '가결' and '폐기'
# Filter rows by disposal category
df_gagyeol = df[df['disposal'] == '가결']
df_pegi = df[df['disposal'] == '폐기']

# Resample to 1000 rows each
df_gagyeol_sampled = resample(df_gagyeol, replace=False, n_samples=1000, random_state=42)
df_pegi_sampled = resample(df_pegi, replace=False, n_samples=1000, random_state=42)

# Combine the sampled dataset
df_test_balanced = pd.concat([df_gagyeol_sampled, df_pegi_sampled])

# Step 2: Separate features and target
X_test_balanced = df_test_balanced[numerical_features + categorical_features]
y_test_balanced = df_test_balanced['disposal']

# Preprocess the features (refit the preprocessor)
# Define the preprocessor again
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ]
)

# Fit the preprocessor using the full dataset
X_full = df[numerical_features + categorical_features]
preprocessor.fit(X_full)

# Apply transformation to the test set
X_test_balanced_preprocessed = preprocessor.transform(X_test_balanced)

# Encode target labels to numeric
label_encoder = LabelEncoder()
y_test_balanced_encoded = label_encoder.fit_transform(y_test_balanced)


# In[26]:


# Step 3: Load the saved LightGBM model
loaded_lgbm_model = joblib.load('lgbm_model.pkl')
print("Model loaded successfully.")

# Step 4: Use the model for prediction
y_pred_balanced = loaded_lgbm_model.predict(X_test_balanced_preprocessed)

# Step 5: Evaluate the model
print("\nBalanced Test Set Classification Report:")
print(classification_report(y_test_balanced_encoded, y_pred_balanced, target_names=label_encoder.classes_))


# In[29]:


# Step 6: Confusion Matrix and Heatmap
cm = confusion_matrix(y_test_balanced_encoded, y_pred_balanced)
cm_labels = label_encoder.classes_  # ['가결', '폐기']

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=cm_labels, yticklabels=cm_labels)
plt.title('Confusion Matrix for Balanced Test Set')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()


# In[30]:


import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Step 1: 데이터 준비 및 전처리
categorical_features = [
    '논의여부', '대표_발의자_성별', '대표_발의자_당선방식', 
    '대표_발의자_정당이념', '대표_발의자_여야', '대표_발의자_의석수'
]

numerical_features = [
    '시의성_직접_빈도', '시의성_직접_기간', '시의성_간접_빈도',
    '발의자_평균당선횟수', '발의자_성비', '발의자_이념비율', '발의자_평균의석수',
    '배경_개정', '배경_사건', '배경_의안', '배경_이슈', '배경_회의록'
]

# 데이터에서 테스트 셋 선택 (df_balanced는 이미 balanced test set)
X_test_balanced = df_balanced[numerical_features + categorical_features]

# Step 2: 불러오기
loaded_lgbm_model = joblib.load('lgbm_model.pkl')
print("LightGBM model loaded successfully.")

# Step 3: 데이터 전처리
# 기존 모델과 동일한 preprocessor 재정의
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ]
)

# 전처리 수행
X_test_balanced_preprocessed = preprocessor.fit_transform(X_test_balanced)

# Step 4: 예측 확률 계산
# 확률값 예측
y_probabilities = loaded_lgbm_model.predict_proba(X_test_balanced_preprocessed)

# 출력
df_probabilities = pd.DataFrame(
    y_probabilities, 
    columns=['Probability_Class_0', 'Probability_Class_1']
)

print("\nPredicted Probabilities:")
print(df_probabilities.head())


# ## 종락 ver.

# In[32]:


jong_df = pd.read_csv('data/train_df_RnR.csv', index_col=0)
jong_df


# In[33]:


# Step 1: 데이터 준비
# disposal_binary 값에 따라 1000개씩 샘플링
df_class_0 = jong_df[jong_df['disposal_binary'] == 0].sample(n=1000, random_state=42)
df_class_1 = jong_df[jong_df['disposal_binary'] == 1].sample(n=1000, random_state=42)

# 테스트 데이터셋 생성
test_df = pd.concat([df_class_0, df_class_1]).reset_index(drop=True)

# Feature와 Target 분리
X_test = test_df.drop(columns=['disposal_binary'])
y_test = test_df['disposal_binary']

# Step 2: Feature Scaling
scaler = StandardScaler()
X_test_scaled = scaler.fit_transform(X_test)


# In[34]:


# Step 3: 모델 로드
model = joblib.load('data/XGB_RnR.joblib')
print("XGBoost model loaded successfully.")

# Step 4: 테스트 수행
y_pred = model.predict(X_test_scaled)

# Step 5: 결과 출력
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Class 0 (폐기)', 'Class 1 (가결)']))

print("\nConfusion Matrix:")
conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)


# In[38]:


# Step 1: 데이터 준비
# disposal_binary 값에 따라 1000개씩 샘플링
df_class_0 = jong_df[jong_df['disposal_binary'] == 0].sample(n=1000, random_state=42)
df_class_1 = jong_df[jong_df['disposal_binary'] == 1].sample(n=1000, random_state=42)

# 테스트 데이터셋 생성
test_df = pd.concat([df_class_0, df_class_1]).reset_index(drop=True)

# Feature와 Target 분리
X_test = test_df.drop(columns=['disposal_binary'])
y_test = test_df['disposal_binary']

# Step 2: Feature Scaling
scaler = StandardScaler()
X_test_scaled = scaler.fit_transform(X_test)


# In[42]:


# Step 3: 모델 로드
rf_model = joblib.load('data/RF_RnR.joblib')
print("Random Forest model loaded successfully.")

# Step 4: 테스트 수행
y_pred_rf = rf_model.predict(X_test_scaled)

# Step 5: 결과 출력
print("\nClassification Report (Random Forest):")
print(classification_report(y_test, y_pred_rf, target_names=['Class 0 (폐기)', 'Class 1 (가결)']))

print("\nConfusion Matrix (Random Forest):")
conf_matrix_rf = confusion_matrix(y_test, y_pred_rf)
print(conf_matrix_rf)


# In[ ]:





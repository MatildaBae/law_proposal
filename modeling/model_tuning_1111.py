#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


import warnings
warnings.filterwarnings('ignore')


# In[3]:


# pandas 옵션 설정: 칼럼 이름 생략 방지
pd.set_option('display.max_columns', None)  # 모든 칼럼을 표시
pd.set_option('display.expand_frame_repr', False)  # 데이터프레임이 여러 줄에 걸쳐 표시되지 않도록 설정


# In[4]:


f1 = pd.read_csv('data/final_final.csv')
f1


# In[146]:


f1.info()


# In[135]:


f1.columns


# ## 지원 시도: 
# #### 시의성 변수 + 발의자 변수 + 배경 변수(데이터 없을 시 전부 0으로 처리) + 논의 여부(1, 0 binary)

# In[5]:


# 드랍할 열 인덱스 리스트
drop_columns = list(range(0, 4)) + list(range(5, 17)) + [20, 24] + list(range(32, 37)) + [43, 46, 48]

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


# In[8]:


jw_f1.info()


# In[6]:


df = jw_f1


# In[28]:


jw_f1


# ### 랜덤포레스트

# In[219]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE

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
df_model = df

# Step 2: Separate features and target
# Identify all features in the dataset
all_features = df_model.columns.tolist()


categorical_features = [
    '논의여부', '대표_발의자_성별', '대표_발의자_당선방식', 
    '대표_발의자_정당이념', '대표_발의자_여야', '대표_발의자_의석수'
]
# Automatically identify numerical features, excluding 'disposal_category' which is the target variable
numerical_features = [feature for feature in all_features if feature not in categorical_features + ['disposal']]

X = df_model[numerical_features + categorical_features]
y = df_model['disposal']

# Step 3: Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, stratify=y, random_state=42)

# Step 4: Preprocessing for numerical and categorical features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ]
)

# Step 5: Apply SMOTE to balance the classes
X_train_preprocessed = preprocessor.fit_transform(X_train)

smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_preprocessed, y_train)

# Step 6: Train a Random Forest Classifier
model = RandomForestClassifier(random_state=777)
model.fit(X_train_resampled, y_train_resampled)

# Step 7: Evaluate the model
X_test_preprocessed = preprocessor.transform(X_test)
y_pred = model.predict(X_test_preprocessed)

print("Classification Report:")
print(classification_report(y_test, y_pred))

# Step 8: Feature Importance Analysis
feature_names = (
    preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features).tolist()
    + numerical_features
)
feature_importances = model.feature_importances_

importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=False)

# Display top features
print("\nTop Features:")
print(importance_df.head(20))


# In[130]:


import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# 1. 혼동 행렬 계산
cm = confusion_matrix(y_test, y_pred, labels=model.classes_)

# 2. 히트맵 그리기
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
plt.title('Confusion Matrix Heatmap')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()


# ### XGBoost/LightGBM

# In[11]:


df = jw_f1
df


# In[7]:


import locale
locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')


# In[12]:


get_ipython().system('pip install lightgbm')


# In[8]:


get_ipython().system('pip install -U scikit-learn')


# In[9]:


import os
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'


# In[10]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier
import joblib


# In[11]:


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
df_model = df
df_model


# In[12]:


df_model.to_csv('data/input_f.csv', index=False)


# In[ ]:


# Step 2: Separate features and target
# Identify all features in the dataset
all_features = df_model.columns.tolist()

categorical_features = [
    '논의여부', '대표_발의자_성별', '대표_발의자_당선방식', 
    '대표_발의자_정당이념', '대표_발의자_여야', '대표_발의자_의석수'
]

# Automatically identify numerical features, excluding 'disposal' which is the target variable
numerical_features = [feature for feature in all_features if feature not in categorical_features + ['disposal']]

X = df_model[numerical_features + categorical_features]
y = df_model['disposal']

# Encode target labels to numeric
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)  # Convert '가결', '폐기' to 0, 1

print("Encoded target classes:", label_encoder.classes_)  # ['가결', '폐기']

# Step 3: Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, stratify=y, random_state=42)

# Step 4: Preprocessing for numerical and categorical features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ]
)

# Step 5: Apply SMOTE to balance the classes
X_train_preprocessed = preprocessor.fit_transform(X_train)

smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_preprocessed, y_train)

# # Step 6: Train an XGBoost Classifier
# xgb_model = XGBClassifier(random_state=777, use_label_encoder=False, eval_metric='mlogloss')
# xgb_model.fit(X_train_resampled, y_train_resampled)

# # Step 7: Evaluate XGBoost
# y_pred_xgb = xgb_model.predict(X_test_preprocessed)

# print("\nXGBoost Classification Report:")
# print(classification_report(y_test, y_pred_xgb))

# Step 8: Train a LightGBM Classifier
lgbm_model = LGBMClassifier(random_state=777)
lgbm_model.fit(X_train_resampled, y_train_resampled)

# Step 9: Evaluate LightGBM
X_test_preprocessed = preprocessor.transform(X_test)
y_pred_lgbm = lgbm_model.predict(X_test_preprocessed)

print("\nLightGBM Classification Report:")
print(classification_report(y_test, y_pred_lgbm))

# Step 10: Feature Importance Analysis
feature_names = (
    preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features).tolist()
    + numerical_features
)


# In[ ]:


def extract_feature_importances(model, preprocessor, numerical_features, categorical_features, model_name):
    # Combine numerical and categorical feature names
    feature_names = (
        preprocessor.named_transformers_['num'].feature_names_in_.tolist()
        + preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features).tolist()
    )

    # Extract feature importances
    feature_importances = model.feature_importances_

    # Create DataFrame
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importances
    }).sort_values(by='Importance', ascending=False)

    # Print top 20 features
    print(f"\nTop 20 Features by Importance for {model_name}:")
    print(importance_df.head(20))

    return importance_df

# Call the function for LightGBM
lgbm_importance_df = extract_feature_importances(
    lgbm_model, preprocessor, numerical_features, categorical_features, "LightGBM"
)


# In[ ]:


lgbm_importance_df


# In[41]:


import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Step 11: Save the trained LightGBM model
joblib.dump(lgbm_model, 'lgbm_model.pkl')
print("\nLightGBM model saved as 'lgbm_model.pkl'")

# Step 12: Load the saved model (for verification or future use)
# loaded_lgbm_model = joblib.load('lgbm_model.pkl')
# print("\nLightGBM model loaded successfully.")

# Step 13: Confusion matrix and heatmap
cm = confusion_matrix(y_test, y_pred_lgbm)
cm_labels = label_encoder.classes_  # ['가결', '폐기']

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=cm_labels, yticklabels=cm_labels)
plt.title('Confusion Matrix for LightGBM')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()


# ### 정부 포함

# In[44]:


f1['정부/기타 발의'].unique()


# In[189]:


# 드랍할 열 인덱스 리스트
drop_columns = list(range(0, 4)) + list(range(5, 17)) + [20, 24] + list(range(32, 37)) + [43, 46, 48]

# 인덱스를 기준으로 열 이름 가져오기
columns_to_drop = f1.columns[drop_columns]

# 열 드랍
jw_f2 = f1.drop(columns=columns_to_drop)
# 여러 열 삭제
jw_f2 = jw_f2.drop(columns=['회의록_유사도_스케일링', '회의록_긍부정'])

# NaN 값을 -1로 채우기
jw_f2['시의성_직접_기간'] = jw_f2['시의성_직접_기간'].fillna(-1)

jw_f2['배경_개정'] = jw_f2['배경_개정'].fillna(0)
jw_f2['배경_사건'] = jw_f2['배경_사건'].fillna(0)
jw_f2['배경_의안'] = jw_f2['배경_의안'].fillna(0)
jw_f2['배경_이슈'] = jw_f2['배경_이슈'].fillna(0)
jw_f2['배경_회의록'] = jw_f2['배경_회의록'].fillna(0)

# '정부/기타 발의' 컬럼에서 값이 1인 경우 '정부'로 변경
jw_f2.loc[jw_f2['정부/기타 발의'] == 1, '정부/기타 발의'] = '정부'

jw_f2.loc[jw_f2['정부/기타 발의'] == 0, '정부/기타 발의'] = jw_f2.loc[jw_f2['정부/기타 발의'] == 0, '대표_발의자_여야']

jw_f2


# In[190]:


jw_f2['정부/기타 발의'].unique()


# In[191]:


jw_f2 = jw_f2[['disposal', '시의성_직접_빈도', '시의성_직접_기간', '시의성_간접_빈도', '논의여부',
              '정부/기타 발의', '배경_개정',
       '배경_사건', '배경_의안', '배경_이슈', '배경_회의록']]

jw_f2


# In[192]:


jw_f2.info()


# In[193]:


df = jw_f2


# ### 랜덤포레스트

# In[186]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE

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
df_model = df

# Step 2: Separate features and target
# Identify all features in the dataset
all_features = df_model.columns.tolist()


categorical_features = [
    '논의여부', '정부/기타 발의'
]

# Automatically identify numerical features, excluding 'disposal_category' which is the target variable
numerical_features = [feature for feature in all_features if feature not in categorical_features + ['disposal']]


# Step 2.5: Ensure data types are consistent
for col in categorical_features:
    df_model[col] = df_model[col].astype(str)  # Ensure all categorical features are strings

for col in numerical_features:
    df_model[col] = pd.to_numeric(df_model[col], errors='coerce')  # Ensure numerical features are numeric

X = df_model[numerical_features + categorical_features]
y = df_model['disposal']

# Step 3: Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, stratify=y, random_state=42)

# Step 4: Preprocessing for numerical and categorical features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ]
)

# Step 5: Apply SMOTE to balance the classes
X_train_preprocessed = preprocessor.fit_transform(X_train)

smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_preprocessed, y_train)

# Step 6: Train a Random Forest Classifier
model = RandomForestClassifier(random_state=777)
model.fit(X_train_resampled, y_train_resampled)

# Step 7: Evaluate the model
X_test_preprocessed = preprocessor.transform(X_test)
y_pred = model.predict(X_test_preprocessed)

print("Classification Report:")
print(classification_report(y_test, y_pred))

# Step 8: Feature Importance Analysis
feature_names = (
    preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features).tolist()
    + numerical_features
)
feature_importances = model.feature_importances_

importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=False)

# Display top features
print("\nTop Features:")
print(importance_df.head(20))


# ### XGB

# In[195]:


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
df_model = df

# Step 2: Separate features and target
all_features = df_model.columns.tolist()

categorical_features = [
    '논의여부', '정부/기타 발의'
]

numerical_features = [feature for feature in all_features if feature not in categorical_features + ['disposal']]

# Step 2.5: Ensure data types are consistent
for col in categorical_features:
    df_model[col] = df_model[col].astype(str)  # Ensure all categorical features are strings

for col in numerical_features:
    df_model[col] = pd.to_numeric(df_model[col], errors='coerce')  # Ensure numerical features are numeric

X = df_model[numerical_features + categorical_features]
y = df_model['disposal']

# Encode target labels to numeric
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)  # Convert '가결', '폐기' to 0, 1

# Step 3: Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, stratify=y, random_state=42)

# Step 4: Preprocessing for numerical and categorical features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ]
)

# Step 5: Apply SMOTE to balance the classes
X_train_preprocessed = preprocessor.fit_transform(X_train)

smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_preprocessed, y_train)

# Step 6.1: Train an XGBoost Classifier
xgb_model = XGBClassifier(random_state=777, use_label_encoder=False, eval_metric='mlogloss')
xgb_model.fit(X_train_resampled, y_train_resampled)

# Step 6.2: Train a LightGBM Classifier
lgbm_model = LGBMClassifier(random_state=777)
lgbm_model.fit(X_train_resampled, y_train_resampled)

# Step 7: Evaluate the models
X_test_preprocessed = preprocessor.transform(X_test)

# XGBoost Evaluation
y_pred_xgb = xgb_model.predict(X_test_preprocessed)
print("\nXGBoost Classification Report:")
print(classification_report(y_test, y_pred_xgb))

# LightGBM Evaluation
y_pred_lgbm = lgbm_model.predict(X_test_preprocessed)
print("\nLightGBM Classification Report:")
print(classification_report(y_test, y_pred_lgbm))

# Step 8: Feature Importance Analysis
feature_names = (
    preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features).tolist()
    + numerical_features
)

# XGBoost Feature Importances
xgb_importances = xgb_model.feature_importances_
xgb_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': xgb_importances
}).sort_values(by='Importance', ascending=False)

print("\nTop Features for XGBoost:")
print(xgb_importance_df.head(20))

# LightGBM Feature Importances
lgbm_importances = lgbm_model.feature_importances_
lgbm_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': lgbm_importances
}).sort_values(by='Importance', ascending=False)

print("\nTop Features for LightGBM:")
print(lgbm_importance_df.head(20))


# In[ ]:





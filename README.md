# 👩‍⚖️ Predicting Legislative Bill Outcomes Using Machine Learning and LLM

**2024 K-DS Hackerthon, Korea Data Science Consortium**  
*3rd Place - November 2024*

**Team Bill Gates**  
*Members: Minseok Kwon, Jongrak Jeong, Ji-won Bae, Ha-yeon Jeong*  

## 🚀 Overview

This project aims to analyze and predict the outcomes of legislative bills in the Korean National Assembly. By integrating **machine learning (ML)**, **natural language processing (NLP)**, and **large language models (LLM)**, the project provides both predictive analytics and interpretative insights. The system also offers feedback reports for enhancing legislative drafting and decision-making processes.

---

## 💡 Objectives

1. **Analysis of Legislative Outcomes**: 
   - Identify factors influencing the passage or rejection of bills.
   - Focus on interpretable models to explain these factors.

2. **Automated Predictions for New Bills**: 
   - Develop a model that predicts the likelihood of a bill’s passage based on input variables extracted from legislative review reports.
   - Ensure the model not only predicts outcomes but also explains variable importance.

3. **LLM-Based Feedback Reports**: 
   - Utilize LLMs to generate detailed feedback reports, offering explanations and suggestions to improve the likelihood of a bill's passage.

---

## 📋 Variables

### 1. Timeliness Variables
- **Direct Frequency**: Frequency of specific keywords in recent news articles.
- **Direct Period**: Recency of keyword mentions.
- **Indirect Frequency**: Frequency of related keywords in broader contexts.

### 2. Transcript Variables
- **Sentiment Score**: Sentiment analysis of committee meeting transcripts linked to the bill.
- **Similarity Score**: Cosine similarity between legislative review reports and meeting transcripts.

### 3. Proposer and Party Variables
- Gender, election count, and ideological bias of the primary proposer.
- Average attributes of all proposers, including party seat counts and ideological diversity.

### 4. Contextual Background Variables
- Legislative history of the bill, including amendments, related incidents, and societal issues.
- Frequency of previous discussions in parliamentary transcripts.

---

## 🔍 Methodology

### 1. Feature Engineering

- **Data Integration**: Combined multiple data sources, including legislative review reports, parliamentary meeting transcripts, and publicly available web corpora.
- **Natural Language Processing**:
  - Extracted **sentiment scores** and **keyword frequency** from text data using pre-trained language models like KoBERT.
  - Calculated **cosine similarity** between text-based features to measure relevance between review reports and meeting transcripts.
- **Custom Timeliness Metrics**: Developed metrics to assess the recency and frequency of keywords related to the bill in both direct and broader contexts.

### 2. Model Selection and Training

- **Model Comparisons**: Conducted extensive experiments with multiple algorithms, including:
  - **Random Forest** for variable importance insights.
  - **XGBoost and LightGBM** for predictive performance.
  - **Logistic Regression** for baseline comparisons.
- **Handling Data Imbalance**:
  - Applied **SMOTE (Synthetic Minority Oversampling Technique)** to balance the dataset, ensuring minority class representation.
- **Feature Selection**:
  - Used **ElasticNet** regularization to handle multicollinearity and select impactful variables.

### 3. Hyperparameter Tuning

- **Optimization Techniques**:
  - GridSearch for exhaustive tuning.
  - Random Search for computational efficiency.
- **Parameters Optimized**:
  - Learning rate, max depth, and number of estimators for tree-based models.
  - Regularization parameters for controlling overfitting.

### 4. Evaluation

- **Metrics**:
  - Accuracy, Precision, Recall, and F1-Score were computed to evaluate model performance.
  - Conducted **Partial Dependence Plots (PDPs)** to visualize the independent impact of each variable.
- **Validation Split**: Employed an 85-15 split for training and validation to ensure robust testing of the model.

### 5. System Integration

- Integrated the final model with an **LLM-powered feedback generation system**, which provides real-time insights and recommendations based on prediction results.

---

## Results

### 1. Variable Importance
- Top contributors: **Proposer Ideological Bias**, **Gender Ratio of Proposers**, and **Average Election Count**.
- Background variables like related issues and legislative history ranked moderately.

### 2. Interpretability via PDP
- **Partial Dependence Plot (PDP)** analysis showed:
  - Higher ideological bias and proposer election counts increased the likelihood of passage.
  - Lower female proposer ratios were associated with higher passage rates.

---

## System Implementation

- **User Inputs**: Legislative review report, proposer information, and date.
- **System Outputs**: A feedback report containing:
  - Bill summary.
  - Key insights from timeliness and background variables.
  - Predicted outcomes with actionable recommendations.

---

## Future Applications

1. **For Parliament**: Enhance the efficiency of legislative drafting and analysis.
2. **For Media**: Provide rapid insights for reporting and political commentary.
3. **For Citizens and Industry**: Improve accessibility and understanding of complex legislative processes.

---

## Acknowledgments

- Developed as part of the **2024 K-DS Hackathon** by Team Bill Gates.
- Data sourced from **AI Hub**, **National Assembly Meeting Transcripts**, and **Web Data Corpora**.


---
# 👩‍⚖️ 국회 법률안 가결 예측 - 정치·사회적 변수 기반 분석과 LLM 기반 해설

**2024 K-DS 해커톤, K-DS 컨소시엄**  
*한국지능정보사회진흥원장상 - 2024년 11월*

**Team Bill Gates**  
*팀원: 권민석, 정종락, 배지원, 정하연*  

## 🚀 개요

이 프로젝트는 한국 국회의 법률안 가결 여부를 분석하고 예측하기 위한 것으로, **머신러닝(ML)**, **자연어 처리(NLP)**, 및 **대규모 언어 모델(LLM)**을 활용하여 예측 분석 및 해석 가능한 인사이트를 제공합니다. 또한, 법률안 초안 작성 및 의사결정 과정을 개선하기 위해 피드백 보고서를 제공합니다.

---

## 💡 목표

1. **법률안 가결/부결 분석**:
   - 법률안 가결 여부에 영향을 미치는 요인을 식별.
   - 설명 가능한 모델을 통해 변수의 중요도를 분석.

2. **신규 법률안의 자동화된 예측**:
   - 법률안 검토 보고서를 입력받아 가결 가능성을 예측하는 모델 개발.
   - 예측력과 변수 해석력을 모두 갖춘 모델 목표.

3. **LLM 기반 피드백 보고서**:
   - LLM을 활용하여 법률안의 개선 방향성을 제시하는 피드백 보고서 자동 생성.

---

## 📋 변수

### 1. 시의성 변수
- **직접 빈도**: 특정 키워드가 최근 뉴스에서 언급된 빈도.
- **직접 기간**: 키워드가 최근에 등장한 시점의 중앙값.
- **간접 빈도**: 관련 키워드의 언급 빈도.

### 2. 회의록 변수
- **긍정/부정 점수**: 법률안 검토보고서와 관련된 회의록 텍스트의 감정 점수.
- **유사도 점수**: 검토보고서와 회의록 간의 코사인 유사도 점수.

### 3. 발의자 및 정당 변수
- 대표 발의자의 성별, 당선 횟수, 소속 정당 이념, 의석 수 등.
- 발의자 집단의 평균 속성(예: 이념 편향, 당선 횟수).

### 4. 논의 배경 변수
- 법률안의 과거 개정, 관련 사건 및 사회적 이슈 등의 빈도.

---

## 🔍 방법론

### 1. 변수 생성

- **데이터 통합**: 검토보고서, 회의록, 웹 데이터 등을 통합하여 주요 변수를 추출.
- **자연어 처리**:
  - KoBERT 모델을 활용해 텍스트 데이터에서 감정 점수 및 키워드 빈도 추출.
  - 검토보고서와 회의록 간 유사도를 측정하기 위해 코사인 유사도 계산.
- **시의성 메트릭 개발**:
  - 특정 키워드의 최신성과 빈도를 평가할 수 있는 메트릭 생성.

### 2. 모델 선택 및 학습

- **모델 비교**:
  - 랜덤포레스트, XGBoost, LightGBM 등 다양한 알고리즘 실험.
- **데이터 불균형 처리**:
  - SMOTE 기법을 활용하여 소수 클래스의 데이터를 증강.
- **변수 선택**:
  - ElasticNet 정규화를 통해 다중공선성을 해결하고 중요한 변수를 선정.

### 3. 하이퍼파라미터 튜닝

- **최적화 기법**:
  - GridSearch와 Random Search를 활용한 하이퍼파라미터 탐색.
- **조정 변수**:
  - 학습률, 모델 깊이, 트리 개수 등.

### 4. 평가

- **평가 메트릭**:
  - 정확도, 정밀도, 재현율, F1-Score 계산.
  - Partial Dependence Plot(PDP)을 활용해 변수의 독립적 영향을 시각화.
- **검증 데이터**:
  - 80:20 비율로 트레이닝 및 검증 데이터셋 분할.

---

## 결과

### 1. 변수 중요도
- 주요 변수: **발의자 이념 편향도**, **발의자 성비**, **평균 당선 횟수**.
- 법안 배경 변수는 중간 중요도로 평가.

### 2. PDP 분석
- 이념 편향도와 발의자 평균 당선 횟수가 높을수록 가결 가능성이 증가.
- 발의자 성비가 낮을수록 가결 가능성이 높아지는 경향.

---

## 시스템 구현

- **사용자 입력**: 법률안 검토보고서, 발의자 정보, 날짜.
- **시스템 출력**: 
  - 법률안 요약.
  - 주요 변수에 따른 예측 결과 및 피드백.

---

## 향후 활용 가능성

1. **국회**: 법률안 검토 및 초안 작성 지원.
2. **언론**: 법률안 관련 신속한 분석 및 정치평론 자료 제공.
3. **산업계/국민**: 복잡한 법률안의 접근성 향상.

---

## 감사의 말

- **2024 K-DS 해커톤**에서 Team Bill Gates가 개발.
- 데이터 출처: **AI Hub**, **국회 회의록**, **웹 데이터 말뭉치**.


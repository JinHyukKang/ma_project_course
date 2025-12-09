# 고객 이탈 예측 프로젝트

## 📋 프로젝트 개요

이 프로젝트는 **Kaggle Telco Customer Churn 데이터셋**을 활용하여 고객 이탈을 예측하는 머신러닝 모델을 구축하고, Streamlit을 통해 웹 애플리케이션으로 구현한 프로젝트입니다.

### 목표
- 고객의 이탈 가능성을 예측하여 사전 대응
- 고객 유지 전략 수립을 위한 인사이트 도출
- 실무에서 활용 가능한 머신러닝 애플리케이션 구축

### 데이터셋 정보
- **출처**: Kaggle - Telco Customer Churn
- **다운로드 링크**: https://www.kaggle.com/datasets/blastchar/telco-customer-churn
- **크기**: 약 7,000개 고객 데이터
- **특성**: 고객 정보, 계약 정보, 서비스 사용 정보 등 19개 특성
- **타겟**: Churn (이탈 여부)

---

## 🚀 시작하기

### 1. 데이터셋 다운로드

1. Kaggle 계정 생성 및 로그인
2. [Telco Customer Churn 데이터셋](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) 페이지 방문
3. "Download" 버튼 클릭하여 데이터셋 다운로드
4. 압축 해제 후 `WA_Fn-UseC_-Telco-Customer-Churn.csv` 파일을 `05_app` 폴더에 저장

### 2. 필요한 라이브러리 설치

```bash
pip install pandas numpy scikit-learn joblib matplotlib seaborn streamlit plotly
```

### 3. 모델 학습

1. Jupyter Notebook 또는 VS Code에서 `model_training.ipynb` 파일 열기
2. 모든 셀을 순서대로 실행
3. 모델 학습 완료 후 다음 파일들이 생성됩니다:
   - `churn_model.joblib` (학습된 모델)
   - `scaler.joblib` (스케일러)
   - `label_encoders.joblib` (범주형 변수 인코더)
   - `feature_names.json` (특성 이름)
   - `model_info.json` (모델 정보)

### 4. Streamlit 앱 실행

```bash
streamlit run app.py
```

브라우저에서 자동으로 앱이 열립니다.

---

## 📁 프로젝트 구조

```
05_app/
├── README.md                    # 프로젝트 설명서
├── model_training.ipynb         # 모델 학습 노트북
├── app.py                       # Streamlit 애플리케이션
├── WA_Fn-UseC_-Telco-Customer-Churn.csv  # 데이터셋 (다운로드 필요)
├── churn_model.joblib          # 학습된 모델 (학습 후 생성)
├── scaler.joblib               # 스케일러 (학습 후 생성)
├── label_encoders.joblib       # 인코더 (학습 후 생성)
├── feature_names.json          # 특성 이름 (학습 후 생성)
└── model_info.json             # 모델 정보 (학습 후 생성)
```

---

## 🔍 데이터셋 특성 설명

### 기본 정보
- **gender**: 성별 (Male, Female)
- **SeniorCitizen**: 고령자 여부 (0, 1)
- **Partner**: 파트너 여부 (Yes, No)
- **Dependents**: 부양가족 여부 (Yes, No)

### 계약 정보
- **tenure**: 계약 기간 (개월)
- **Contract**: 계약 유형 (Month-to-month, One year, Two year)
- **PaperlessBilling**: 무인 청구서 여부 (Yes, No)
- **PaymentMethod**: 결제 방법

### 서비스 정보
- **PhoneService**: 전화 서비스 여부 (Yes, No)
- **MultipleLines**: 다중 회선 여부
- **InternetService**: 인터넷 서비스 유형 (DSL, Fiber optic, No)
- **OnlineSecurity**: 온라인 보안 서비스
- **OnlineBackup**: 온라인 백업 서비스
- **DeviceProtection**: 기기 보호 서비스
- **TechSupport**: 기술 지원 서비스
- **StreamingTV**: 스트리밍 TV 서비스
- **StreamingMovies**: 스트리밍 영화 서비스

### 요금 정보
- **MonthlyCharges**: 월 요금 ($)
- **TotalCharges**: 총 요금 ($)

### 타겟 변수
- **Churn**: 이탈 여부 (Yes, No)

---

## 🎯 모델 정보

### 사용된 알고리즘
1. **로지스틱 회귀 (Logistic Regression)**
   - 선형 분류 모델
   - 해석이 용이함
   - 빠른 학습 속도

2. **랜덤 포레스트 (Random Forest)**
   - 앙상블 모델
   - 높은 성능
   - 특성 중요도 제공

### 모델 선택 기준
- ROC AUC 점수를 기준으로 최적 모델 선택
- 일반적으로 랜덤 포레스트가 더 높은 성능을 보임

---

## 📊 Streamlit 앱 기능

### 1. 예측 탭
- 고객 정보 입력
- 실시간 이탈 예측
- 예측 확률 시각화
- 상세 해석 및 권장 사항 제공

### 2. 모델 성능 탭
- 모델 성능 지표 시각화
- 정확도, 정밀도, 재현율, F1 점수, ROC AUC 표시
- 성능 지표 설명

### 3. ROC 곡선 탭
- ROC 곡선 시각화
- 모델의 분류 성능 평가

### 4. 예측 히스토리 탭
- 예측 기록 조회
- 예측 결과 분포 시각화
- 이탈 확률 분포 히스토그램

---

## 💡 사용 팁

### 모델 학습 시
- 데이터 전처리 과정을 자세히 확인하세요
- 모델 성능 비교를 통해 최적 모델을 선택하세요
- 특성 중요도를 확인하여 비즈니스 인사이트를 도출하세요

### 앱 사용 시
- 다양한 고객 정보를 입력하여 예측을 테스트해보세요
- 이탈 확률이 높은 경우의 공통 특성을 분석해보세요
- 예측 히스토리를 통해 패턴을 파악해보세요

---

## 🔧 문제 해결

### 모델 파일을 찾을 수 없다는 오류
- `model_training.ipynb`를 실행하여 모델을 먼저 학습하세요
- 생성된 파일들이 `05_app` 폴더에 있는지 확인하세요

### 데이터셋 파일을 찾을 수 없다는 오류
- Kaggle에서 데이터셋을 다운로드했는지 확인하세요
- 파일명이 정확한지 확인하세요: `WA_Fn-UseC_-Telco-Customer-Churn.csv`
- 파일이 `05_app` 폴더에 있는지 확인하세요

### 인코딩 오류
- 새로운 범주형 값이 입력된 경우, 모델 학습 시 사용된 값만 입력 가능합니다
- 범주형 변수의 가능한 값들을 확인하세요

---

## 📚 참고 자료

- [Kaggle Telco Customer Churn 데이터셋](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- [Streamlit 공식 문서](https://docs.streamlit.io/)
- [Scikit-learn 공식 문서](https://scikit-learn.org/)

---

## 📝 라이선스

이 프로젝트는 교육 목적으로 제작되었습니다.

---

## 👥 기여자

- 프로젝트 개발: [당신의 이름]
- 데이터셋 제공: Kaggle

---

**Happy Coding! 🚀**


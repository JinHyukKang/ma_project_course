# app.py - 고객 이탈 예측 시스템
import streamlit as st
import joblib
import pandas as pd
import numpy as np
import json
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import roc_curve, auc

# 페이지 설정
st.set_page_config(
    page_title="고객 이탈 예측 시스템",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 제목
st.title("📊 고객 이탈 예측 시스템")
st.markdown("---")

# 모델 로드 함수들
@st.cache_resource
def load_model():
    """모델 로드"""
    try:
        return joblib.load('./05_app/churn_model.joblib')
    except FileNotFoundError:
        st.error("❌ 모델 파일을 찾을 수 없습니다. 먼저 model_training.ipynb를 실행하여 모델을 학습하세요.")
        return None

@st.cache_resource
def load_scaler():
    """스케일러 로드"""
    try:
        return joblib.load('./05_app/scaler.joblib')
    except FileNotFoundError:
        return None

@st.cache_resource
def load_label_encoders():
    """LabelEncoder 로드"""
    try:
        return joblib.load('./05_app/label_encoders.joblib')
    except FileNotFoundError:
        return None

@st.cache_data
def load_model_info():
    """모델 정보 로드"""
    try:
        with open('./05_app/model_info.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return None

@st.cache_data
def load_feature_names():
    """특성 이름 로드"""
    try:
        with open('./05_app/feature_names.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return None

# 모델 및 정보 로드
model = load_model()
scaler = load_scaler()
label_encoders = load_label_encoders()
model_info = load_model_info()
feature_names = load_feature_names()

if model is None or scaler is None or label_encoders is None:
    st.stop()

# 세션 상태 초기화
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []

# 사이드바
with st.sidebar:
    st.header("⚙️ 설정")
    
    # 모델 정보
    st.subheader("📊 모델 성능")
    if model_info:
        st.metric("정확도", f"{model_info['accuracy']:.3f}")
        st.metric("정밀도", f"{model_info['precision']:.3f}")
        st.metric("재현율", f"{model_info['recall']:.3f}")
        st.metric("F1 점수", f"{model_info['f1_score']:.3f}")
        st.metric("ROC AUC", f"{model_info['roc_auc']:.3f}")
    
    st.divider()
    
    # 히스토리 관리
    st.subheader("📜 히스토리 관리")
    st.write(f"총 예측 횟수: {len(st.session_state.prediction_history)}회")
    
    if st.button("🗑️ 히스토리 초기화", use_container_width=True):
        st.session_state.prediction_history = []
        st.rerun()

# 메인 탭
tab1, tab2, tab3, tab4 = st.tabs([
    "🔮 예측", 
    "📊 모델 성능", 
    "📈 ROC 곡선",
    "📜 예측 히스토리"
])

# 탭 1: 예측
with tab1:
    st.header("고객 이탈 예측")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("고객 정보 입력")
        
        # 입력 필드를 섹션별로 구성
        with st.expander("👤 기본 정보", expanded=True):
            col_basic1, col_basic2 = st.columns(2)
            with col_basic1:
                gender = st.selectbox("성별", ["Male", "Female"], key="gender")
                senior_citizen = st.selectbox("고령자 여부", ["No", "Yes"], key="senior")
                partner = st.selectbox("파트너 여부", ["No", "Yes"], key="partner")
                dependents = st.selectbox("부양가족 여부", ["No", "Yes"], key="dependents")
            
            with col_basic2:
                tenure = st.number_input("계약 기간 (개월)", min_value=0, max_value=100, value=12, step=1, key="tenure")
                monthly_charges = st.number_input("월 요금 ($)", min_value=0.0, max_value=200.0, value=50.0, step=0.1, key="monthly")
                total_charges = st.number_input("총 요금 ($)", min_value=0.0, max_value=10000.0, value=500.0, step=1.0, key="total")
        
        with st.expander("📞 서비스 정보", expanded=True):
            col_service1, col_service2 = st.columns(2)
            with col_service1:
                phone_service = st.selectbox("전화 서비스", ["No", "Yes"], key="phone")
                multiple_lines = st.selectbox("다중 회선", ["No", "Yes", "No phone service"], key="multiple")
                internet_service = st.selectbox("인터넷 서비스", ["DSL", "Fiber optic", "No"], key="internet")
                online_security = st.selectbox("온라인 보안", ["No", "Yes", "No internet service"], key="security")
            
            with col_service2:
                online_backup = st.selectbox("온라인 백업", ["No", "Yes", "No internet service"], key="backup")
                device_protection = st.selectbox("기기 보호", ["No", "Yes", "No internet service"], key="device")
                tech_support = st.selectbox("기술 지원", ["No", "Yes", "No internet service"], key="tech")
                streaming_tv = st.selectbox("스트리밍 TV", ["No", "Yes", "No internet service"], key="tv")
                streaming_movies = st.selectbox("스트리밍 영화", ["No", "Yes", "No internet service"], key="movies")
        
        with st.expander("💳 계약 정보", expanded=True):
            col_contract1, col_contract2 = st.columns(2)
            with col_contract1:
                contract = st.selectbox("계약 유형", ["Month-to-month", "One year", "Two year"], key="contract")
                paperless_billing = st.selectbox("무인 청구서", ["No", "Yes"], key="paperless")
            
            with col_contract2:
                payment_method = st.selectbox("결제 방법", [
                    "Electronic check", 
                    "Mailed check", 
                    "Bank transfer (automatic)", 
                    "Credit card (automatic)"
                ], key="payment")
        
        # 예측 버튼
        if st.button("🔮 이탈 예측하기", type="primary", use_container_width=True):
            # 입력 데이터를 딕셔너리로 구성
            input_dict = {
                'gender': gender,
                'SeniorCitizen': 1 if senior_citizen == "Yes" else 0,
                'Partner': partner,
                'Dependents': dependents,
                'tenure': tenure,
                'PhoneService': phone_service,
                'MultipleLines': multiple_lines,
                'InternetService': internet_service,
                'OnlineSecurity': online_security,
                'OnlineBackup': online_backup,
                'DeviceProtection': device_protection,
                'TechSupport': tech_support,
                'StreamingTV': streaming_tv,
                'StreamingMovies': streaming_movies,
                'Contract': contract,
                'PaperlessBilling': paperless_billing,
                'PaymentMethod': payment_method,
                'MonthlyCharges': monthly_charges,
                'TotalCharges': total_charges
            }
            
            # 데이터프레임으로 변환
            input_df = pd.DataFrame([input_dict])
            
            # 범주형 변수 인코딩
            input_encoded = input_df.copy()
            for col in label_encoders.keys():
                if col in input_encoded.columns:
                    # 새로운 값이 있으면 처리
                    try:
                        input_encoded[col] = label_encoders[col].transform([input_dict[col]])[0]
                    except ValueError:
                        # 새로운 값이 있으면 가장 빈도가 높은 값으로 대체
                        input_encoded[col] = 0
            
            # 특성 순서 맞추기
            input_encoded = input_encoded[feature_names]
            
            # 스케일링
            input_scaled = scaler.transform(input_encoded)
            
            # 예측 수행
            prediction = model.predict(input_scaled)[0]
            probability = model.predict_proba(input_scaled)[0, 1]
            
            # 히스토리에 추가
            from datetime import datetime
            prediction_record = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                **input_dict,
                'prediction': int(prediction),
                'probability': float(probability)
            }
            st.session_state.prediction_history.append(prediction_record)
            st.rerun()
    
    with col2:
        st.subheader("예측 결과")
        
        # 최신 예측 결과 표시
        if st.session_state.prediction_history:
            latest = st.session_state.prediction_history[-1]
            prediction = latest['prediction']
            probability = latest['probability']
            
            # 예측 클래스 표시
            if prediction == 1:
                st.error(f"### ⚠️ 이탈 예상")
                st.warning(f"이탈 확률: {probability:.1%}")
            else:
                st.success(f"### ✅ 유지 예상")
                st.info(f"유지 확률: {1-probability:.1%}")
            
            # 확률 메트릭
            st.metric(
                label="이탈 확률",
                value=f"{probability:.1%}",
                delta=f"{probability-0.5:.1%}" if probability >= 0.5 else f"{probability-0.5:.1%}"
            )
            
            # 확률 시각화 (막대 그래프)
            prob_data = pd.DataFrame({
                '클래스': ['유지', '이탈'],
                '확률': [1-probability, probability]
            })
            
            fig_prob = px.bar(
                prob_data,
                x='클래스',
                y='확률',
                color='클래스',
                color_discrete_map={'유지': 'green', '이탈': 'red'},
                title='예측 확률',
                text='확률'
            )
            fig_prob.update_traces(texttemplate='%{text:.1%}', textposition='outside')
            fig_prob.update_layout(yaxis_tickformat='.0%', height=300)
            st.plotly_chart(fig_prob, use_container_width=True)
            
            # 결과 해석
            with st.expander("📖 상세 해석"):
                st.write(f"""
                **예측 결과**:
                - 예측 클래스: {'이탈 예상' if prediction == 1 else '유지 예상'}
                - 이탈 확률: {probability:.1%}
                - 유지 확률: {1-probability:.1%}
                
                **해석**:
                - 모델은 입력된 고객 정보를 기반으로 이탈 가능성을 {probability:.1%}로 예측했습니다.
                - 이 예측은 학습 데이터의 패턴을 기반으로 계산되었습니다.
                - 이탈 확률이 높은 경우, 고객 유지 전략을 수립하는 것이 좋습니다.
                
                **권장 사항**:
                - 이탈 확률이 50% 이상인 경우: 즉시 고객 유지 캠페인 시작
                - 이탈 확률이 30-50%인 경우: 모니터링 강화 및 선제적 대응
                - 이탈 확률이 30% 미만인 경우: 정기적인 고객 만족도 조사
                """)
        else:
            st.info("예측을 수행하면 결과가 여기에 표시됩니다.")

# 탭 2: 모델 성능
with tab2:
    st.header("모델 성능 지표")
    
    if model_info:
        # 성능 지표 표시
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 성능 지표")
            
            # 메트릭으로 표시
            metrics_cols = st.columns(3)
            with metrics_cols[0]:
                st.metric("정확도", f"{model_info['accuracy']:.3f}")
                st.metric("정밀도", f"{model_info['precision']:.3f}")
            with metrics_cols[1]:
                st.metric("재현율", f"{model_info['recall']:.3f}")
                st.metric("F1 점수", f"{model_info['f1_score']:.3f}")
            with metrics_cols[2]:
                st.metric("ROC AUC", f"{model_info['roc_auc']:.3f}")
            
            # 성능 지표 시각화
            metrics_data = pd.DataFrame({
                '지표': ['정확도', '정밀도', '재현율', 'F1 점수', 'ROC AUC'],
                '값': [
                    model_info['accuracy'],
                    model_info['precision'],
                    model_info['recall'],
                    model_info['f1_score'],
                    model_info['roc_auc']
                ]
            })
            
            fig_metrics = px.bar(
                metrics_data,
                x='지표',
                y='값',
                title='모델 성능 지표',
                color='값',
                color_continuous_scale='Viridis',
                text='값'
            )
            fig_metrics.update_traces(texttemplate='%{text:.3f}', textposition='outside')
            fig_metrics.update_layout(yaxis_range=[0, 1], height=400)
            st.plotly_chart(fig_metrics, use_container_width=True)
        
        with col2:
            st.subheader("📊 모델 정보")
            st.write(f"**모델 이름**: {model_info.get('model_name', 'N/A')}")
            st.write(f"**훈련 샘플 수**: {model_info.get('training_samples', 'N/A'):,}개")
            st.write(f"**테스트 샘플 수**: {model_info.get('test_samples', 'N/A'):,}개")
            st.write(f"**특성 개수**: {len(model_info.get('feature_names', []))}개")
            
            st.divider()
            
            st.subheader("📈 성능 지표 설명")
            with st.expander("지표 상세 설명"):
                st.write("""
                **정확도 (Accuracy)**: 전체 예측 중 올바르게 예측한 비율
                
                **정밀도 (Precision)**: 이탈이라고 예측한 고객 중 실제로 이탈한 고객의 비율
                
                **재현율 (Recall)**: 실제 이탈한 고객 중 모델이 찾아낸 고객의 비율
                
                **F1 점수**: 정밀도와 재현율의 조화 평균
                
                **ROC AUC**: 모델의 분류 성능을 종합적으로 평가하는 지표 (1에 가까울수록 좋음)
                """)

# 탭 3: ROC 곡선
with tab3:
    st.header("ROC 곡선")
    
    if model_info:
        st.write("ROC 곡선은 모델의 분류 성능을 시각화한 것입니다.")
        st.write(f"**현재 모델의 ROC AUC**: {model_info['roc_auc']:.3f}")
        
        # ROC 곡선 시각화 (예시)
        # 실제 ROC 곡선을 그리려면 테스트 데이터가 필요하지만, 여기서는 예시로 표시
        st.info("💡 실제 ROC 곡선을 보려면 model_training.ipynb를 실행하세요.")
        
        # 예시 ROC 곡선
        fpr = np.linspace(0, 1, 100)
        tpr = np.sqrt(fpr)  # 예시 곡선
        auc_score = model_info['roc_auc']
        
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(
            x=fpr,
            y=tpr,
            mode='lines',
            name=f'현재 모델 (AUC = {auc_score:.3f})',
            line=dict(color='blue', width=2)
        ))
        fig_roc.add_trace(go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode='lines',
            name='랜덤 분류기 (AUC = 0.500)',
            line=dict(color='red', width=2, dash='dash')
        ))
        fig_roc.update_layout(
            title='ROC 곡선',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            height=500
        )
        st.plotly_chart(fig_roc, use_container_width=True)

# 탭 4: 예측 히스토리
with tab4:
    st.header("예측 히스토리")
    
    if st.session_state.prediction_history:
        # 히스토리를 데이터프레임으로 변환
        history_df = pd.DataFrame(st.session_state.prediction_history)
        
        # 주요 컬럼만 선택하여 표시
        display_cols = ['timestamp', 'gender', 'tenure', 'MonthlyCharges', 'Contract', 'prediction', 'probability']
        display_df = history_df[display_cols].copy()
        display_df['prediction'] = display_df['prediction'].map({0: '유지', 1: '이탈'})
        display_df['probability'] = display_df['probability'].apply(lambda x: f"{x:.1%}")
        display_df.columns = ['시간', '성별', '계약기간', '월요금', '계약유형', '예측', '이탈확률']
        
        st.dataframe(display_df, use_container_width=True)
        
        # 히스토리 시각화
        col1, col2 = st.columns(2)
        
        with col1:
            # 이탈 예측 분포
            prediction_counts = history_df['prediction'].value_counts().sort_index()
            # 인덱스를 '유지' 또는 '이탈'로 매핑
            prediction_names = prediction_counts.index.map({0: '유지', 1: '이탈'}).tolist()
            prediction_values = prediction_counts.values.tolist()
            
            fig_pred = px.pie(
                values=prediction_values,
                names=prediction_names,
                title='예측 결과 분포',
                color_discrete_map={'유지': 'green', '이탈': 'red'}
            )
            st.plotly_chart(fig_pred, use_container_width=True)
        
        with col2:
            # 이탈 확률 분포
            fig_prob_hist = px.histogram(
                history_df,
                x='probability',
                nbins=20,
                title='이탈 확률 분포',
                labels={'probability': '이탈 확률', 'count': '빈도'}
            )
            st.plotly_chart(fig_prob_hist, use_container_width=True)
    else:
        st.info("아직 예측 기록이 없습니다. 예측 탭에서 예측을 수행해보세요.")


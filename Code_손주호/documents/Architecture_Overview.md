# Tire Defect Prediction - Hybrid Model Architecture

## 📊 프로젝트 개요

**목적**: 타이어 설계 및 시뮬레이션 데이터를 활용한 불량(Defect) 예측
**문제 유형**: Binary Classification (Good vs NG)
**접근 방법**: Hybrid Stacking Ensemble (Boosting + Deep Learning)

---

## 🏗️ 전체 시스템 아키텍처

```
┌─────────────────────────────────────────────────────────────────┐
│                        Input Data (train.csv)                    │
│  - Design Features: Mass_Pilot, Width, Aspect, Proc_Param1~11  │
│  - Simulation Features: x0~x255, y0~y255, p0~p255, G1~G4       │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│              Step 1: Feature Engineering Pipeline                │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │  Missing Val │→ │  Categorical │→ │   Feature    │         │
│  │   Handling   │  │   Encoding   │  │   Scaling    │         │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│          Step 2: Feature Selection (XGBoost Importance)          │
│  All Features → Top N Important Features (Default: Top 100)     │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
         ┌───────────────┴───────────────┐
         │                               │
         ▼                               ▼
┌────────────────────┐          ┌────────────────────┐
│   Branch 1         │          │   Branch 2         │
│   XGBoost Model    │          │   DNN Model        │
│  (Design Features) │          │ (Simulation Feats) │
│                    │          │                    │
│  Output:           │          │  Output:           │
│  • p1: Probability │          │  • h2: Latent (64) │
│  • h1: Leaf Index  │          │  • p2: Prediction  │
│     (300 trees)    │          │                    │
└─────────┬──────────┘          └──────────┬─────────┘
          │                                │
          └────────────┬───────────────────┘
                       │
                       ▼
          ┌────────────────────────┐
          │  Fusion Model (MLP)    │
          │                        │
          │  Input: [p1, h1, h2,   │
          │         p2]            │
          │                        │
          │  Layers:               │
          │  • FC(total_dim → 64)  │
          │  • FC(64 → 32)         │
          │  • FC(32 → 1)          │
          │                        │
          │  Output: Final Defect  │
          │          Probability   │
          └────────────────────────┘
```

---

## 🔧 각 컴포넌트 상세 설명

### 1️⃣ **Feature Engineering Pipeline**

#### 1.1 FeaturePreprocessor Class

**역할**: 모듈화된 전처리 파이프라인으로 데이터 일관성 보장

**주요 메서드**:
- `handle_missing_values()`: 결측치 처리 (median/mean imputation)
- `encode_categorical()`: 범주형 변수 인코딩 (LabelEncoder)
- `scale_features()`: 그룹별 표준화 (StandardScaler)
- `fit_transform()`: 학습 데이터에 fit + transform
- `transform()`: 검증/테스트 데이터에 transform만 적용

**특징**:
- Verbose 모드로 각 단계별 진행상황 출력
- Feature 그룹별(Design/Simulation) 독립적인 스케일링
- 재사용 가능한 객체 지향 구조

```python
# 사용 예시
preprocessor = FeaturePreprocessor(verbose=True)
df_processed = preprocessor.fit_transform(df_train,
                                          design_features,
                                          simulation_features,
                                          target_col)
```

---

### 2️⃣ **Feature Selection**

#### 2.1 XGBoost 기반 Feature Importance

**목적**: 고차원 데이터에서 중요한 특징만 선택하여 모델 성능 향상 및 과적합 방지

**프로세스**:
1. 전체 특징으로 임시 XGBoost 모델 학습
2. `feature_importances_` 추출
3. 상위 N개 또는 임계값 이상 특징 선택
4. Design 특징과 Simulation 특징으로 분리

**하이퍼파라미터**:
```python
TOP_N_FEATURES = 100  # 상위 100개 특징 선택
IMPORTANCE_THRESHOLD = 0.001  # 대안: 임계값 기반 선택
```

**결과**:
- 선택된 Design 특징 → Branch 1 (XGBoost)
- 선택된 Simulation 특징 → Branch 2 (DNN)

---

### 3️⃣ **Branch 1: XGBoost Model (Design Features)**

#### 3.1 모델 구조

**입력**: 선택된 Design/Process Parameters (e.g., Width, Aspect, Proc_Param1~11)

**XGBoost 하이퍼파라미터**:
```python
xgb.XGBClassifier(
    n_estimators=300,        # 트리 개수
    max_depth=7,             # 트리 깊이
    learning_rate=0.05,      # 학습률
    subsample=0.8,           # 행 샘플링 비율
    colsample_bytree=0.8,    # 열 샘플링 비율
    tree_method='hist',      # 히스토그램 기반 (빠름)
    eval_metric='logloss'    # 평가 지표
)
```

#### 3.2 출력

1. **p1 (Prediction Probability)**
   - 크기: `(n_samples, 1)`
   - XGBoost의 불량 예측 확률
   - `predict_proba()[:, 1]`로 추출

2. **h1 (Latent Features - Leaf Indices)**
   - 크기: `(n_samples, n_estimators)` = `(n_samples, 300)`
   - 각 트리에서 샘플이 도달한 리프 노드 인덱스
   - `apply()` 메서드로 추출
   - **의미**: 의사결정 경로를 인코딩한 고수준 특징

#### 3.3 특징

- **장점**: 범주형/수치형 혼합 데이터 처리 우수
- **해석력**: Feature importance 제공
- **강건성**: Outlier에 강함

---

### 4️⃣ **Branch 2: Deep Neural Network (Simulation Features)**

#### 4.1 모델 구조 (SimulationDNN)

**입력**: 선택된 Simulation Features (e.g., x0~x255, y0~y255, p0~p255, G1~G4)

**아키텍처**:

```python
SimulationDNN(
    input_dim=len(selected_simulation_features),
    latent_dim=64,
    dropout_rate=0.3
)
```

**레이어 구성**:

```
┌─────────────────────────────────────┐
│ Input Layer (input_dim)             │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│ Linear(input_dim → 256)             │
│ BatchNorm1d(256)                    │
│ ReLU()                              │
│ Dropout(0.3)                        │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│ Linear(256 → 128)                   │
│ BatchNorm1d(128)                    │
│ ReLU()                              │
│ Dropout(0.3)                        │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│ Linear(128 → 64)    [Latent h2]    │
│ BatchNorm1d(64)                     │
│ ReLU()                              │
└──────────────┬──────────────────────┘
               │
               ├──────────────────────────┐
               │                          │
               ▼                          │
┌─────────────────────────┐              │
│ Prediction Head         │              │
│ Linear(64 → 32)         │              │
│ ReLU()                  │              │
│ Dropout(0.3)            │              │
│ Linear(32 → 1)  [p2]    │              │
└─────────────────────────┘              │
                                         │
                        h2 (Latent) ─────┘
```

#### 4.2 출력

1. **h2 (Latent Features)**
   - 크기: `(n_samples, 64)`
   - 인코더의 압축된 표현
   - **의미**: 시뮬레이션 데이터의 고차원 패턴을 저차원으로 압축

2. **p2 (Prediction Logits)**
   - 크기: `(n_samples, 1)`
   - DNN의 불량 예측 (로짓)
   - Sigmoid를 거쳐 확률로 변환 가능

#### 4.3 학습 설정

```python
# Loss Function
criterion = nn.BCEWithLogitsLoss()

# Optimizer
optimizer = optim.Adam(model.parameters(),
                       lr=0.001,
                       weight_decay=1e-5)

# Learning Rate Scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', patience=5, factor=0.5
)

# Early Stopping
patience = 15
n_epochs = 100
```

#### 4.4 특징

- **장점**: 고차원 벡터 데이터(프로파일, 곡선) 처리 우수
- **정규화**: Batch Normalization + Dropout으로 과적합 방지
- **적응적 학습**: ReduceLROnPlateau로 학습률 자동 조정

---

### 5️⃣ **Fusion Model (Final Decision Layer)**

#### 5.1 모델 구조 (HybridFusionModel)

**입력**: Branch 1과 Branch 2의 모든 출력 결합

- `p1`: XGBoost 예측 확률 `(n_samples, 1)`
- `h1`: XGBoost 리프 인덱스 `(n_samples, 300)` → StandardScaler로 정규화
- `h2`: DNN 잠재 특징 `(n_samples, 64)`
- `p2`: DNN 예측 로짓 `(n_samples, 1)`

**Total Input Dimension**: `1 + 300 + 64 + 1 = 366`

**아키텍처**:

```python
HybridFusionModel(
    boosting_pred_dim=1,
    boosting_latent_dim=300,
    dnn_latent_dim=64,
    dropout_rate=0.3
)
```

**레이어 구성**:

```
┌─────────────────────────────────────────┐
│ Concatenate [p1, h1, h2, p2]            │
│ Input Dimension: 366                    │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│ Linear(366 → 64)                        │
│ BatchNorm1d(64)                         │
│ ReLU()                                  │
│ Dropout(0.3)                            │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│ Linear(64 → 32)                         │
│ BatchNorm1d(32)                         │
│ ReLU()                                  │
│ Dropout(0.3)                            │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│ Linear(32 → 1)                          │
│ Output: Final Defect Probability        │
│ (Apply Sigmoid for probability)         │
└─────────────────────────────────────────┘
```

#### 5.2 학습 전략

**2단계 학습 (Two-Stage Training)**:

1. **Stage 1**: Branch 1과 Branch 2 독립적으로 학습
   - XGBoost: Design 특징에 대해 학습
   - DNN: Simulation 특징에 대해 학습

2. **Stage 2**: Fusion Model 학습
   - Branch 1, 2의 출력을 입력으로 사용
   - Branch 모델은 고정 (frozen) 또는 미세조정 가능
   - 현재 구현: Branch 모델 고정, Fusion만 학습

**하이퍼파라미터**:
```python
n_epochs = 80
batch_size = 64
learning_rate = 0.001
patience = 15  # Early stopping
```

#### 5.3 장점

✅ **앙상블 효과**: 두 개의 이질적인 모델 결합으로 일반화 성능 향상
✅ **특징 상호작용**: 서로 다른 데이터 소스의 상호작용 학습
✅ **유연성**: 각 브랜치를 독립적으로 개선 가능
✅ **강건성**: 한 브랜치의 약점을 다른 브랜치가 보완

---

## 📈 데이터 흐름 (Data Flow)

### 학습 단계 (Training Phase)

```
Raw Data (train.csv)
    │
    ├─→ Feature Engineering
    │       │
    │       ├─→ Missing Value Handling
    │       ├─→ Categorical Encoding
    │       └─→ Feature Scaling
    │
    ├─→ Feature Selection (XGBoost Importance)
    │       │
    │       ├─→ Selected Design Features
    │       └─→ Selected Simulation Features
    │
    ├─→ Train/Validation Split (80/20)
    │
    ├─→ Branch 1: XGBoost Training
    │       │
    │       ├─→ Input: Design Features
    │       └─→ Output: p1, h1
    │
    ├─→ Branch 2: DNN Training
    │       │
    │       ├─→ Input: Simulation Features
    │       └─→ Output: h2, p2
    │
    └─→ Fusion Model Training
            │
            ├─→ Input: [p1, h1, h2, p2]
            └─→ Output: Final Prediction
```

### 추론 단계 (Inference Phase)

```
New Data (test.csv)
    │
    ├─→ Apply Fitted Preprocessor
    │       │
    │       └─→ Transform (no fitting)
    │
    ├─→ Extract Selected Features
    │       │
    │       ├─→ Design Features
    │       └─→ Simulation Features
    │
    ├─→ XGBoost Inference
    │       │
    │       └─→ p1, h1
    │
    ├─→ DNN Inference
    │       │
    │       └─→ h2, p2
    │
    └─→ Fusion Model Inference
            │
            └─→ Final Defect Probability
```

---

## 🎯 모델 성능 평가

### 평가 지표

1. **Accuracy**: 전체 정확도
2. **Precision**: 불량으로 예측한 것 중 실제 불량 비율
3. **Recall**: 실제 불량 중 올바르게 예측한 비율
4. **F1-Score**: Precision과 Recall의 조화평균
5. **ROC-AUC**: 분류 임계값에 무관한 성능 지표

### 모델 비교

노트북에서는 다음 3가지 모델을 비교:

1. **XGBoost Only** (Design Features)
2. **DNN Only** (Simulation Features)
3. **Hybrid Fusion** (Combined)

→ Hybrid 모델이 일반적으로 가장 우수한 성능 기대

---

## 💾 저장된 모델 아티팩트

학습 완료 후 다음 파일들이 저장됨:

| 파일명 | 설명 | 사용 용도 |
|--------|------|-----------|
| `preprocessor.pkl` | FeaturePreprocessor 객체 | 전처리 파이프라인 재사용 |
| `xgb_model.json` | XGBoost 모델 | Branch 1 추론 |
| `scaler_xgb_latent.pkl` | XGBoost 잠재특징 스케일러 | h1 정규화 |
| `best_dnn_model.pth` | DNN 모델 가중치 | Branch 2 추론 |
| `best_fusion_model.pth` | Fusion 모델 가중치 | 최종 추론 |
| `feature_config.pkl` | 선택된 특징 리스트 | 특징 추출 |

---

## 🔄 추론 파이프라인 사용법

```python
# 1. 모델 및 아티팩트 로드
import pickle
import torch

with open('preprocessor.pkl', 'rb') as f:
    preprocessor = pickle.load(f)

xgb_model = xgb.XGBClassifier()
xgb_model.load_model('xgb_model.json')

dnn_model = SimulationDNN(input_dim, latent_dim=64)
dnn_model.load_state_dict(torch.load('best_dnn_model.pth'))

fusion_model = HybridFusionModel(...)
fusion_model.load_state_dict(torch.load('best_fusion_model.pth'))

with open('feature_config.pkl', 'rb') as f:
    feature_config = pickle.load(f)

# 2. 새로운 데이터 예측
df_test = pd.read_csv('test.csv')

predictions = predict_tire_defects(
    df_test,
    preprocessor,
    xgb_model,
    dnn_model,
    fusion_model,
    feature_config['selected_design_features'],
    feature_config['selected_simulation_features'],
    scaler_xgb_latent,
    device
)

# 3. 결과 저장
submission = pd.DataFrame({
    'id': df_test['id'],
    'prediction': predictions.flatten()
})
submission.to_csv('submission.csv', index=False)
```

---

## 🚀 성능 최적화 팁

### 1. 하이퍼파라미터 튜닝

**XGBoost**:
- `n_estimators`: [100, 200, 300, 500]
- `max_depth`: [5, 7, 9, 11]
- `learning_rate`: [0.01, 0.05, 0.1]
- `subsample`: [0.7, 0.8, 0.9, 1.0]

**DNN**:
- `latent_dim`: [32, 64, 128]
- `dropout_rate`: [0.2, 0.3, 0.4]
- `learning_rate`: [0.0001, 0.001, 0.01]
- Hidden layer sizes: 실험적으로 조정

**Fusion**:
- Layer sizes: [64, 32] vs [128, 64, 32]
- `dropout_rate`: [0.2, 0.3, 0.4, 0.5]

### 2. Feature Engineering 개선

- **도메인 특징 생성**:
  - 곡선 통계량 (평균, 표준편차, 최대/최소, 기울기)
  - x, y, p 간의 상호작용 특징
  - Fourier transform 계수 (주파수 도메인)

- **차원 축소**:
  - PCA로 x0~x255 압축
  - Autoencoder로 고차원 특징 압축

### 3. 앙상블 확장

- **다양한 Boosting 모델**:
  - CatBoost (범주형 특징 처리 우수)
  - LightGBM (빠른 학습 속도)

- **Model Stacking**:
  - Level 1: XGBoost, CatBoost, LightGBM, DNN
  - Level 2: Logistic Regression / Simple MLP

### 4. Cross-Validation

현재: Single train/val split (80/20)
개선: 5-Fold Stratified CV로 더 강건한 평가

```python
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    # 각 fold마다 모델 학습 및 평가
    ...
```

### 5. 클래스 불균형 처리

만약 NG/Good 비율이 불균형하다면:
- **Oversampling**: SMOTE
- **Undersampling**: Random undersampling
- **Class Weights**: `scale_pos_weight` (XGBoost), `pos_weight` (BCE Loss)

---

## 📚 제조 AI 모범 사례 (Manufacturing AI Best Practices)

### ✅ 구현된 사항

1. **도메인 지식 반영**:
   - Design vs Simulation 특징 분리
   - 각 특징 그룹에 적합한 모델 선택

2. **재현성 (Reproducibility)**:
   - 모든 random seed 고정
   - 버전 관리 가능한 구조

3. **모듈화 (Modularity)**:
   - 독립적인 전처리 파이프라인
   - 재사용 가능한 컴포넌트

4. **추적성 (Traceability)**:
   - 모든 단계에서 verbose 로그
   - 모델 아티팩트 저장

5. **생산 준비 (Production-Ready)**:
   - End-to-end 추론 파이프라인
   - 모델 직렬화 및 로딩

### 🎯 추가 권장사항

1. **모델 모니터링**:
   - Prediction drift 감지
   - Feature distribution 변화 추적

2. **A/B 테스팅**:
   - 새 모델 vs 기존 모델 성능 비교
   - 점진적 배포

3. **설명 가능성 (Explainability)**:
   - SHAP values로 예측 설명
   - Feature importance 시각화

4. **데이터 버전 관리**:
   - DVC (Data Version Control) 사용
   - 학습 데이터 스냅샷 관리

---

## 🔗 참고 자료

- **XGBoost Documentation**: https://xgboost.readthedocs.io/
- **PyTorch Documentation**: https://pytorch.org/docs/
- **Scikit-learn User Guide**: https://scikit-learn.org/stable/user_guide.html
- **Manufacturing AI Papers**:
  - "Deep Learning for Smart Manufacturing" (2018)
  - "Hybrid Models for Quality Prediction in Manufacturing" (2020)

---

## 📞 문의 및 지원

프로젝트 관련 문의:
- 작성자: 손주호
- 날짜: 2025
- 용도: 데이터 대회 / 타이어 불량 예측

---

**Document Version**: 1.0
**Last Updated**: 2025-12-06
**Architecture Type**: Hybrid Stacking Ensemble (XGBoost + DNN)

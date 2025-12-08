# Main Model Documentation - Tire Defect Prediction

## 📊 프로젝트 개요

**파일명**: `main copy.ipynb`
**목적**: 타이어 불량 예측 및 최적 선별 의사결정
**문제 유형**: Binary Classification with Business Optimization
**핵심 목표**: Profit 최대화를 위한 불량 확률 예측 및 임계값 최적화

---

## 🎯 비즈니스 목표: Profit 정의

### ⚠️ **PROFIT 계산식 (핵심 개념)**

이 프로젝트의 가장 중요한 평가 지표는 **Profit(이익)**입니다.

```python
# Profit 계산 공식
Profit = (TN × 100) - (FN × 2000) - Penalty

where:
    TN  = True Negative  (Good을 Good으로 올바르게 예측)
    FN  = False Negative (NG를 Good으로 잘못 예측)
    Penalty = 99,999 (if 선택된 제품 수 > 200개 기준)
```

### 💰 **세부 비용 구조**

| 예측 결과 | 실제 상태 | 의사결정 | 비용/수익 | 설명 |
|-----------|----------|----------|-----------|------|
| **Good 예측 (선택)** | Good | 출하 | **+100원** | 정상 제품 판매 수익 |
| **Good 예측 (선택)** | NG | 출하 | **-2,000원** | 불량품 출하 → 고객 클레임 손실 |
| **NG 예측 (거부)** | NG | 폐기 | 0원 | 불량품 사전 차단 (손실 없음) |
| **NG 예측 (거부)** | Good | 폐기 | 0원 | 정상품 폐기 (기회비용, 계산 미포함) |

### 📌 **제약 조건 (Penalty)**

```
IF 선택된 제품 수 > (200 × 전체 데이터 수 / 466):
    Profit -= 99,999
```

- **의미**: 너무 많은 제품을 선택하면 대량 패널티 부과
- **비율**: 전체 데이터의 약 42.9% (200/466) 초과 시
- **전략**: 보수적으로 선택하여 고품질 제품만 출하

### 🎲 **Expected Profit (기댓값 기반 최적화)**

실제 검증 데이터가 없을 때, 예측 확률로 기댓값을 계산:

```python
# 각 샘플의 기댓값
Expected_Profit_per_sample = (1 - p) × 100 - p × 2000

where:
    p = 모델이 예측한 불량 확률 (0~1)

# 전체 기댓값
Total_Expected_Profit = Σ(Expected_Profit_per_sample) - Penalty
```

**해석**:
- 불량 확률 `p = 0.01` (1%) → 기댓값 = 99 - 20 = **+79원**
- 불량 확률 `p = 0.05` (5%) → 기댓값 = 95 - 100 = **-5원** (손실!)
- 불량 확률 `p = 0.10` (10%) → 기댓값 = 90 - 200 = **-110원** (큰 손실)

**임계값 예시**:
- `p < 0.048` → 기댓값 > 0 (선택 권장)
- `p ≥ 0.048` → 기댓값 ≤ 0 (거부 권장)

---

## 🏗️ 모델 아키텍처

### 1️⃣ **데이터 구조**

#### 입력 데이터

**Feature Groups**:
1. **Summary Features (X_sum)**: 설계 및 공정 파라미터
   - `Mass_Pilot`: 양산/파일럿 구분 (Boolean)
   - `Width`, `Aspect`, `Inch`: 타이어 규격
   - `Plant`: 생산 공장
   - `Proc_Param1` ~ `Proc_Param11`: 공정 파라미터
   - `G1` ~ `G4`: 통계 특징

2. **FEM Features (X_fem)**: 시뮬레이션 곡선 데이터
   - `x0` ~ `x255`: X좌표 (256개 포인트)
   - `y0` ~ `y255`: Y좌표 (256개 포인트)
   - `p0` ~ `p255`: 압력 값 (256개 포인트)

**Target Variable**:
- `Class`: 'Good' or 'NG' (Binary)

#### 데이터 분할 함수

```python
def split_data(df, train=True):
    """
    데이터를 Summary 특징과 FEM 특징으로 분리

    Returns:
        - X_sum: 요약 특징 (설계/공정 파라미터)
        - X_fem: FEM 시뮬레이션 특징 (x, y, p 곡선)
        - y (train=True) or ids (train=False)
    """
```

---

### 2️⃣ **Feature Engineering**

#### 범주형 변수 처리

```python
def numerize(X_sum, oe=None):
    """
    범주형 특징을 One-Hot Encoding으로 변환

    Categorical Features:
        - Mass_Pilot (Boolean)
        - Plant (공장 코드)
        - Proc_Param6 (범주형 공정 파라미터)

    Returns:
        - X: One-Hot Encoded + Numerical Features
        - oe: OneHotEncoder 객체 (재사용용)
    """
```

#### FEM 특징 추출 (선택 사항)

고차원 FEM 데이터(768차원)를 저차원 통계 특징으로 압축:

```python
fem_feat_fns = [
    lambda x, y, p: p.max(axis=-1),    # 최대 압력
    curve_length,                       # 곡선 전체 길이
    stress_length,                      # 고응력 구간 길이 (p > 2.5)
    bend_extent,                        # 굽힘 정도 (y_range / x_range)
    max_curvature                       # 최대 곡률
]
```

**함수 세부 설명**:

1. **`curve_length(x, y, p)`**: 전체 곡선 길이
   ```python
   # 인접 포인트 간 거리 합
   L = Σ√((x[i+1] - x[i])² + (y[i+1] - y[i])²)
   ```

2. **`stress_length(x, y, p, p_thr=2.5)`**: 고응력 구간 길이
   ```python
   # 압력 > 2.5인 포인트만 선택하여 길이 계산
   high_stress_points = points[p > 2.5]
   L_stress = calculate_length(high_stress_points)
   ```

3. **`bend_extent(x, y, p)`**: 굽힘 비율
   ```python
   # Y 범위를 X 범위로 나눔
   bend_ratio = (y_max - y_min) / (x_max - x_min)
   ```

4. **`max_curvature(x, y, p)`**: 최대 곡률
   ```python
   # 곡률 공식: κ = |dx·ddy - dy·ddx| / (dx² + dy²)^(3/2)
   curvature = |x'y'' - y'x''| / (x'² + y'²)^1.5
   max_κ = max(curvature)
   ```

---

### 3️⃣ **모델 선택 및 학습**

#### 지원 모델

1. **CatBoost** (범주형 특징 직접 처리)
   ```python
   CatBoostClassifier(
       cat_features=['Plant', 'Proc_Param6'],
       verbose=0
   )
   ```

2. **XGBoost** (권장)
   ```python
   XGBClassifier(
       n_estimators=500,
       scale_pos_weight=1  # 클래스 균형 조정
   )
   ```

3. **Random Forest**
   ```python
   RandomForestClassifier(
       n_estimators=1200,
       max_features="sqrt",
       bootstrap=True,
       criterion="log_loss"
   )
   ```

#### 하이퍼파라미터 튜닝

**RandomizedSearchCV로 최적 파라미터 탐색**:

```python
param_dist = {
    "n_estimators": [300, 500, 800, 1200],
    "max_features": ["sqrt", "log2", None, 0.1~1.0],
    "max_depth": [None, 5, 10, 15, 20, 25, 30],
    "min_samples_split": [2, 5, 10, 20, 50],
    "min_samples_leaf": [1, 2, 4, 8, 16],
    "bootstrap": [True, False],
    "class_weight": [None, "balanced", "balanced_subsample"],
    "criterion": ["gini", "entropy", "log_loss"]
}
```

**최적 파라미터 (예시 결과)**:
```python
{
    'n_estimators': 500,
    'min_samples_split': 50,
    'min_samples_leaf': 4,
    'max_features': 'log2',
    'max_depth': 30,
    'criterion': 'log_loss',
    'class_weight': None,
    'bootstrap': False
}
```

---

### 4️⃣ **클래스 불균형 처리 (선택 사항)**

```python
def apply_smote(X, y, random_state=42):
    """
    SMOTE (Synthetic Minority Over-sampling Technique)
    소수 클래스(NG)의 합성 샘플 생성으로 균형 맞춤
    """
    smote = SMOTE(random_state=random_state)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    return X_resampled, y_resampled
```

**주의사항**:
- SMOTE는 학습 데이터에만 적용
- 검증/테스트 데이터는 원본 분포 유지
- Profit 최적화 시 실제 분포가 중요하므로 신중히 사용

---

## 📈 평가 및 시각화

### 1️⃣ **성능 지표**

#### 기본 분류 지표

```python
def print_result(model, X, y_true):
    """
    모델 성능 출력:
    - Accuracy: (TP + TN) / Total
    - Recall (TPR): TP / (TP + FN)
    - Precision: TP / (TP + FP)
    - NPV: TN / (TN + FN)  ← 특히 중요!
    - F1 Score: 2 × (Recall × Precision) / (Recall + Precision)
    """
```

**NPV (Negative Predictive Value)**:
- **정의**: Good으로 예측한 것 중 실제 Good 비율
- **의미**: 선택한 제품의 품질 신뢰도
- **목표**: NPV를 높여 불량품 출하 최소화

---

### 2️⃣ **ROC/PRC 곡선**

```python
def plot_curves(p_hat, y_true, ax, curve_type='auroc', n=100):
    """
    분류 성능 곡선 시각화

    curve_type:
        - 'auroc': ROC Curve (FPR vs TPR)
        - 'auprc': Precision-Recall Curve
        - 'NPV': NPV vs Threshold
        - 'profit': Profit vs Threshold
    """
```

**AUROC (Area Under ROC Curve)**:
- **범위**: 0.5 ~ 1.0
- **의미**: 0.5 = 랜덤 예측, 1.0 = 완벽한 분류
- **목표**: 0.75 이상

**AUPRC (Area Under Precision-Recall Curve)**:
- 불균형 데이터에서 더 민감한 지표
- Precision과 Recall의 트레이드오프 시각화

---

### 3️⃣ **Profit 곡선 (핵심)**

#### Actual Profit Curve

```python
# 실제 레이블 기반 Profit 계산
plot_curves(p_hat, y_true, ax, curve_type='profit', n=2000)
```

**계산 로직**:
```python
for threshold in [0, 0.001, ..., 1.0]:
    y_pred = (p_hat < threshold)  # Good 예측 (선택)

    TP = (y_pred == 1) & (y_true == 1)  # 미사용 (FP와 동일 취급)
    TN = (y_pred == 1) & (y_true == 0)  # Good 맞춤 → +100
    FP = (y_pred == 0) & (y_true == 1)  # 미사용
    FN = (y_pred == 0) & (y_true == 0)  # NG 놓침 → -2000

    profit = TN × 100 - FN × 2000 - penalty

    if TN + FN > (200 × N / 466):
        penalty = 99999
```

#### Expected Profit Curve (추론용)

```python
plot_profit(p_hat, y_true=None, ax, quantile=[0.05, 0.95], n=1000)
```

**기댓값 계산**:
```python
for threshold in [0, 0.001, ..., 1.0]:
    selected = (p_hat < threshold)
    p_selected = p_hat[selected]

    # 각 샘플의 기댓값
    profit_per_sample = (1 - p_selected) × 100 - p_selected × 2000

    # 전체 기댓값
    expected_profit = profit_per_sample.sum() - penalty

    # 표준편차 (불확실성)
    std_profit = 2100 × √(Σ(p × (1 - p)))
```

**출력 예시**:
```
Expected Optimal Profit: 12,345(±1,234) at thr=0.016
Decision Profit: 11,890 at thr=0.016
```

**시각화 요소**:
- **검은색 실선**: 기댓값 평균
- **회색 영역**: 신뢰구간 (5%, 25%, 75%, 95% quantile)
- **최적 임계값**: Expected Profit이 최대인 지점

---

### 4️⃣ **Calibration Plot (확률 보정 검증)**

```python
# 예측 확률 vs 실제 비율 비교
bins = np.linspace(0, 0.1, 11)
freq_positive = histogram(p_hat[y_true == 1], bins)
freq_negative = histogram(p_hat[y_true == 0], bins)

ratio_actual = freq_positive / (freq_positive + freq_negative)
p_predicted = (bins[1:] + bins[:-1]) / 2

plt.plot(p_predicted, ratio_actual)  # 이상적으로 y=x
```

**해석**:
- **y = x 선 위**: 모델이 확률을 과소평가 (안전)
- **y = x 선 아래**: 모델이 확률을 과대평가 (위험)
- **목표**: 대각선에 가깝게 (Well-calibrated)

---

### 5️⃣ **SHAP Analysis (Feature Importance)**

```python
import shap

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# NG 클래스에 대한 기여도
shap_value_pos = shap_values[:, :, 1]
shap.summary_plot(shap_value_pos, X_test, plot_type="dot")
```

**해석**:
- **빨간색 점**: 높은 특징값이 NG 예측에 기여
- **파란색 점**: 낮은 특징값이 NG 예측에 기여
- **상위 특징**: 가장 영향력 있는 변수

---

## 🔄 Cross-Validation

```python
def cross_validate(model, X, y, cv=5, seed=None):
    """
    K-Fold Cross-Validation으로 일반화 성능 평가

    Parameters:
        cv: Fold 개수 (기본 5, 더 엄격한 평가는 10~15)

    Returns:
        ps: 각 Fold의 예측 확률 (cv, n_val)
        ys: 각 Fold의 실제 레이블 (cv, n_val)
    """
```

**장점**:
- 모든 데이터를 검증에 활용
- 과적합 조기 발견
- 모델 안정성 평가

---

## 🎯 최종 점수 (Final Score)

```python
def final_score(auroc, profit_mean, profit_std):
    """
    대회 최종 평가 지표

    Formula:
        Score = √(max(AUROC - 0.5, 0) / 0.5 × max(Profit, 0) / 20,000)

    Components:
        - AUROC: 분류 성능 (0.5~1.0 정규화)
        - Profit: 비즈니스 가치 (0~20,000 정규화)

    Range: 0 ~ 1
    """
    return np.sqrt(
        max(auroc - 0.5, 0) / 0.5 *
        max(profit_mean, 0) / 20000
    )
```

**해석**:
- AUROC와 Profit의 기하평균 (√ 사용)
- 둘 중 하나라도 낮으면 점수 크게 하락
- **목표**: AUROC > 0.75, Profit > 10,000 → Score > 0.5

---

## 📊 추론 및 제출

### 1️⃣ **테스트 데이터 예측**

```python
# 1. 테스트 데이터 로드
df_exam = pd.read_csv('data/test.csv')

# 2. 전처리
X_sum_exam, X_fem_exam, ids = split_data(df_exam, train=False)
X_sum_exam = numerize(X_sum_exam, oe=oe)

# 3. 예측
p_exam = model.predict_proba(X_sum_exam)[:, 1]

# 4. 최적 임계값으로 의사결정
thr_optimal = 0.016  # Expected Profit 곡선에서 결정
decision = (p_exam < thr_optimal)
```

---

### 2️⃣ **제출 파일 생성**

```python
import datetime

# 현재 시간 (KST)
now = datetime.datetime.now(
    tz=datetime.timezone(datetime.timedelta(hours=9))
).strftime("%m-%d-%H-%M")

# 제출 양식 로드
submission = pd.read_csv('data/sample_submission.csv')

# 예측 결과 할당
submission['probability'] = np.concatenate([p_exam, p_exam])
submission['decision'] = np.concatenate([decision, decision])

# 저장
submission.to_csv(f"submission_HAIYONG_{now}.csv", index=False)
```

**파일 형식**:
```csv
ID,probability,decision
0,0.0123,True
1,0.0456,True
2,0.0789,False
...
```

---

## 🚀 모델 성능 개선 전략

### 1️⃣ **Feature Engineering**

✅ **구현된 기법**:
- One-Hot Encoding (범주형 변수)
- FEM 통계 특징 추출

🔧 **추가 가능 기법**:
- Polynomial Features (변수 간 상호작용)
- Target Encoding (범주형 변수 → 평균 불량률)
- Time-series features (공정 순서 정보)
- Dimensionality Reduction (PCA, t-SNE for FEM data)

---

### 2️⃣ **모델 앙상블**

**Stacking**:
```python
# Level 1 모델들
models_L1 = [
    XGBClassifier(...),
    CatBoostClassifier(...),
    RandomForestClassifier(...)
]

# Level 2 메타 모델
meta_model = LogisticRegression()
```

**Voting**:
```python
from sklearn.ensemble import VotingClassifier

ensemble = VotingClassifier(
    estimators=[
        ('xgb', XGBClassifier(...)),
        ('cat', CatBoostClassifier(...)),
        ('rf', RandomForestClassifier(...))
    ],
    voting='soft'  # 확률 평균
)
```

---

### 3️⃣ **임계값 최적화**

**Grid Search for Threshold**:
```python
thresholds = np.linspace(0.01, 0.05, 100)
profits = []

for thr in thresholds:
    decision = (p_val < thr)
    profit = calculate_profit(decision, y_val)
    profits.append(profit)

thr_optimal = thresholds[np.argmax(profits)]
```

**Bayesian Optimization**:
- 임계값을 연속 변수로 최적화
- Expected Profit 최대화 목표

---

### 4️⃣ **확률 보정 (Calibration)**

```python
from sklearn.calibration import CalibratedClassifierCV

# Platt Scaling 또는 Isotonic Regression
calibrated_model = CalibratedClassifierCV(
    model,
    method='sigmoid',  # or 'isotonic'
    cv=5
)

calibrated_model.fit(X_train, y_train)
p_calibrated = calibrated_model.predict_proba(X_test)[:, 1]
```

**효과**: 예측 확률이 실제 비율과 일치 → Expected Profit 신뢰도 향상

---

## 📁 파일 구조

```
Code_손주호/
├── data/
│   ├── train.csv              # 학습 데이터
│   ├── test.csv               # 테스트 데이터
│   └── sample_submission.csv  # 제출 양식
├── documents/
│   ├── Architecture_Overview.md      # Hybrid 모델 아키텍처 문서
│   └── Main_Model_Documentation.md   # 본 문서
├── result/
│   └── {timestamp}_{model_name}.csv  # 예측 결과 파일
├── main copy.ipynb            # 본 노트북
└── tire_defect_prediction.ipynb  # Hybrid DL 모델 노트북
```

---

## 🔑 핵심 요약

### **Profit 최적화 체크리스트**

✅ **모델 학습**
- [ ] AUROC > 0.75 달성
- [ ] NPV > 0.95 달성 (낮은 임계값에서)
- [ ] Calibration Plot 검증

✅ **임계값 설정**
- [ ] Expected Profit 곡선 분석
- [ ] 최적 임계값 = 0.015 ~ 0.025 범위
- [ ] 선택 제품 수 < 200개 기준 확인

✅ **검증**
- [ ] Cross-Validation Score 안정적
- [ ] SHAP으로 feature 해석 가능
- [ ] Test set 예측 분포 확인

✅ **제출**
- [ ] `probability` 컬럼: 0~1 범위
- [ ] `decision` 컬럼: True/False
- [ ] 파일명: `submission_HAIYONG_{timestamp}.csv`

---

## 📚 참고 자료

### 주요 라이브러리

- **scikit-learn**: https://scikit-learn.org/
- **XGBoost**: https://xgboost.readthedocs.io/
- **CatBoost**: https://catboost.ai/
- **SHAP**: https://shap.readthedocs.io/
- **imbalanced-learn**: https://imbalanced-learn.org/

### 유용한 논문/자료

- *"Calibration of Machine Learning Models"* (2019)
- *"Cost-Sensitive Learning for Imbalanced Classification"* (2020)
- *"SHAP: A Unified Approach to Interpreting Model Predictions"* (2017)

---

## 🎓 학습 포인트

### 이 프로젝트에서 배울 수 있는 것

1. **비즈니스 중심 머신러닝**
   - 분류 정확도보다 비즈니스 가치(Profit) 우선
   - 비용-편익 분석을 모델에 통합

2. **확률 해석**
   - 예측 확률의 Calibration 중요성
   - Expected Value 기반 의사결정

3. **불균형 데이터 처리**
   - SMOTE, Class Weights 활용
   - NPV 같은 대안 지표 사용

4. **모델 해석**
   - SHAP으로 블랙박스 모델 설명
   - Feature Importance 시각화

5. **엔지니어링 우수 사례**
   - 모듈화된 전처리 파이프라인
   - 재사용 가능한 평가 함수
   - 자동화된 결과 저장

---

**Document Version**: 1.0
**Author**: 손주호
**Last Updated**: 2025-12-08
**Model Type**: XGBoost / CatBoost / Random Forest
**Optimization Target**: Profit Maximization

# 기술 소개

## 1. 라이브러리 및 프레임워크

### Data Analysis / Machine Learning

* **Pandas**
* **NumPy**
* **Matplotlib**
* **Scikit-learn**
* **Optuna / Hyperopt**

### Deep Learning / Computer Vision

* **PyTorch**
* **TensorFlow**
* **OpenCV**

### Application

* **FastAPI**
* **Streamlit**
* **Git**
* **Docker**

---

## 2. Related Competition Experience

### 데이콘 Basic 스트레스 지수 예측

| 항목 | 내용 |
|---|---|
| 과제 유형 | Tabular Regression task |
| 참여 형태 | 개인 |
| 평가 지표 | MAE |
| 데이터 크기 | train 3,000 × 18 / test 3,000 × 17 |
| Target | `stress_score` (연속형, 0.0 ~ 1.0) |
| GitHub | [Dacon_stress_score_predic](https://github.com/yunjihyung/Dacon_stress_score_predic) |

---

## 3. 데이터 구성 및 결측치 현황

원본 train 데이터는 18개 컬럼으로, **신체 계측 / 혈액검사 / 생활습관 / 사회경제** 정보가 섞여 있는 형태였습니다.

| 컬럼 | 타입 | 결측 수 | 성격 |
|---|---|---:|---|
| `ID` | object | 0 | 식별자 (drop) |
| `gender` | object | 0 | 범주형 (F/M) |
| `age` | int | 0 | 수치형 |
| `height` | float | 0 | 신체 계측 |
| `weight` | float | 0 | 신체 계측 |
| `cholesterol` | float | 0 | 혈액검사 |
| `systolic_blood_pressure` | int | 0 | 혈액검사 (수축기 혈압) |
| `diastolic_blood_pressure` | int | 0 | 혈액검사 (이완기 혈압) |
| `glucose` | float | 0 | 혈액검사 (혈당) |
| `bone_density` | float | 0 | 신체 계측 |
| `activity` | object | 0 | 순서형 (light/moderate/intense) |
| `smoke_status` | object | 0 | 순서형 (non/ex/current-smoker) |
| **`medical_history`** | object | **1,289** | 범주형 (본인 병력) |
| **`family_medical_history`** | object | **1,486** | 범주형 (가족력) |
| `sleep_pattern` | object | 0 | 순서형 |
| **`edu_level`** | object | **607** | 순서형 (학력) |
| **`mean_working`** | float | **1,032** | 수치형 (평균 근로시간) |
| `stress_score` | float | 0 | **Target** |

**결측치 처리 판단**

* 결측이 있는 4개 컬럼은 전체의 **20~50%**에 달해, 단순 행 제거(dropna)를 하면 3,000개 중 절반 가까이가 사라져 사용 불가.
* `medical_history` / `family_medical_history` / `edu_level` → 결측 자체가 **"병력 없음 / 미응답"이라는 정보**일 수 있다고 보고, 삭제나 최빈값 대치 대신 **`'Unknown'`이라는 하나의 독립 범주**로 남김.
* `mean_working` → 근로시간 결측은 **"근로하지 않음"** 으로 해석해 `0`으로 대치.

```python
df['mean_working'] = df['mean_working'].fillna(0)   # 결측 = 미취업으로 해석
df = df.fillna('Unknown')                           # 나머지는 독립 범주로 보존
```

---

## 4. 인코딩 (Encoding)

범주형 변수를 **순서가 있는 것 / 없는 것**으로 나누어 다르게 처리했습니다.

**(1) Ordinal Encoding** — 크기 관계가 실제로 존재하는 변수

```python
d_activity      = {"light": 0, "moderate": 1, "intense": 2}
d_smoke_status  = {"non-smoker": 0, "ex-smoker": 1, "current-smoker": 2}
d_edu_level     = {'Unknown': 0, 'high school diploma': 1,
                   'bachelors degree': 2, 'graduate degree': 3}
d_sleep_pattern = {'sleep difficulty': 0, 'normal': 1, 'oversleeping': 2}

df['activity']  = df['activity'].map(d_activity)
df['edu_level'] = df['edu_level'].map(d_edu_level)
```

**(2) One-Hot Encoding** — 순서 관계가 없는 병력 변수

```python
mh_dummies  = pd.get_dummies(df['medical_history'],        prefix="mh",  dtype='int')
fmh_dummies = pd.get_dummies(df['family_medical_history'], prefix="fmh", dtype='int')
df = pd.concat([df, mh_dummies, fmh_dummies], axis=1)
df = df.drop(["ID", 'medical_history', 'family_medical_history'], axis=1)
```

* `medical_history`는 `diabetes / heart disease / high blood pressure / Unknown` 4종 → 숫자로 매핑하면 "당뇨 < 심장병" 같은 **없는 순서 관계가 생기므로** One-Hot 선택.
* 결과: **18개 컬럼 → 25개 컬럼**으로 확장.
* **train / test에 동일한 매핑 딕셔너리를 그대로 재사용**하여 컬럼 불일치 방지.

---

## 5. Feature Engineering — Domain Feature 생성

의료 도메인 기준치를 이용해, 모델이 스스로 찾기 어려운 **비선형 임계값(threshold)** 을 명시적인 파생변수로 만들어 실험했습니다.

### (1) BMI — 채택 ✅

키와 몸무게를 **각각** 주는 것보다, 비만도라는 하나의 축으로 압축한 값이 스트레스와 더 직접적일 것으로 보고 생성.

```python
df['bmi']      = (df['weight']      / ((df['height']      / 100.0) ** 2)).round(2)
test_df['bmi'] = (test_df['weight'] / ((test_df['height'] / 100.0) ** 2)).round(2)
```

### (2) 고혈압 / 고혈당 / 고콜레스테롤 플래그 — 실험 후 제외 ❌

혈압·혈당은 **연속적으로 나빠지는 게 아니라 특정 기준선을 넘는 순간 "질환"** 이 되므로, 임상 기준치로 이진 플래그를 만들어 봤습니다.

```python
# 고콜레스테롤 여부 (총콜레스테롤 > 230 mg/dL)
df['is_high_chol'] = (df['cholesterol'] > 230).astype(int)

# 고혈압 여부 (수축기 > 130 or 이완기 > 80 mmHg)
df['is_high_bp'] = ((df['systolic_blood_pressure']  > 130) |
                    (df['diastolic_blood_pressure'] > 80)).astype(int)

# 고혈당(당뇨 위험) 여부 (공복혈당 > 126 mg/dL)
df['is_high_glucose'] = (df['glucose'] > 126).astype(int)
```

### (3) 근로시간 구간화 `working_category` — 실험 후 제외 ❌

`mean_working`(일 평균 근로시간)을 그대로 쓰면 "0시간"과 "8시간"이 단순 크기 차이로만 취급됩니다. 실제로는 **미취업 / 파트타임 / 정규 근로 / 초과근무**라는 질적으로 다른 상태이므로, 주당 근로시간 기준으로 구간화했습니다.

```python
mw = df['mean_working']
weekly_hours = mw * 7

# 구간 설정 (미취업 / 파트 / 정규 / 초과)
conditions = [
    (mw == 0),                                    # 무직 (혹은 결측)
    (weekly_hours < 28) & (mw != 0),              # 파트타임
    (weekly_hours < 41) & (weekly_hours >= 28),   # 정규 근로
    (weekly_hours >= 41)                          # 초과 근무
]
choices = ['none', 'part_time', 'full_time', 'overwork']
df['working_category'] = np.select(conditions, choices)

df['working_category'] = df['working_category'].map({
    'none': 0, 'part_time': 1, 'full_time': 2, 'overwork': 3})
```
* 근로 강도는 순서가 있는 구간이므로 One-Hot이 아닌 **0/1/2/3 Ordinal 매핑**.

### Feature 선택 방식

> 파생변수를 추가할 때마다 **K-Fold Validation MAE를 다시 측정**하여, 실제로 성능이 개선된 feature만 남기는 방식으로 검증했습니다.

| 파생 Feature | 생성 근거 | 채택 여부 |
|---|---|---|
| `bmi` | 신장·체중을 비만도 한 축으로 압축 | ✅ 채택 (CV MAE 개선) |
| `is_high_bp` | 임상 기준 130/80 mmHg | ❌ 제외 (개선 없음) |
| `is_high_glucose` | 공복혈당 기준 126 mg/dL | ❌ 제외 (개선 없음) |
| `is_high_chol` | 총콜레스테롤 230 mg/dL | ❌ 제외 (개선 없음) |
| `working_category` | 근로 형태 4구간 | ❌ 제외 (개선 없음) |

**왜 제외했는가:** 이진 플래그는 원본 연속형 변수(`glucose`, `blood_pressure`)에서 파생된 값이라 **정보량이 새로 늘어나지 않고**, RBF Kernel SVR은 이미 비선형 경계를 학습할 수 있어 임계값을 수동으로 알려주는 이득이 크지 않았습니다. 오히려 차원만 늘어나 거리 계산이 희석되는 부작용이 있었습니다.

→ **"도메인 지식으로 만든 feature가 항상 성능을 올리지는 않는다"** 는 것을 검증 지표로 확인하고, 최종적으로는 **BMI만 채택**했습니다.

---

## 6. Scaling — 왜 RobustScaler인가

### (1) 스케일링이 필요한 모델 / 필요 없는 모델

먼저 "이 모델이 스케일에 민감한가"를 확인했습니다. 모델이 내부에서 **거리·내적·계수 크기**를 쓰면 스케일에 민감하고, **대소 비교(순서)만** 쓰면 둔감합니다.

| 구분 | 모델 | 왜 그런가 |
|---|---|---|
| **매우 민감** | **SVM** | 커널이 `exp(-γ‖xᵢ−xⱼ‖²)` — 유클리드 **거리**를 직접 계산 |
| **매우 민감** | KNN, K-Means, DBSCAN | 이웃·군집 판정이 전부 거리 기반 |
| 민감 | Ridge / Lasso / ElasticNet | 페널티가 **계수 크기**에 걸려, 스케일이 큰 변수가 부당하게 강한 규제를 받음 |
| 민감 | 신경망(MLP), 경사하강 기반 회귀 | 스케일이 제각각이면 손실 곡면이 길쭉해져 **수렴이 느리고 불안정** |
| **둔감** | Decision Tree, RandomForest, XGBoost, LightGBM | 분할 기준이 `x > t` 형태의 **순서 비교**뿐이라 단위 변환에 불변 |

> 이번에 선택한 **RBF SVR은 위 표에서 가장 민감한 부류**입니다. `age`(20~80)와 `cholesterol`(150~300)이 섞여 있으면 커널 거리가 사실상 콜레스테롤 한 변수에 의해 결정됩니다. 그래서 Scaling은 선택이 아니라 **성능을 좌우하는 필수 단계**였습니다.

### (2) 어떤 Scaler를 쓸 것인가

| Scaler | 기준 통계량 | 특징 / 판단 |
|---|---|---|
| **StandardScaler** | 평균 · 표준편차 | 평균과 표준편차 **둘 다 이상치에 끌려감**. 혈압·혈당에 극단값이 있어 부적합 |
| **MinMaxScaler** | 최솟값 · 최댓값 | [0,1]로 고정되지만 **최댓값 하나가 전체 스케일을 결정**. 이상치에 가장 취약 |
| **RobustScaler** | **중앙값 · IQR** | 사분위수는 극단값에 거의 흔들리지 않음 → **채택** ✅ |

```python
from sklearn.preprocessing import RobustScaler
# (x - median) / IQR  →  이상치의 영향을 받지 않는 중심화 · 스케일링
```

$$
x' = \frac{x - \mathrm{Median}(X)}{Q_3 - Q_1}
$$


* 혈압·혈당·콜레스테롤에 극단값이 존재 → 평균·표준편차 기반 스케일링은 **정상 범위 데이터를 좁은 구간에 뭉치게** 만듦.
* **RobustScaler는 중앙값과 IQR(Q3−Q1)** 을 쓰기 때문에 상·하위 극단값에 강건.
* 세 Scaler를 동일 조건에서 CV 비교했을 때 **RobustScaler의 MAE가 가장 낮았습니다.**

⚠️ **Data Leakage 방지:** Scaler를 전체 데이터에 미리 fit하면 Validation 정보가 새어 들어갑니다. 따라서 `make_pipeline`으로 묶어 **각 Fold의 train 부분에서만 fit** 되도록 구성했습니다.

---

## 7. Target Transformation — 균등분포 Target을 정규분포로

SVR은 Feature의 Scale뿐만 아니라 Target Scale의 영향도 받을 수 있습니다. 특히 SVR의 epsilon, C와 같은 Hyperparameter는 Target의 값 크기와 함께 영향을 받기 때문에, Target을 다른 형태로 변환했을 때 성능이 달라지는지 실험했습니다.


### (1) 그러면 균등분포 → 정규분포 변환은 무엇을 바꾸는가

`QuantileTransformer(output_distribution="normal")`는 **순위(rank)를 보존한 채 값의 간격을 재배치**하는 변환입니다.
```
원본(균등)   0.0    0.1    0.3    0.5    0.7    0.9    1.0
                ↓  Quantile → normal  ↓
변환 후     -5.2   -1.28  -0.52   0.0   +0.52  +1.28  +5.2
            └─ 양끝은 크게 벌어짐 ─┘ └ 중앙은 촘촘하게 압축 ┘
```

1. **중앙부는 압축되고, 양 끝은 크게 늘어납니다.** 스트레스가 매우 낮거나(0 근처) 매우 높은(1 근처) 샘플들이 변환 공간에서 서로 멀어져, **모델이 극단 구간을 구분하는 데 더 큰 손실 가중치**를 받습니다.

2. **경계 포화(boundary saturation) 문제가 완화됩니다.** 타깃이 [0, 1]로 잘려 있는데 SVR의 출력은 제한이 없어 음수나 1 초과를 예측할 수 있고, 경계 근처 예측이 안쪽으로 수축되는 경향이 있습니다. 정규 공간에서는 경계가 ±∞ 쪽으로 밀려나 이 압박이 줄어듭니다.

**이론만으로는 결론이 안 나므로 A/B로 측정했습니다.**

### (2) 비교 실험 결과
| 조건 | 구성 | 결과 |
|---|---|---|
| A. 원본 Target | `RobustScaler → SVR(rbf)` | 기준선 |
| B. Quantile 변환 Target | `RobustScaler → QuantileTransformer(normal) → SVR(rbf)` | **CV MAE 개선 → 채택** ✅ |

```python
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import QuantileTransformer

TransformedTargetRegressor(
    regressor=SVR(kernel="rbf", C=..., gamma=..., epsilon=0.0),
    transformer=QuantileTransformer(output_distribution="normal",
                                    n_quantiles=min(1000, len(y_tr)))
)
```

**구현 포인트**

* `TransformedTargetRegressor`가 **학습 시 정변환 / 예측 시 자동 역변환**을 수행 → 예측값이 원래 스트레스 지수 단위로 반환됩니다 (`inverse_transform`을 깜빡하는 실수를 구조적으로 방지).

---

## 8. Model & Hyperparameter Optimization

**모델 비교**

| 계열 | 모델 | 판단 |
|---|---|---|
| Linear | LinearRegression, Ridge, Lasso | 변수 간 비선형 상호작용을 못 잡음 |
| Tree-based | RandomForest, XGBoost, LightGBM | 3,000개 규모에서 과적합 경향 |
| **Kernel** | **SVR (RBF)** | **소규모 tabular + 스케일링/타깃변환과 궁합 → 최종 채택** ✅ |

**탐색 방식**

* **Hyperopt TPE (Tree-structured Parzen Estimator)** 기반 Bayesian Optimization.
* 넓은 범위 1차 탐색으로 대략적인 최적 영역을 찾은 뒤, 그 근방을 **`bestC/3 ~ bestC*3` 로그 스케일로 재탐색(refine)** 하는 2단계 전략.
* 목적함수는 **10-Fold CV MAE의 평균** — 단일 holdout이 아니라 CV 평균을 최소화해 과적합된 파라미터 선택을 방지.

```python
space_refine = {
    "C":     hp.loguniform("C",     np.log(bestC / 3), np.log(bestC * 3)),
    "gamma": hp.loguniform("gamma", np.log(bestG / 3), np.log(bestG * 4)),
}
best = fmin(fn=objective, space=space_refine, algo=tpe.suggest,
            max_evals=80, trials=trials,
            rstate=np.random.default_rng(42))
```

* `hp.loguniform`을 쓴 이유: `C`와 `gamma`는 **자릿수(order of magnitude) 단위로 영향**을 주는 파라미터라, 균등 탐색보다 로그 스케일 탐색이 효율적.

**최종 하이퍼파라미터**

```python
Best_params = {'C': 3.963530707518144, 'gamma': 1.0631617004546035}
```

* **`C`** : 오차 허용 정도와 모델 복잡도의 트레이드오프. 클수록 train 오차를 줄이려 하지만 과적합 위험 증가.
* **`gamma`** : RBF Kernel에서 한 데이터 포인트의 영향 반경. 클수록 결정 경계가 국소적으로 복잡해짐.

---

## 9. 최종 Pipeline 구조

```
전처리 (결측 처리 · Encoding)
      ↓
Feature Engineering (BMI)
      ↓
┌──────────── sklearn Pipeline ────────────┐
│  RobustScaler            (X 스케일링)     │
│        ↓                                 │
│  TransformedTargetRegressor              │
│    ├ transformer : QuantileTransformer   │  ← y 정규화 + 자동 역변환
│    └ regressor   : SVR(kernel='rbf')     │
└──────────────────────────────────────────┘
      ↓
10-Fold CV + Hyperopt TPE (C, gamma 최적화)
      ↓
전체 데이터 재학습 → test 예측 → submission
```

**이 Pipeline 구조의 핵심 장점**

1. **Data Leakage 원천 차단** — Scaler와 Target Transformer가 Pipeline 안에 있어, CV의 각 fold에서 train 부분에만 fit 됩니다.
2. **X 변환과 y 변환의 분리** — `make_pipeline`이 X축(RobustScaler)을, `TransformedTargetRegressor`가 y축(Quantile)을 담당해 역할이 명확합니다.
3. **역변환 자동화** — `pipe.predict()` 결과가 이미 원래 스트레스 지수 단위이므로 별도 후처리가 필요 없습니다.
4. **재현성** — 전처리~예측이 하나의 객체로 묶여 test 데이터에 동일한 변환이 보장됩니다.

---

## 10. 정리 — 이 프로젝트에서 얻은 것

* **결측 20~50% 컬럼**을 삭제하지 않고 `'Unknown'` 범주 / `0` 대치로 정보를 보존하는 판단
* **도메인 지식 기반 파생변수를 만들고, 검증 지표로 채택 여부를 결정**하는 과정 (BMI 채택 / 임계값 플래그·근로구간 제외)
* **모델 특성에 맞는 전처리 선택** — SVR은 거리 기반이므로 Scaler가 성능을 좌우하고, 이상치가 있는 데이터에는 RobustScaler
* **들은 기법을 그대로 쓰지 않고 직접 A/B 실험으로 검증** — Target Transformation의 실제 CV MAE 개선 확인
* **Pipeline으로 Leakage를 구조적으로 차단**하는 습관

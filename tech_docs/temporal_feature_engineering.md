# Temporal Feature Engineering for Sequential Data

Rev. 3 | Created: 2026-09-08 | Updated: 2026-09-13 19:15 KST

## 1. Purpose

- **Problem Statement**: 일반적인 tabular feature는 개별 시점의 상태를 표현하지만, 순차 데이터의 과거 상태·변화량·최근 변동성·추세를 직접 표현하지 못한다.
- **Goal**: 순차 데이터를 고정 길이 feature로 변환하는 Temporal Feature Engineering의 원리, 적용 조건, 선택 기준을 정리한다.
- **Non-Goal**: Derived Variable 설계는 다루지 않으며, 생성된 feature를 이용한 model training 또한 다루지 않는다.

## 2. Summary

Temporal Feature Engineering은 시간 또는 처리 순서가 있는 관측값에서 과거 상태와 동적 패턴을 feature로 변환하여, 일반적인 regression·classification model이 temporal dependency를 사용할 수 있게 하는 방법이다. 기본 설계는 **History-based**, **Window-based**, **Decay-based** feature부터 시작하고, 데이터의 sequence length·sampling structure·feature budget이 충분할 때 **Dependency-based**, **Frequency-based**, **Automated Extraction**으로 확장하는 것이 적절하다.

핵심 원칙은 feature 종류보다 **prediction timestamp에서 사용 가능한 정보의 경계**를 먼저 정의하는 것이다. Lag, rolling, EWMA가 과거 방향으로 계산되어도 실제 예측 시점에 관측할 수 없는 값을 포함하면 leakage가 발생한다. 시계열 forecasting 예제에서도 random split은 time-aware split보다 낙관적인 평가를 만들 수 있으므로 chronological validation이 필요하다 [[1](#ref-1)].

## 3. Principle

### 3.1 Taxonomy

Temporal feature는 무엇을 보존하거나 요약하는지에 따라 구분할 수 있다. Lag와 difference는 개별 과거 상태와 변화량을 보존하고, rolling과 EWMA는 여러 과거 관측을 하나의 recent state로 압축한다. ACF·PACF와 spectral feature는 더 긴 sequence의 구조를 요약하며, automated extraction은 여러 후보 특징을 일괄 생성한다.

Table 1. Temporal feature taxonomy

| Category | Core Features | Captured Information | Main Requirement |
| --- | --- | --- | --- |
| History-based | Lag, Difference, Rate of Change | 과거 상태, 직전 변화량, 변화 속도 | 신뢰 가능한 processing order |
| Window-based | Rolling Mean, Std, Quantile, Slope | 최근 level, variability, trend | 적절한 window length |
| Decay-based | EWMA, EWM Variance, Deviation | 최근 관측에 더 큰 weight를 둔 state | decay parameter |
| Dependency-based | ACF, PACF, Cross-Correlation | lag별 autocorrelation과 변수 간 지연 관계 | 충분한 sequence length |
| Frequency-based | Periodogram, Spectral Power | 반복 주기와 frequency structure | 일정하거나 해석 가능한 sampling interval |
| State/Event-based | Change Flag, Time Since Event, Regime | 상태 전환과 상태 지속기간 | event 또는 state 정의 |
| Automated Extraction | tsfresh, catch22 | 다수의 통계·동역학 특성 | feature selection과 계산 budget |

### 3.2 History-based Features

Lag feature는 과거 관측값 자체를 현재 row의 입력으로 추가한다. 일반적인 $k$-step lag는 다음과 같다.

$$X_{t-k}$$

Lag는 원래 값을 보존하므로 해석이 쉽고, sequence를 tabular regression 또는 classification 문제로 변환하는 가장 직접적인 방법이다. scikit-learn의 time-series forecasting 예제도 여러 lag와 과거 rolling statistic을 생성하여 일반 regression model의 입력으로 사용한다 [[1](#ref-1)].

Difference는 현재값과 과거값의 차이를 사용한다.

$$\Delta X_t = X_t - X_{t-1}$$

Lag가 "직전 상태가 얼마였는가"를 표현한다면 difference는 "직전 상태에서 얼마나 이동했는가"를 표현한다. 불규칙한 sampling interval에서는 단순 difference보다 시간 간격으로 나눈 rate of change가 더 적합할 수 있다.

$$R_t = \frac{X_t - X_{t-1}}{T_t - T_{t-1}}$$

History-based feature의 주요 한계는 lag 수가 증가할수록 feature dimension과 collinearity가 빠르게 증가한다는 점이다. 따라서 모든 lag를 추가하기보다 domain cycle, validation result, ACF·PACF를 이용해 후보 범위를 제한하는 것이 적절하다.

### 3.3 Window-based Features

Window-based feature는 최근 $N$개 관측을 하나의 통계량으로 압축한다. 대표적인 rolling mean은 다음과 같다.

$$M_t = \frac{1}{N}\sum_{i=1}^{N}X_{t-i}$$

Rolling statistic은 목적에 따라 세 종류로 구분할 수 있다.

- **Level**: rolling mean, median, quantile
- **Variability**: rolling standard deviation, range, IQR
- **Trend**: rolling slope, rolling difference of means

**Level**은 "최근에 대체로 어느 정도 수준이었나?"를 나타낸다. 또한 동일한 현재값이라도 최근 분산이 큰 sequence와 안정적인 sequence는 다른 상태일 수 있다. 따라서 **Variability**의  rolling standard deviation은 current value가 제공하지 못하는 local stability 정보를 추가한다. **Trend**은 최근 값이 증가, 감소 또는 정체하는 방향성을 나타낸다. Rolling slope​는 최근 window의 관측값에 선형 추세를 적합하고, 그 기울기를 하나의 feature로 사용하여 최근 변화의 방향과 정도를 요약한다. 기울기가 양수이면 증가 추세, 음수이면 감소 추세, 0에 가까우면 뚜렷한 변화가 없는 상태를 의미한다.

**Expanding statistic**은 최근 일정 구간만 사용하는 rolling window와 달리, 현재 시점까지의 모든 과거 관측값을 사용한다. 따라서 장기적인 평균이나 누적 상태를 표현하는 데 유용하지만, system의 상태나 운전 조건이 중간에 변한 경우에는 오래된 데이터가 계속 포함되어 현재 상태를 충분히 반영하지 못할 수 있다.

### 3.4 Decay-based Features

Decay-based feature는 과거 관측값을 모두 동일하게 다루지 않고, 최근 관측에 더 큰 weight를 부여하고 오래된 관측의 영향은 점차 감소시키는 방식이다.

대표적인 방법이 EWMA (Exponentially Weighted Moving Average)이다.

$$
S_t = \alpha X_t + (1-\alpha)S_{t-1}
$$

여기서 $S_t$는 현재 시점의 EWMA, $X_t$는 현재 관측값이며, 0 < α ≤ 1이다.

α가 클수록 현재 값의 영향이 커지기 때문에 최근 변화에 빠르게 반응하고, α가 작을수록 과거 값의 영향이 오래 유지되어 더 부드러운 baseline을 만든다.

Rolling window가 최근 N개 관측만 사용하고 그 이전 값은 완전히 제외하는 것과 달리, EWMA는 오래된 값도 유지하되 시간이 지날수록 그 영향력을 점진적으로 감소시킨다.

EWMA를 기준값으로 사용하면 현재 값이 최근 상태에서 얼마나 벗어났는지를 나타내는 deviation feature도 만들 수 있다.

$$
D_t = X_t - S_{t-1}
$$
현재 값과 최근 EWMA의 차이가 크면 최근 상태와 다른 변화가 발생했음을 의미하고, 차이가 작으면 최근 상태와 유사한 수준임을 의미한다.

- D_t ≈ 0: 현재 값이 최근 수준과 비슷함
- D_t > 0: 현재 값이 최근 수준보다 높음
- D_t < 0: 현재 값이 최근 수준보다 낮음


### 3.5 Dependency-based Features

Dependency-based feature를 이해하기 위해서는 먼저 sequence 내부의 시간적 관계를 측정하는 대표적인 방법인 ACF, PACF, Cross-correlation​을 이해할 필요가 있고, 정의는 [Appendix A. Terminology](#appendix-a-terminology)를 확인한다.

Dependency-based 방법은 두 가지 방식으로 활용할 수 있다.

첫째, lag feature를 선택하기 위한 진단 도구로 사용할 수 있다. 여러 lag에 대해 ACF 또는 PACF를 계산한 뒤 관계가 강하게 나타나는 lag를 후보로 선택한다. 예를 들어 lag 1, lag 2, lag 5에서 높은 관계가 확인된다면 해당 lag의 과거 값을 feature 후보로 고려할 수 있다. 다만 correlation이 높다는 이유만으로 feature를 확정하는 것은 아니며, 최종 사용 여부는 validation을 통해 확인해야 한다.

둘째, sequence 자체의 dependency structure를 고정 길이 feature로 요약할 수 있다. 예를 들어 각 sample이 하나의 긴 sequence를 가지고 있다면 ACF lag 1, ACF lag 2와 같은 값을 계산하여 sequence의 자기상관 구조를 몇 개의 feature로 표현할 수 있다.

Cross-correlation도 유사하게 서로 다른 variable 사이에서 가장 강한 관계가 나타나는 lag를 탐색하거나, 해당 lag의 correlation 값을 sequence-level feature로 사용할 수 있다. 다만 correlation이 존재한다고 해서 한 variable이 다른 variable의 원인이라고 바로 판단할 수는 없다.

정리하자면 Dependency-based feature는 주로 다음과 같은 용도로 사용할 수 있다.

* 어떤 lag를 feature로 사용할지 결정
* 반복되는 주기나 패턴 확인
* 서로 다른 variable 사이의 시간차 관계 탐색


### 3.6 Frequency-based Features

Frequency-based feature는 sequence를 time domain이 아닌 frequency domain에서 요약한다. 즉 시계열을 시간 순서대로 보는 대신, 얼마나 빠르게 반복되는 패턴이 있는지로 바꿔서 본다. Periodogram은 time-series measurement의 power spectral density를 추정하여 어떤 frequency component가 강한지 표현한다 [[4](#ref-4)].

대표 feature는 다음과 같다.

- dominant frequency
- spectral power
- low-frequency power와 high-frequency power의 비율
- frequency band별 energy

이 계열은 반복 cycle이나 oscillation이 중요한 sequence에서 유용하다. 반대로 sequence가 짧거나 sampling interval이 불규칙하고 그 불규칙성이 보정되지 않았다면 frequency feature의 해석 가능성이 낮아진다. 해당 feature는 아직 더 공부가 필요하다.

### 3.7 State/Event-based Features

연속적인 값의 변화뿐 아니라, 상태가 바뀌었다는 사실 자체도 중요한 시간 정보가 될 수 있다. 특정 상태 변화나 event를 정의할 수 있다면 다음과 같은 feature를 만들 수 있다.

+ state_change_flag: 직전 상태와 달라졌는지 여부
+ count_since_event: event 발생 이후 몇 번의 관측이 지났는지
+ elapsed_time_since_event: event 발생 이후 얼마나 시간이 지났는지
+ current_state_duration: 현재 상태가 얼마나 오래 유지되고 있는지
+ regime_identifier: 현재 system이 어떤 상태 또는 운전 mode에 있는지 표시

예를 들어 system이 정상 상태 → 조건 변경 → 적응 중 → 안정 상태와 같이 여러 상태를 거친다면, 현재 값만으로는 이 차이를 알기 어려울 수 있다. 이때 현재 system이 어떤 상태에 있는지를 feature로 추가하면 상태 변화 직후의 일시적인 변화나 서로 다른 운전 상태의 차이를 모델에 전달할 수 있다.

다만 어떤 상태를 하나의 regime으로 볼지는 실제 system에 대한 기준이 필요하다. 명확한 기준 없이 임의의 threshold로 상태를 나누면 feature의 의미도 임의적으로 결정될 수 있다.

### 3.8 Automated Feature Extraction

Automated Feature Extraction은 사람이 lag, rolling, spectral feature를 하나씩 직접 설계하는 대신, 다양한 time-series feature를 자동으로 생성하여 후보 feature를 탐색하는 방법이다.

대표적인 도구로 tsfresh와 catch22가 있다.

+ tsfresh: 다양한 time-series characterization method를 이용하여 많은 수의 feature를 자동으로 생성하고, 이후 통계적 검정을 이용해 target과 관련성이 있는 feature를 선택하는 방식이다. 기본 설정에서는 수백 개의 feature가 생성될 수 있기 때문에, 넓은 후보 공간을 탐색하는 데 유용하지만 sample size가 작을 경우 feature 수가 과도하게 증가할 수 있다. [[5](#ref-5)]
+ catch22: 대규모 time-series feature 집합에서 서로 중복되는 feature를 줄이고, 대표성이 높은 22개의 feature만 선택한 방법이다. Distribution, autocorrelation, successive difference, fluctuation 등 서로 다른 sequence 특성을 비교적 적은 수의 feature로 요약할 수 있다. [[6](#ref-6)]

두 방법의 차이는 feature 탐색 범위에 있다. tsfresh는 많은 feature를 생성한 뒤 필요한 feature를 선택하는 방식이고, catch22는 미리 선정된 소수의 대표 feature만 계산하는 방식이다.

Automated Feature Extraction은 manual feature engineering을 완전히 대체하기보다는, 사람이 미처 고려하지 못한 temporal pattern을 찾기 위한 candidate discovery tool로 사용하는 것이 적절하다. 특히 생성된 feature가 많아질 경우 sample size, feature redundancy, feature selection을 함께 고려해야 한다.

## 4. Application
Temporal feature들을 실제로 쓸 때 어떤 조건과 주의사항이 있는지는 다음과 같다.

### 4.1 Prediction Timestamp

Temporal feature 설계에서는 먼저 model이 어느 시점에 예측을 수행하는지와 그 시점까지 실제로 관측 가능한 정보가 무엇인지를 명확히 정의해야 한다.

Prediction timestamp 이전에 이미 관측된 sensor value나 historical information은 feature로 사용할 수 있다. 반대로 예측 이후에 생성되거나 확인되는 값은 dataset에 존재하더라도 model input으로 사용하면 안 된다.

Rolling, expanding, EWMA와 같은 temporal statistic도 동일한 기준을 따른다. Feature를 계산할 때 prediction timestamp 이후의 관측값이나 전체 dataset의 정보를 포함하면 미래 정보가 model input에 유입되어 temporal leakage가 발생할 수 있다.

### 4.2 Validation

Temporal dependency가 있는 데이터는 **학습과 검증에서도 시간 순서를 유지해야 한다.** 미래 시점의 데이터가 training에 포함되면 실제 예측 환경보다 쉬운 조건에서 평가하게 되어 성능이 과대평가될 수 있다. 따라서 random split보다는 시간 순서를 유지하는 chronological split이나 `TimeSeriesSplit`을 사용하는 것이 적절하다. [[1](#ref-1)]

Validation에서는 다음을 확인해야 한다.

* training data가 validation data보다 시간상 앞에 있는가
* feature를 만들 때 validation 이후의 정보를 사용하지 않았는가
* scaler, imputer, feature selector가 training data만 이용해 학습되었는가
* 같은 entity의 매우 가까운 sequence가 train과 validation에 동시에 포함되어 정보가 과도하게 공유되지 않았는가


### 4.3 Window and Lag Length

Lag와 window length는 얼마나 과거의 정보를 사용할 것인지를 결정하는 값이다. 너무 짧게 설정하면 최근 변화는 빠르게 반영할 수 있지만 일시적인 noise에 민감할 수 있고, 너무 길게 설정하면 값은 안정적으로 요약되지만 최근 상태의 변화를 늦게 반영할 수 있다.

따라서 window와 lag는 데이터의 특성과 실제 system의 변화 속도를 고려하여 설정해야 한다. 특히 일정한 반복 주기나 계절성, 공정 cycle이 존재한다면 해당 주기를 기준으로 window 또는 lag 후보를 정할 수 있다. 예를 들어 약 20 step마다 비슷한 패턴이 반복된다면 lag 20이나 20 step 전후의 window를 후보로 고려할 수 있다.

하나의 window만으로 단기 변화와 장기 상태를 모두 표현하기 어려운 경우에는 서로 다른 길이의 소수 window를 함께 사용할 수 있다. 예를 들어 짧은 window는 최근 변동을, 긴 window는 장기적인 수준을 나타내도록 구성할 수 있다.

다만 lag와 window의 종류를 많이 늘리면 feature 수도 함께 증가하므로, 필요한 후보만 선택하고 validation을 통해 실제로 도움이 되는지 확인하는 것이 필요하다.


### 4.4 Failure Conditions

Temporal feature를 추가하는 것이 항상 개선을 의미하지는 않는다. 다음 조건에서는 추가 이득이 작거나 오히려 일반화 성능이 저하될 수 있다.

- row order가 실제 sequence order를 나타내지 않는 경우
- sequence 간 시간 간격의 의미가 크게 다른데 동일 lag로 처리한 경우
- history가 거의 없는 짧은 sequence
- feature count가 sample size에 비해 과도하게 증가한 경우
- current state만으로 target이 충분히 설명되는 경우
- regime change가 잦아 long historical window가 현재 상태를 왜곡하는 경우

## 5. Comparison

기술 선택은 "어떤 방법이 가장 고급인가"가 아니라 "어떤 temporal information이 필요한가"를 기준으로 해야 한다. Raw history가 중요하면 lag, 변화량이 중요하면 difference, recent baseline이 중요하면 rolling 또는 EWMA, 반복 주기가 중요하면 frequency feature가 우선이다.

## References

<a id="ref-1"></a>
[1] scikit-learn developers. [Lagged features for time series forecasting](https://scikit-learn.org/stable/auto_examples/applications/plot_time_series_lagged_features.html). *scikit-learn Documentation*.<br>
<a id="ref-2"></a>
[2] NIST/SEMATECH. [EWMA Control Charts](https://www.itl.nist.gov/div898/handbook/pmc/section3/pmc324.htm). *e-Handbook of Statistical Methods*.<br>
<a id="ref-3"></a>
[3] statsmodels developers. [Time Series analysis tsa](https://www.statsmodels.org/stable/tsa.html). *statsmodels Documentation*.<br>
<a id="ref-4"></a>
[4] SciPy developers. [Signal processing — Spectral analysis](https://docs.scipy.org/doc/scipy/reference/signal.html). *SciPy Documentation*.<br>
<a id="ref-5"></a>
[5] Christ, M., Braun, N., Neuffer, J., & Kempa-Liehr, A. W. (2018). [Time Series FeatuRe Extraction on basis of Scalable Hypothesis tests (tsfresh – A Python package)](https://doi.org/10.1016/j.neucom.2018.03.067). *Neurocomputing*, 307, 72–77.<br>
<a id="ref-6"></a>
[6] Lubba, C. H., Sethi, S. S., Knaute, P., Schultz, S. R., Fulcher, B. D., & Jones, N. S. (2019). [catch22: CAnonical Time-series CHaracteristics](https://doi.org/10.1007/s10618-019-00647-x). *Data Mining and Knowledge Discovery*, 33, 1821–1852.<br>
<a id="ref-7"></a>
[7] Bifet, A., & Gavaldà, R. (2007). [Learning from Time-Changing Data with Adaptive Windowing](https://doi.org/10.1137/1.9781611972771.42). *Proceedings of the 2007 SIAM International Conference on Data Mining*, 443–448.

---

## Appendix A. Terminology

- **Sampling structure**: 데이터를 어떤 간격으로 측정했는지에 대한 구조. 측정 간격이 일정하거나 최소한 그 간격을 해석할 수 있어야 의미가 존재.
- **Feature budget**: 만들 수 있는 feature 수를 얼마나 허용할 수 있는지에 대한 범위
- **Lag**: 현재 시점을 기준으로 일정 step 이전의 관측값.
- **ACF (Autocorrelation Function)**:하나의 sequence에서 현재 값과 k step 이전 값 사이의 correlation을 lag별로 나타내는 함수.  
예를 들어 `lag 1`의 ACF가 높다면 현재 값이 바로 이전 값과 강한 관계를 가진다는 뜻이다.

  ```text
  10 → 11 → 12 → 13 → 14
  ```

  위와 같이 값이 비슷한 흐름을 계속 유지한다면 현재 값과 직전 값의 correlation이 높게 나타날 수 있다.

- **PACF (Partial Autocorrelation Function)**:중간 lag의 영향을 제거한 뒤, 현재 값과 특정 lag의 과거 값 사이의 직접적인 correlation을 나타내는 함수.  
  특정 lag가 현재 값과 얼마나 직접적으로 관련되어 있는지를 확인한다. 예를 들어 `lag 2`를 볼 때 `lag 1`의 영향을 제외하고, 2개 전 값 자체가 현재 값과 얼마나 관련 있는지를 본다.

- **Cross-correlation**:
  서로 다른 두 variable 사이의 시간차 관계를 확인.  
  예를 들어 한 variable이 먼저 변한 뒤 일정 시간이 지나 다른 variable이 따라 변하는 패턴을 탐색할 수 있다.

  ```text
  Variable A : 10 → 12 → 14 → 15
                     ↓
  Variable B :  5 →  5 →  7 →  9
  ```
- **Concept drift**: 시간이 지남에 따라 input 또는 target을 포함한 data distribution의 관계가 변하는 현상.
- **spectral featur**: 시계열을 "시간에 따른 값 변화"로 보지 않고, "어떤 주파수 성분이 얼마나 강한가"로 바꿔서 만든 feature
- **Difference**: 현재 관측값과 이전 관측값의 차이로 정의한 temporal feature.
- **EWMA**: Exponentially Weighted Moving Average. 최근 관측에 더 큰 weight를 주는 recursive moving average.
- **Periodogram**: Periodogram은 sequence 안에 존재하는 여러 frequency가 얼마나 강한지 보여주는 방법.
- **Regime**: 동일한 statistical 또는 operational property가 유지되는 구간 또는 상태.
- **Rolling window**: 최근 고정 개수 또는 고정 시간 범위의 관측값으로 statistic을 계산하는 방식.
- **Temporal Feature Engineering**: ordered data의 history, variation, trend, dependency를 fixed-length feature로 변환하는 과정.
- **Temporal leakage**: prediction timestamp에서 사용할 수 없는 미래 또는 사후 정보를 feature 생성이나 model fitting에 사용하는 문제.

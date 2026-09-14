# Trace Channel Selection

Rev. 3 | Created: 2026-09-13 | Updated: 2026-09-13 13:13 UTC

## 1. Purpose

- **Problem Statement**: 하나의 wafer에서 다수의 trace channel이 기록될 때 서로 비슷한 역할이나 정보를 가지는 channel이 함께 존재할 수 있으므로, summary feature 생성에 사용할 대표 channel을 체계적으로 선택할 기준이 필요하다.
- **Goal**: channel에 대한 의미 정보의 유무에 따라 grouping 방법을 구분하고, 각 group에서 대표 channel을 선택한 뒤 그 선택의 대표성과 안정성을 검증하는 방법을 정리한다.
- **Non-Goal**: target 예측 성능을 직접 최적화하는 supervised feature selection과, 선택된 channel의 어느 구간에서 어떤 통계값으로 최종 summary feature를 생성할지는 본문 범위에서 제외한다.

## 2. Summary

Trace Channel Selection은 **Channel Grouping, Representative Channel Selection, Selection Validation**의 세 단계로 정리할 수 있다. Channel의 의미와 장비 정보가 충분하면 의미 기반 grouping을 우선할 수 있고, 의미 정보가 제한적이거나 channel 수가 많으면 실제 signal의 관계를 이용한 data-driven grouping을 사용할 수 있다.

Table 1. Trace Channel Selection taxonomy

| Stage | Method | Main Idea | Output |
| --- | --- | --- | --- |
| Channel Grouping | Knowledge-based Grouping | 물리적 의미, 장비 위치, 공정 역할을 기준으로 channel 분류 | 의미가 유사한 channel group |
| Channel Grouping | Data-driven Grouping | `correlation`, signal shape, distance 등을 기준으로 channel 분류 | signal이 유사한 channel group |
| Representative Channel Selection | Group Representative | group 내부를 가장 잘 대표하는 channel 선택 | group별 대표 channel |
| Representative Channel Selection | Redundancy Reduction | 선택된 대표 channel 사이의 중복 확인 | 중복이 줄어든 channel set |
| Selection Validation | Stability / Information Preservation | lot, 기간, data subset이 달라져도 대표성이 유지되는지 확인 | 검증된 representative channel set |

이 분류에서 중요한 점은 **channel의 의미를 항상 안다고 가정하지도 않고, 항상 모른다고 가정하지도 않는 것**이다. 사용 가능한 정보가 있으면 활용하고, 부족한 부분은 data 자체의 관계로 보완한다. 또한 대표 channel을 선택하는 단계와 최종 summary statistic을 만드는 단계는 서로 다른 기술문제로 구분한다.

## 3. Principle

### 3.1 Channel Grouping

Channel Grouping의 목적은 여러 trace channel을 개별적으로 비교하기 전에 서로 비슷한 역할이나 정보를 가지는 channel을 묶어 selection 단위를 줄이는 것이다. Grouping은 channel의 의미 정보를 사용할 수 있는지에 따라 Knowledge-based Grouping과 Data-driven Grouping으로 나눌 수 있다.

### 3.2 Knowledge-based Grouping

Knowledge-based Grouping은 channel의 이름, 물리적 의미, 장비 위치, 공정 역할과 같은 metadata를 이용하여 channel을 분류하는 방법이다. 의미 정보가 충분한 경우에는 data pattern만으로 grouping하는 것보다 group의 해석이 쉽고, 서로 다른 물리량이 우연히 비슷하게 움직여 같은 group으로 묶이는 문제를 줄일 수 있다.

예를 들어 channel을 Voltage, Pressure, Temperature, Gas Flow, Position, Counter / Lifetime과 같은 역할로 나눌 수 있다. 같은 의미의 channel이 여러 개 존재한다면 먼저 의미 기준으로 범위를 좁힌 뒤, group 내부에서 실제 signal 관계를 확인하여 대표 channel을 선택할 수 있다.

다만 같은 물리량을 측정하는 channel이라도 측정 위치, 장비 module, 공정 단계가 다르면 서로 다른 정보를 가질 수 있다. 따라서 이름이나 단위가 같다는 이유만으로 같은 정보를 가진다고 단정하지 않고, 실제 trace의 변화도 함께 확인해야 한다.

### 3.3 Data-driven Grouping

Data-driven Grouping은 channel 의미에 의존하지 않고 실제 trace가 어떻게 움직이는지를 기준으로 channel을 묶는 방법이다. 의미 정보가 제한적인 경우뿐 아니라, channel 수가 많아 사람이 모든 관계를 직접 확인하기 어려운 경우에도 사용할 수 있다.

#### Correlation-based Grouping

`correlation` 기반 방법은 여러 channel이 함께 증가하거나 감소하는 정도를 이용한다. Pearson correlation은 두 dataset의 선형 관계를 측정하고, Spearman correlation은 두 dataset의 단조 관계를 순위 기반으로 측정한다 [[1](#ref-1)] [[2](#ref-2)].

높은 `correlation`은 두 channel이 비슷한 정보를 포함할 가능성을 보여주지만, 동일한 역할을 의미하지는 않는다. 서로 다른 물리량도 같은 공정 조건의 영향을 받아 함께 움직일 수 있기 때문이다. 따라서 `correlation`은 grouping 후보를 만드는 기준으로 사용하고, 가능한 경우 metadata나 signal shape를 함께 확인하는 것이 적절하다.

#### Signal Similarity-based Grouping

Signal Similarity-based Grouping은 단순한 값의 크기보다 trace 전체의 모양을 비교한다. 상승, 유지, 하강, 반복 진동과 같은 시간적 패턴이 유사한 channel을 같은 group 후보로 볼 수 있다.

같은 시간축에서 길이와 위치가 맞는 trace에는 point-wise distance를 사용할 수 있다. 변화 시점이 조금씩 어긋나는 trace는 Dynamic Time Warping과 같이 time-series alignment를 허용하는 방법을 사용할 수 있다 [[3](#ref-3)].

이 방법은 channel의 전체적인 동작 형태를 비교할 수 있다는 장점이 있지만, sampling interval이나 trace 길이가 크게 다르면 distance 자체가 왜곡될 수 있으므로 비교 조건을 먼저 맞춰야 한다.

#### Clustering-based Grouping

Clustering-based Grouping은 channel 사이의 `correlation`, distance 또는 signal representation을 계산한 뒤 비슷한 channel을 자동으로 묶는 방법이다. Hierarchical Clustering은 distance를 기준으로 cluster를 단계적으로 결합하는 방식으로 사용할 수 있고 [[4](#ref-4)], K-means는 각 channel을 고정 길이의 representation으로 변환할 수 있을 때 적용할 수 있다 [[5](#ref-5)].

Clustering의 목적은 algorithm 자체로 정답 group을 만드는 것이 아니라, 사람이 개별적으로 확인하기 어려운 channel 관계를 구조화하여 representative selection의 후보를 만드는 것이다.

### 3.4 Representative Channel Selection

Representative Channel Selection은 각 group에서 이후 summary feature 생성에 사용할 channel을 하나 또는 소수로 줄이는 단계이다. 단순히 값이 큰 channel이나 변화가 많은 channel을 선택하는 것이 아니라, group 내부의 정보를 얼마나 잘 대표하는지를 기준으로 판단한다.

#### Group Representative

가장 단순한 기준은 group 내부의 다른 channel과 평균적으로 가장 유사한 channel을 선택하는 것이다. 예를 들어 group 내부 pairwise correlation의 평균이 가장 높은 channel은 다른 channel과 공통적으로 나타나는 변화를 잘 반영하는 대표 후보가 될 수 있다.

Table 2. Example of representative channel selection

| Channel | Group Similarity | Interpretation |
| --- | --- | --- |
| Channel A | 높음 | group과 전반적으로 유사 |
| Channel B | 가장 높음 | representative channel 후보 |
| Channel C | 보통 | 일부 channel과 차이 존재 |
| Channel D | 높음 | group과 전반적으로 유사 |

이 예에서는 Channel B가 group 내부의 다른 channel과 평균적인 관계가 가장 높으므로 representative channel 후보가 된다. 실제 selection에서는 하나의 기준만으로 결정하기보다 여러 wafer와 기간에서 같은 관계가 유지되는지를 함께 확인해야 한다.

#### Redundancy Reduction

서로 다른 group에서 대표 channel을 하나씩 선택했더라도 representative channel 사이에 다시 높은 중복이 나타날 수 있다. 이는 grouping 기준이나 threshold 때문에 사실상 같은 정보를 가지는 channel이 별도 group으로 나뉜 경우에 발생할 수 있다.

따라서 representative channel set을 만든 뒤 channel 사이의 `correlation`이나 signal similarity를 다시 확인하고, 거의 동일한 정보를 가지는 channel이 남아 있다면 group 구조와 selection 결과를 재검토한다.

#### Information Coverage

대표 channel의 수를 줄이는 목적은 단순한 차원 축소가 아니라 원래 channel들이 가진 주요 정보를 가능한 한 유지하면서 중복을 줄이는 것이다. 따라서 representative channel은 group 내부의 공통 pattern을 설명하고, 여러 wafer에서도 비슷한 관계를 유지해야 한다.

Information Coverage는 group 내부 similarity, representative channel과 나머지 channel의 평균 similarity, channel reduction 전후의 clustering 구조와 같은 기준으로 확인할 수 있다. 고정 길이의 matrix representation을 만들 수 있는 경우에는 reconstruction error와 같이 원래 data를 얼마나 설명하는지도 추가 기준으로 사용할 수 있다.

### 3.5 Selection Validation

Selection Validation은 선택된 channel이 특정 lot이나 특정 기간에서만 우연히 대표로 보이는지 확인하는 단계이다. 대표 channel은 한 번의 grouping 결과보다 data subset이 달라졌을 때도 비슷한 결과가 유지되는지가 중요하다.

Lot을 나누어 같은 group과 representative channel이 반복적으로 나타나는지 확인하면 lot stability를 평가할 수 있다. 시간 구간을 나누어 selection 결과를 비교하면 공정 상태 변화에 대한 time stability를 확인할 수 있다. 또한 일부 wafer를 제외하거나 다시 sampling했을 때도 비슷한 channel이 선택되는지 확인하면 selection 결과의 민감도를 평가할 수 있다.

최종적으로 representative channel은 **group 대표성, 중복 감소, data subset에 대한 안정성**을 함께 만족하는 channel로 판단하는 것이 적절하다.

## 4. Application

Trace Channel Selection은 사용 가능한 channel 정보의 수준에 따라 적용 방법을 달리하는 것이 적절하다. 의미 정보가 충분하면 Knowledge-based Grouping으로 큰 범위를 먼저 나누고, group 내부에서 data-driven similarity를 확인하는 방식이 해석과 효율 측면에서 유리하다.

의미 정보가 제한적인 경우에는 Data-driven Grouping을 먼저 사용하여 channel 관계를 구조화할 수 있다. 이때 group에 물리적 의미를 억지로 부여하기보다, 먼저 signal이 실제로 유사한지와 여러 wafer에서 관계가 반복되는지를 확인한다.

의미 정보가 일부 channel에만 존재하는 경우에는 알려진 정보에는 Knowledge-based Grouping을 적용하고, 의미 정보만으로 판단하기 어려운 관계에는 Data-driven Grouping을 적용할 수 있다. 두 방법은 서로 배타적인 선택지가 아니라 사용 가능한 정보를 어디까지 활용할 것인지의 차이로 보는 것이 적절하다.

Channel selection을 적용할 때는 다음 조건을 함께 고려해야 한다.

- trace 간 sampling interval과 길이의 비교 가능성
- 공정 단계 차이로 인한 signal pattern 변화
- 같은 물리량이라도 측정 위치에 따른 역할 차이
- lot 또는 기간 변화에 따른 channel 관계 변화
- 높은 `correlation`이 동일한 물리적 의미를 보장하지 않는 점


## 5. Further Work

Representative channel이 선택된 이후의 다음 기술문제는 **각 channel의 어느 trace region에서 어떤 summary statistic을 사용할 것인지 결정하는 것**이다. Channel Selection이 “무엇을 볼 것인가”를 정한다면, Further Work는 “선택된 channel을 wafer 단위 값으로 어떻게 표현할 것인가”를 정하는 단계이다.

### 5.1 Region Selection

- **What**: Entire Trace, 특정 Process Step, relative window, event 전후, stable region과 같이 summary를 계산할 trace 범위를 정의한다.
- **Why Now**: representative channel이 정해진 뒤에는 channel마다 의미 있는 구간이 다를 수 있으므로, 전체 trace를 동일하게 요약하면 공정과 무관한 구간이 함께 포함될 수 있다.
- **Required**: step 정보, timestamp, event 정보 또는 signal 변화에 기반한 region 정의 기준.

### 5.2 Summary Statistic Selection

- **What**: 선택된 region에서 first, last, mean, median, standard deviation, range, quantile, slope, area under the curve와 같은 summary statistic 중 어떤 값을 사용할지 결정한다.
- **Why Now**: 같은 channel이라도 대표 수준, 변동성, 변화량, 누적량 중 무엇을 표현하느냐에 따라 서로 다른 summary feature가 생성되기 때문이다.
- **Required**: channel의 signal 특성, region 정의 결과, summary feature의 안정성과 재현성을 비교할 평가 기준.

Table 4. Region and summary statistic examples

| Signal Characteristic | Region Example | Statistic Example | Representation |
| --- | --- | --- | --- |
| 일정한 공정 수준 | stable process region | mean, median | 대표 수준 |
| 시작 또는 종료 상태 | process start / end | first, last | 특정 시점 상태 |
| 공정 중 변동 | active process region | standard deviation, interquartile range, range | 변동성 |
| 최대·최소 상태 | active process region | min, max, quantile | 극값과 분포 위치 |
| 증가·감소 경향 | process region | last - first, slope | 변화량 |
| 누적된 signal | process region | area under the curve | 누적량 |

후속 단계에서는 `Channel × Region × Summary Statistic`의 조합을 summary feature candidate로 정의하고, 각 candidate가 wafer 간에서 안정적으로 계산되는지와 동일 조건에서 재현되는지를 평가하는 방향으로 확장할 수 있다.

## References

<a id="ref-1"></a>[1] SciPy Developers. [pearsonr — SciPy Manual](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pearsonr.html). SciPy documentation.<br>
<a id="ref-2"></a>[2] SciPy Developers. [spearmanr — SciPy Manual](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.spearmanr.html). SciPy documentation.<br>
<a id="ref-3"></a>[3] tslearn Developers. [Dynamic Time Warping](https://tslearn.readthedocs.io/en/latest/gen_modules/metrics/tslearn.metrics.dtw_path.html). tslearn documentation.<br>
<a id="ref-4"></a>[4] SciPy Developers. [linkage — SciPy Manual](https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html). SciPy documentation.<br>
<a id="ref-5"></a>[5] scikit-learn Developers. [Clustering](https://scikit-learn.org/stable/modules/clustering.html). scikit-learn documentation.

---

## Appendix A. Terminology

- **Channel**: 장비 trace에서 하나의 sensor, setting, counter 또는 상태값을 시간 순서로 기록하는 항목.
- **Channel Grouping**: 의미 또는 signal pattern이 비슷한 channel을 하나의 group으로 묶는 과정.
- **Correlation**: 두 channel의 값이 함께 변하는 정도를 나타내는 관계.
- **Data-driven Grouping**: channel의 실제 signal 관계와 pattern을 이용하여 channel을 grouping하는 방법.
- **Dynamic Time Warping**: 두 시계열의 시간 위치 차이를 허용하면서 shape similarity를 비교하는 방법.
- **Information Coverage**: 선택된 representative channel이 원래 group의 주요 signal 정보를 얼마나 유지하는지를 나타내는 개념.
- **Knowledge-based Grouping**: channel의 물리적 의미, 장비 정보, 공정 역할과 같은 사전 정보를 이용하여 channel을 grouping하는 방법.
- **Region**: trace에서 summary statistic을 계산할 시간 또는 공정 구간.
- **Representative Channel**: 하나의 channel group을 대표하여 이후 summary feature 생성 대상으로 사용하는 channel.
- **Signal Similarity**: channel 사이의 trace shape 또는 변화 pattern이 얼마나 유사한지를 나타내는 관계.
- **Summary Feature**: trace의 특정 region을 statistic으로 요약하여 만든 wafer 단위 feature.
- **Summary Statistic**: trace region을 wafer 단위 값으로 변환하기 위해 계산하는 대표값, 변동값, 변화량 또는 누적값.
- **Trace**: 공정 중 장비 signal을 시간 순서대로 기록한 시계열 data.

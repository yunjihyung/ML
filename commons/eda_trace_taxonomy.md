# Trace EDA Taxonomy

Rev. 0 | Created: 2026-09-27 | Updated: 2026-09-27 17:52 KST

## 1. Purpose

- **Problem Statement**: 반도체 설비는 wafer마다 여러 sensor의 trace를 기록하며, 이 trace에는 logging 오류, 공정 조건 차이, 정보가 없는 channel, 설비 상태 변화가 함께 섞여 있다. 어떤 EDA (Exploratory Data Analysis) 를 어떤 순서로 할지 정해져 있지 않으면 데이터마다 확인 범위가 달라지고 빠뜨리는 항목이 생긴다.
- **Goal**: trace 데이터의 EDA에 필요한 영역과 각 영역이 필요한 이유, 수행 순서를 하나의 taxonomy로 정리한다.
- **Non-Goal**: 각 영역의 세부 방법과 판정 기준은 영역별 문서에서 다루며, 이 문서는 큰 틀만 정리한다.

## 2. Summary

**Trace EDA는 7개 영역으로 나뉜다.** 앞의 두 영역은 기록과 공정 구조를 믿을 수 있는지 확인하고, 다음 두 영역은 분석에 사용할 channel과 구간을 고르며, 나머지 영역은 결과를 흔드는 요인과 target과의 관계를 확인한다. 이 문서에 쓰인 용어는 [Appendix A](#appendix-a-terminology) 에 정리했다.

Table 1. Trace EDA areas and writing status

| Order | Area | ID | Question | Document | Module |
| --- | --- | --- | --- | --- | --- |
| 1 | Data Quality | `data_quality` | 기록 자체의 신뢰 가능 여부 | 작성 예정 | 작성 예정 |
| 2 | Step and Recipe Structure | `step_structure` | wafer 간 공정의 동일 여부 | 작성 예정 | 작성 예정 |
| 3 | Channel Screening | `channel_screening` | 실제 정보를 담은 channel | 작성 중 | 작성 중 |
| 4 | Transient and Steady State | `transient` | 통계값을 계산할 구간 | 작성 예정 | 작성 예정 |
| 병행 | Wafer-to-Wafer Variation and Drift | `drift` | 설비 상태에 따른 값의 변화 | 작성 예정 | 작성 예정 |
| 병행 | Outlier Wafer | `outlier_wafer` | 기준에서 벗어난 wafer | 작성 예정 | 작성 예정 |
| 마지막 | Target Relation | - | 결과와 관련된 값 | 범위 밖 | 범위 밖 |

각 영역의 세부 내용은 `eda_trace_<AREA_ID>.md` 에, 계산은 같은 이름의 module에 둔다. 여러 영역이 함께 쓰는 data 읽기와 step 정렬은 `eda_trace_core` module이 담당하며, 현재 작성 중이다. 상태는 작성 예정, 작성 중, 작성 완료의 세 가지로 표시한다.

```text
 Trust the record          Choose what to use              Relate to result
 ────────────────          ──────────────────              ────────────────
 1 Data Quality       ─►   3 Channel Screening       ─►    Target Relation
 2 Step and Recipe         4 Transient and
   Structure                 Steady State

        Wafer-to-Wafer Variation and Drift, Outlier Wafer  (alongside 1-4)
```

*Fig 1. Order of the trace EDA areas.*

## 3. Principle

**앞 영역에 문제가 있으면 뒤 영역의 결과를 신뢰할 수 없다.** 예를 들어 기록 간격이 일정하지 않은 데이터에서 step 길이를 비교하면 공정의 차이가 아니라 logging의 차이를 보게 된다. 모든 영역에는 다음 원칙을 공통으로 적용한다.

- 앞 영역의 산출물을 뒤 영역의 전제로 사용
- 정규화 전 원본값을 기준으로 판정하고, 정규화 그림은 모양 비교에만 사용
- 판정 유형과 근거 수치를 한 쌍으로 기록
- module은 계산 값까지만 제공하고, 판정은 module을 호출하는 script가 수행
- 제외한 channel과 wafer의 원본 보존
- 적용 사례의 판정 기준값은 참고값으로만 사용하고, 데이터마다 다시 설정

## 4. Area

### 4.1 Data Quality

Data Quality 영역은 sensor 값을 해석하기 전에 기록 자체를 믿을 수 있는지 확인한다. Logging 오류는 공정 변화와 구분되지 않으므로, 이 영역을 건너뛰면 기록 누락을 공정의 차이로 해석할 수 있다. 예를 들어 기록이 빠진 wafer는 step이 짧아 보이고, 통신이 끊긴 구간은 값이 안정된 것처럼 보인다.

- 확인 항목
  - 기록 간격의 일정성
  - 중복 기록과 누락 기록
  - 결측값의 위치와 분포
  - 공정 중간에 고정된 값
  - 측정 상한과 하한에 붙은 값
  - 값이 기록되는 resolution
  - 특정 시점 이후 값의 scale 변화
- 산출물: 분석에 사용할 wafer와 channel 목록, 제외 사유

### 4.2 Step and Recipe Structure

Step and Recipe Structure 영역은 비교할 wafer들이 같은 공정을 거쳤는지 확인한다. 서로 다른 공정을 섞어 비교하면, 값의 차이가 공정 결과에서 온 것인지 공정 조건의 차이에서 온 것인지 구분할 수 없다.

- 확인 항목
  - wafer 간 step 순서의 일치 여부
  - 누락되거나 반복된 step
  - step 길이의 분포와 극단값
  - recipe 종류와 version의 혼재
  - step 번호를 기록한 channel의 신뢰성
- 산출물: 비교 가능한 wafer group, 분석에 사용할 step 정의

### 4.3 Channel Screening

Channel Screening 영역은 각 channel이 실제 정보를 담고 있는지 확인한다. 기록되는 channel 가운데 상당수는 값이 변하지 않거나 다른 channel과 같은 값을 기록하므로, 걸러내지 않으면 channel별 역할과 정보량을 판단하기 어렵다. 중복 여부는 상관계수만으로 판정하지 않고, 값이 정확히 같은 비율과 최대 차이를 함께 확인한다.

- 판정 유형
  - Constant: 값이 변하지 않는 channel
  - Sparse pulse: 대부분 한 값이고, 변화가 한두 기록으로 끝나는 channel
  - Redundant: 다른 channel과 사실상 같은 값을 기록하는 channel
  - Derived: 다른 channel들의 계산으로 얻어지는 channel
  - Retained: 위 유형에 해당하지 않는 channel
- 산출물: channel별 처리표 (제외, 축약, 유지와 근거 수치)

### 4.4 Transient and Steady State

Transient and Steady State 영역은 step 안에서 통계값을 계산할 구간을 정한다. Step이 바뀐 직후에는 값이 목표로 이동하는 중이므로, 이 구간이 섞이면 step의 통계값이 흔들린다.

- 확인 항목
  - 값이 안정될 때까지 걸리는 시간 (settling time)
  - 목표값을 넘었다가 되돌아오는 overshoot
  - 값이 증가하거나 감소하는 속도 (ramp rate)
  - setpoint channel이 있는 경우, setpoint와 실제값의 차이
- 산출물: channel과 step별 통계 구간 정의

### 4.5 Wafer-to-Wafer Variation and Drift

Wafer-to-Wafer Variation and Drift 영역은 같은 recipe에서도 설비 상태에 따라 값이 달라지는지 확인한다. 설비 상태에 따른 변동을 모른 채 model을 만들면, model이 공정의 영향이 아니라 설비 상태를 학습할 수 있다.

- 확인 항목
  - 공정 순서에 따른 값의 추세 (drift)
  - PM (Preventive Maintenance) 전후의 차이
  - lot의 첫 wafer와 긴 idle 뒤 wafer의 차이
  - chamber와 slot별 차이
- 산출물: 변동 요인 목록, 보정이나 group 분리의 필요 여부

### 4.6 Outlier Wafer

Outlier Wafer 영역은 대부분의 wafer와 다르게 기록된 wafer를 찾는다. 소수의 outlier wafer가 평균과 상관계수를 왜곡할 수 있으므로, 찾은 뒤에는 원인을 확인하고 처리 방침을 정한다.

- 확인 항목
  - median trace에서 벗어난 정도
  - 정상 범위를 벗어난 구간의 위치와 길이
  - 앞 영역에서 확인한 원인과의 대응 여부
- 산출물: outlier wafer 목록, 처리 방침 (제외 또는 별도 분석)

### 4.7 Target Relation

Target Relation 영역은 앞 영역을 거친 channel과 구간의 통계값이 target과 어떤 관계를 가지는지 확인한다. 이 영역은 feature 선택과 modeling에 가까우므로 trace EDA의 범위 밖에 두고, 이 문서에서는 앞 영역과 연결되는 조건만 정리한다.

- 확인 항목
  - 앞 영역의 산출물로 만든 feature의 사용
  - drift나 group 차이로 생긴 관계인지 여부
  - wafer 수 대비 feature 수
- 산출물: 후보 feature 목록

---

## Appendix A. Terminology

- **Chamber**: 설비 안에서 wafer가 공정되는 공간
- **Channel**: trace 안의 sensor column 하나
- **Drift**: 공정 순서에 따라 값이 서서히 이동하는 현상
- **Idle**: 설비가 wafer를 공정하지 않고 대기하는 시간
- **Lot**: 함께 공정되는 wafer의 묶음
- **Median trace**: 여러 wafer의 trace를 step 기준으로 정렬한 뒤, 시점별 중앙값으로 만든 trace
- **Outlier wafer**: 대부분의 wafer와 다른 trace를 가진 wafer
- **Overshoot**: 값이 목표값을 넘었다가 되돌아오는 현상
- **Ramp rate**: 값이 목표로 이동하는 속도
- **Recipe**: 설비가 wafer를 공정하는 조건과 step 구성
- **Setpoint**: 설비에 지정된 목표값
- **Settling time**: step이 바뀐 뒤 값이 안정될 때까지 걸리는 시간
- **Slot**: wafer를 담는 용기 안에서 wafer가 놓이는 위치
- **Steady state**: step 안에서 값이 목표 근처에 안정된 구간
- **Step**: recipe가 나눈 공정 구간
- **Target**: 공정 후 계측한 결과값
- **Trace**: wafer 한 장을 공정하는 동안 시간 순서로 기록된 sensor 값
- **Transient**: step이 바뀐 직후 값이 목표로 이동하는 구간

# Technical Documentation

이 폴더는 프로젝트 진행 중 조사하거나 정리한 **기술적인 자료와 문서**를 작성하고 보관하기 위한 공간이다.

주요 내용은 다음과 같다.

* 기술 개념 및 방법론 정리
* 데이터 분석 및 모델링 관련 기술 조사
* 구현 방법 및 사용 방법 정리
* 프로젝트에서 활용할 수 있는 기술적 참고 자료
* 장기적으로 재사용할 수 있는 기술 문서

각 문서는 특정 문제를 해결하기 위한 코드보다는 **기술의 개념, 원리, 적용 조건 및 활용 방법을 이해하고 참고할 수 있도록 정리하는 것**을 목적으로 한다.

## 문서 목록

* [class_design.md](class_design.md) — AI가 생성한 Python code에서 흩어지는 function과 state를 정리하기 위한 class 설계 기준(Constructor Injection, Single Responsibility 등)과 function/class 구분 기준.
* [long_data_distribution_shift.md](long_data_distribution_shift.md) — Long data에서 Train과 Test의 feature distribution이 달라질 때 발생하는 generalization 저하를 Temporal Shift와 Support / Range Shift로 분류하고, 각각의 diagnosis·validation·mitigation 방법을 정리.
* [temporal_feature_engineering.md](temporal_feature_engineering.md) — 순차 데이터를 고정 길이 feature로 변환하는 방법(Lag, Rolling, EWMA, ACF/PACF, Spectral, 자동 추출 등)의 원리와 적용 조건, leakage를 피하기 위한 정보 경계 설정 기준.
* [trace_channel_selection_taxonomy.md](trace_channel_selection_taxonomy.md) — 하나의 wafer에서 기록되는 다수의 trace channel 중 대표 channel을 고르는 절차를 Channel Grouping, Representative Channel Selection, Selection Validation의 세 단계로 정리.
* [wide_data_generalization.md](wide_data_generalization.md) — Sample 수에 비해 feature 수가 많은 wide / high-dimensional data에서 나타나는 overfitting, redundant feature, noisy feature, validation instability 문제와 각각의 진단·완화 방법을 정리.

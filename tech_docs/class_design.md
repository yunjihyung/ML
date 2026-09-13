# Class Design Guidelines for AI-Generated Python Code

Rev. 3 | Created: 2026-09-06 | Updated: 2026-09-06 22:20 KST

## 1. Problem

AI로 Python code를 생성하면 기능이 추가될 때마다 독립적인 `def`가 계속 늘어나는 형태가 쉽게 만들어진다. function 자체가 문제는 아니지만, 여러 function이 같은 `model`, `config`, `path` 같은 값을 반복해서 전달받거나 호출 순서에 의존하기 시작하면 코드의 책임과 state가 흩어진다.

예를 들어 다음 구조는 처음에는 단순하지만 function이 많아질수록 관리하기 어려워진다.

```python
def train_model(model, config, X, y):
    model.fit(X, y)


def evaluate_model(model, config, X, y):
    prediction = model.predict(X)
    return prediction


def save_model(model, config, output_path):
    ...
```

이 경우 `model`, `config`처럼 여러 기능이 공통으로 사용하는 값이 계속 argument로 전달된다. 장기적으로는 **같은 responsibility와 dependency를 공유하는 동작을 class로 묶고, 단순 계산은 function으로 유지하는 기준**이 필요하다.

## 2. Recommended Methods

### 2.1 Constructor Injection

class가 공통으로 사용하는 dependency는 method마다 전달하지 않고 constructor에서 한 번 받는다.

쉽게 말하면 **class가 계속 사용할 준비물은 `__init__()`에서 한 번 받아두는 방식**이다.

```python
class Trainer:
    def __init__(self, model, config):
        self.model = model
        self.config = config

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)
```

이 구조에서는 `fit()`과 `predict()`가 `model`, `config`를 반복해서 받을 필요가 없다. 또한 object 생성 시 필요한 dependency가 한곳에 보여서 class가 무엇을 사용하는지 확인하기 쉽다. Constructor Injection은 dependency 설정과 실제 사용을 분리하는 대표적인 Dependency Injection 방식이다. [[1](#ref-1)]

### 2.2 Single Responsibility

하나의 class에는 하나의 명확한 responsibility를 두는 것이 좋다.

쉽게 말하면 **한 class가 여러 직업을 동시에 가지지 않도록 나누는 방식**이다. 예를 들어 하나의 `Experiment` class 안에 loading, feature engineering, split, training, evaluation을 모두 넣기보다 역할별로 분리한다.

```text
DataLoader
FeatureEngineer
Splitter
Trainer
Evaluator
```

예를 들어 `Trainer`는 학습과 예측만 담당하고, `Evaluator`는 evaluation metric 계산만 담당한다. 이렇게 나누면 특정 기능을 수정할 때 다른 기능까지 함께 건드릴 가능성이 줄어든다.

반대로 `Manager`, `Utils`, `Helper`처럼 범위가 불명확한 class에 method가 계속 추가된다면 responsibility가 너무 넓어진 신호로 볼 수 있다.

### 2.3 Strategy and Composition

같은 역할에 여러 방법이 존재하는 경우에는 Strategy 형태로 분리할 수 있다.

쉽게 말하면 **바뀔 가능성이 있는 방법을 별도 부품으로 만들어 필요할 때 교체하는 방식**이다. 예를 들어 time-series data의 split 방식이 여러 개라면 하나의 function 안에서 `if`를 계속 늘리는 대신 각각을 독립된 object로 만든다.

나쁜 예시는 다음과 같다.

```python
def split_data(X, method):
    if method == "chronological":
        ...
    elif method == "rolling":
        ...
    elif method == "expanding":
        ...
```

개선하면 다음처럼 사용할 수 있다.

```python
class ChronologicalSplit:
    def split(self, X, y):
        ...


class ExpandingWindowSplit:
    def split(self, X, y):
        ...
```

상위 code에서는 필요한 object만 교체한다.

```python
splitter = ExpandingWindowSplit()
X_train, X_test, y_train, y_test = splitter.split(X, y)
```

이렇게 하면 새로운 split 방식이 추가되어도 기존 function의 조건문을 계속 수정하지 않아도 된다. 또한 inheritance 계층을 깊게 만드는 대신 작은 object를 조합하는 Composition을 사용하면 조합이 늘어날 때 subclass가 과도하게 증가하는 문제를 줄일 수 있다. [[2](#ref-2)]

### 2.4 Orchestrator

여러 component를 순서대로 실행해야 한다면 전체 흐름만 관리하는 `Runner` 또는 `Pipeline` class를 둘 수 있다.

쉽게 말하면 **각 component가 자기 일을 하고, `Runner`는 어떤 순서로 실행할지만 정하는 구조**이다.

```python
class ExperimentRunner:
    def __init__(self, feature_engineer, splitter, model, evaluator):
        self.feature_engineer = feature_engineer
        self.splitter = splitter
        self.model = model
        self.evaluator = evaluator

    def run(self, df):
        X, y = self.feature_engineer.transform(df)
        X_train, X_test, y_train, y_test = self.splitter.split(X, y)
        self.model.fit(X_train, y_train)
        prediction = self.model.predict(X_test)
        return self.evaluator.evaluate(y_test, prediction)
```

`ExperimentRunner`는 feature engineering이나 evaluation 알고리즘을 직접 구현하지 않고 **실행 순서만 관리**한다. 실제 구현은 각 component에 맡긴다.

scikit-learn도 `fit()`, `transform()`, `predict()`처럼 역할이 명확한 interface를 사용하고, `Pipeline` 같은 object가 여러 estimator를 조합하도록 설계되어 있다. [[3](#ref-3)]

## 3. Function and Class Criteria

모든 function을 class로 바꾸는 것은 좋은 설계가 아니다. state가 없고 입력을 받아 결과만 반환하는 단순 계산은 function으로 두는 편이 더 간단하다.

```python
def calculate_power(voltage, current):
    return voltage * current
```

반면 다음 조건이 있다면 class 사용을 고려할 수 있다.

- 여러 method가 같은 `model`, `config`, client 등의 dependency를 공유
- 생성 이후 유지해야 하는 state 존재
- 동일한 역할에 여러 구현이 존재
- 여러 단계의 실행 순서를 하나의 workflow로 관리

config나 result처럼 동작보다 데이터 보관이 중심인 object는 `dataclass`를 사용할 수 있다. Python `dataclass`는 `__init__()`과 `__repr__()` 같은 반복 코드를 자동 생성한다. [[4](#ref-4)]

## 4. Skill Candidate

AI code generation rule에는 다음 기준을 적용하는 것이 적절하다.

- 공유 dependency를 method마다 반복 전달하지 않고 constructor로 injection
- 하나의 class에 하나의 명확한 responsibility 부여
- 같은 역할의 구현이 여러 개이면 Strategy 형태로 분리
- inheritance보다 Composition 우선
- workflow class는 실행 순서만 관리하고 세부 알고리즘은 별도 component에 배치
- state가 없는 단순 계산은 function으로 유지
- `Manager`, `Utils`, `Helper`처럼 책임 범위가 불명확한 대형 class 생성 지양

핵심은 **`def`를 줄이기 위해 class를 쓰는 것이 아니라, AI가 새 기능을 추가할 때 기존 구조 안에서 적절한 책임 위치를 찾도록 만드는 것**이다.

## References

<a id="ref-1"></a>
[1] Martin Fowler, [Inversion of Control Containers and the Dependency Injection pattern](https://www.martinfowler.com/articles/injection.html), 2004. Accessed 2026-09-06.

<a id="ref-2"></a>
[2] Brandon Rhodes, [The Composition Over Inheritance Principle](https://python-patterns.guide/gang-of-four/composition-over-inheritance/). Accessed 2026-09-06.

<a id="ref-3"></a>
[3] scikit-learn Developers, [Developing scikit-learn estimators](https://scikit-learn.org/stable/developers/develop.html). Accessed 2026-09-06.

<a id="ref-4"></a>
[4] Python Software Foundation, [dataclasses — Data Classes](https://docs.python.org/3/library/dataclasses.html). Accessed 2026-09-06.

---

## Appendix A. Terminology

- **Composition**: inheritance 대신 여러 object를 조합하여 기능을 구성하는 방식
- **Constructor Injection**: class가 필요한 dependency를 constructor argument로 전달받는 방식
- **Dataclass**: 데이터 보관이 중심인 class의 반복 코드를 줄여주는 Python 기능
- **Dependency**: class 또는 function이 동작하기 위해 사용하는 외부 object나 service
- **Inheritance**: 기존 class를 기반으로 새로운 class의 기능을 확장하는 방식
- **Orchestrator**: 여러 component의 실행 순서와 연결을 담당하는 object
- **Responsibility**: class 또는 function이 맡는 명확한 역할
- **Single Responsibility**: 하나의 class가 하나의 명확한 responsibility에 집중하도록 하는 원칙
- **State**: object 내부에 저장되어 이후 method 호출에서도 유지되는 값
- **Strategy**: 동일한 역할의 여러 알고리즘을 공통 방식으로 교체할 수 있도록 분리하는 pattern

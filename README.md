# 2021 인공지능 온라인 경진대회

## 1. 프로그래밍 언어

본 경진대회용 프로그램은 [**Python**](https://www.python.org)을 프로그래밍 언어로 사용한다.

## 2. Python

### 2.1. 버전 정보

본 경진대회용 프로그램은 Python의 최신 안정화 버전인 **3.7.10**을 사용한다 (2021년 6월 11일 기준).

### 2.2. 외부 라이브러리

_추가 작성예정_

## 3. 프로그램 실행

```
$ python app.py (예정)
```

## 4. 분류모델 성능평가

분류모델의 성능평가는 **Macro F1 Score**를 사용하여 분류성능을 평가한다.

- True Positive (TP): 실제 **참**인 값을 **참**으로 예측한 결과의 집합
- True Negative (TN): 실제 **거짓**인 값을 **거짓**으로 예측한 결과의 집합
- False Positive (FP): 실제 **거짓**인 값을 **참**으로 예측한 결과의 집합
- False Negative (FN): 실제 **참**인 값을 **거짓**으로 예측한 결과의 집합

<center>

|               | 실제 참 | 실제 거짓 |
| :-----------: | :-----: | :-------: |
|  **예측 참**  |   TP    |    FP     |
| **예측 거짓** |   FN    |    FP     |

##### **표 1.** Confusion Matrix

</center>

정밀도 (Precision)

- 정의
- _Precision = TP / (TP + FP)_

재현율 (Recall)

- 정의
- _Recall = TP / (TP + TN)_

Macro F1 Score

- 정의
- _F1 score = 2 x (precision x recall) / (precision + recall)_

## 5. 환경변수 파일

.gitignore

- Git 관련 환경변수 파일

requirements.txt

- 파이썬 라이브러리 종속성 파일

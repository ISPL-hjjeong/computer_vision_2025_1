# 🧠 과제 1. 간단한 이미지 분류기 구현 (MNIST)

이 프로젝트는 TensorFlow와 Keras를 사용하여 MNIST 손글씨 숫자 이미지(28x28 픽셀, 흑백)를 분류하는 간단한 신경망 모델을 구축하고 학습시키는 과정입니다.

## 📌 1. 라이브러리 임포트

``` python

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
```
- tensorflow.keras를 활용하여 모델 구성, 학습, 평가 수행
- MNIST 데이터셋은 keras.datasets.mnist에서 직접 불러옴
- to_categorical()은 레이블을 원-핫 인코딩 형태로 변환

## 📥 2. 데이터 로드
```
(x_train, y_train), (x_test, y_test) = mnist.load_data()
```
- 6만 개의 훈련 이미지와 1만 개의 테스트 이미지 로딩
- 각 이미지는 28x28 크기의 흑백 이미지

## 🧹 3. 데이터 전처리
```
x_train = x_train / 255.0
x_test = x_test / 255.0
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)
```
- 이미지 픽셀값을 0~255에서 0~1로 정규화 (모델 학습 안정화)
- 레이블을 one-hot encoding 형식으로 변환 (예: 3 → [0,0,0,1,0,0,0,0,0,0])

## 🏗️ 4. 모델 구성

```
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])
```
- Sequential 모델 사용 (레이어를 순차적으로 쌓음)
- Flatten: 28x28 이미지를 784개의 1D 벡터로 변환
- Dense 레이어: 완전 연결 신경망
  - 첫 번째 은닉층: 128개 뉴런, ReLU 활성화
  - 두 번째 은닉층: 64개 뉴런, ReLU
  - 출력층: 10개 클래스, softmax 활성화 함수 사용
 
## ⚙️ 5. 모델 컴파일
```
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
```
- 옵티마이저: adam (효율적인 학습을 위한 알고리즘)
- 손실 함수: categorical_crossentropy (다중 클래스 분류에 적합)
- 평가 지표: 정확도 (accuracy)

## 🏃 6. 모델 학습

```
model.fit(x_train, y_train, epochs=5, batch_size=32, validation_split=0.1)
```
- 총 5 에폭(epoch) 동안 훈련
- 배치 크기: 32개
- 검증 데이터(validation_split=0.1): 훈련 데이터의 10%를 검증용으로 사용

## 📊 7. 모델 평가
```
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"테스트 정확도: {test_acc * 100:.2f}%")
```
- 테스트 데이터셋을 사용하여 모델의 최종 정확도 출력

![image](https://github.com/user-attachments/assets/0193092b-09a2-4521-8b9d-341fceca3f5b)

---

# 🧠 과제 2. CIFAR-10 데이터셋을 활용한 CNN 모델 구축

CIFAR-10 이미지 데이터셋을 이용하여 합성곱 신경망(CNN)을 구축하고, 이미지 분류를 수행합니다. 총 10개의 클래스에 대한 이미지 분류 모델을 구현합니다.

---

## 📌 1. 라이브러리 임포트

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
```

---

## 📥 2. 데이터 로드

```python
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
```

- 5만 개의 훈련 이미지, 1만 개의 테스트 이미지
- 이미지 크기: 32x32, 컬러(RGB)

---

## 🧹 3. 데이터 전처리

```python
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)
```

- 이미지 정규화 (0~1)
- 레이블 원-핫 인코딩

---

## 🏗️ 4. CNN 모델 구성

```python
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    MaxPooling2D((2, 2)),
    
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    
    Conv2D(128, (3, 3), activation='relu'),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])
```

---

## ⚙️ 5. 모델 컴파일

```python
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
```

---

## 🏃 6. 모델 학습

```python
model.fit(x_train, y_train, epochs=10, batch_size=64, validation_split=0.1)
```

- 에폭: 10
- 배치 크기: 64
- 검증 데이터: 훈련 데이터의 10%

---
![image](https://github.com/user-attachments/assets/df7062b0-b053-4206-ad55-66457ff24aaf)


# 이진화 및 히스토그램 구하기

![image](https://github.com/user-attachments/assets/a226ea57-0949-4263-921a-69574a3b748a)

## 설명
이 프로젝트에서는 주어진 이미지를 불러와서 다음 과정을 수행합니다.
1. 이미지를 그레이스케일로 변환
2. 특정 임계값을 설정하여 이진화
3. 이진화된 이미지의 히스토그램을 계산하고 시각화

## 요구사항
- `cv2.imread()`를 사용하여 이미지를 불러옵니다.
```python
# 이미지 로드
image_path = "./image/rose.png"
image = cv2.imread(image_path, cv2.IMREAD_COLOR)
```

- `cv2.cvtColor()`를 사용하여 그레이스케일로 변환합니다.
```python
# 그레이스케일 변환
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
```

- `cv2.threshold()`를 사용하여 이진화합니다.
```python
# 이진화 수행 (임계값 127 사용)
threshold_value = 127
_, binary_image = cv2.threshold(gray_image, threshold_value, 255, cv2.THRESH_BINARY)
```


- `cv2.calcHist()`를 사용하여 히스토그램을 계산하고, `matplotlib`를 사용하여 시각화합니다.
```python
# 히스토그램 계산
hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256])

# 원본 이미지, 이진화된 이미지, 히스토그램 출력
plt.figure(figsize=(12, 5))

plt.subplot(1, 3, 1)
plt.imshow(gray_image, cmap='gray')
plt.title('Grayscale Image')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(binary_image, cmap='gray')
plt.title('Binary Image')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.plot(hist, color='black')
plt.title('Histogram')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')

plt.show()
```

# 모폴로지 연산 적용하기
![image](https://github.com/user-attachments/assets/0eec0c8f-480e-4f05-991e-27f9cdebe3e5)

## 설명
이 프로젝트에서는 주어진 이진화된 이미지에 대해 다양한 **모폴로지 연산**을 적용합니다.  
모폴로지 연산은 이미지 처리에서 중요한 기법으로, 구조적 요소(커널)를 사용하여 이미지를 변형합니다.  
주요 연산은 다음과 같습니다.
- **팽창(Dilation)**: 객체를 확장하여 작은 구멍을 채우거나 강조
- **침식(Erosion)**: 객체를 축소하여 노이즈 제거 및 경계 부드럽게 만듦
- **열림(Open)**: 침식 후 팽창을 수행하여 작은 노이즈 제거
- **닫힘(Close)**: 팽창 후 침식을 수행하여 작은 구멍을 채움

---

## 요구사항 및 코드

### 1. `cv2.getStructuringElement()`를 사용하여 **사각형 커널(5x5)**을 생성합니다.
```python
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
```

### 2. cv2.morphologyEx()를 사용하여 각각의 모폴로지 연산을 적용합니다.
```python
# 팽창(Dilation) 적용
dilation = cv2.morphologyEx(binary, cv2.MORPH_DILATE, kernel)

# 침식(Erosion) 적용
erosion = cv2.morphologyEx(binary, cv2.MORPH_ERODE, kernel)

# 열림(Open) 적용
opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

# 닫힘(Close) 적용
closing = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
```

### 3. 원본 이미지와 모폴로지 연산 결과를 한 화면에 출력합니다.
```python
result = np.hstack([binary, dilation, erosion, opening, closing])

plt.figure(figsize=(12, 6))
plt.imshow(result, cmap='gray')
plt.title("Binary | Dilation | Erosion | Opening | Closing")
plt.axis("off")
plt.show()
```
---

# 기하 연산 및 선형 보간 적용하기

![image](https://github.com/user-attachments/assets/dfbf695e-9b43-4265-bffd-550bac252bcd)


## 설명
이 프로젝트에서는 주어진 이미지를 변환하는 **기하 변환(Geometric Transformation)** 을 적용합니다.  
수행할 작업은 다음과 같습니다.
1. **이미지를 45도 회전**합니다.
2. **회전된 이미지를 1.5배 확대**합니다.
3. **선형 보간(Bilinear Interpolation)** 을 적용하여 부드러운 변환을 수행합니다.

---

## 요구사항 및 코드

### 1. `cv2.getRotationMatrix2D()`를 사용하여 회전 변환 행렬을 생성하세요.
```python
# 이미지 크기 가져오기 (cols: 너비, rows: 높이)
rows, cols = image.shape[:2]

# 회전 변환 행렬 생성 (중심: (cols/2, rows/2), 각도: 45도, 배율: 1.5)
M = cv2.getRotationMatrix2D((cols/2, rows/2), 45, 1.5)
```

### 2. cv2.warpAffine()를 사용하여 이미지를 회전 및 확대하세요.
```python
# 회전 및 확대 적용
rotated_scaled_image = cv2.warpAffine(image, M, (int(cols*1.5), int(rows*1.5)), flags=cv2.INTER_LINEAR)
```

### 3. cv2.INTER_LINEAR를 사용하여 선형 보간을 적용하세요.
```python
# warpAffine() 함수에서 INTER_LINEAR를 사용하여 부드러운 변환 적용
rotated_scaled_image = cv2.warpAffine(image, M, (int(cols*1.5), int(rows*1.5)), flags=cv2.INTER_LINEAR)
```

### 4. 원본 이미지와 회전 및 확대된 이미지를 한 화면에 비교하세요.
```python
plt.figure(figsize=(10, 5))

# 원본 이미지 출력
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("Original Image")
plt.axis("off")

# 회전 및 확대된 이미지 출력
plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(rotated_scaled_image, cv2.COLOR_BGR2RGB))
plt.title("Rotated & Scaled Image")
plt.axis("off")

plt.tight_layout()
plt.show()
```

# 📌 1. Sobel 필터를 활용한 엣지(Edge) 검출 및 시각화

이 프로젝트는 OpenCV를 활용하여 **Sobel 필터를 이용한 엣지 검출**을 수행하고,  
검출된 엣지 강도를 시각화하는 실습 과제입니다.

![image](https://github.com/user-attachments/assets/b18a7bfc-9e92-48ee-9802-2f8d1e2b6a8a)

---

## 📷 실습 개요

- 입력 이미지를 Grayscale로 변환  
- Sobel 필터를 통해 X축, Y축 방향의 엣지 검출  
- 엣지 강도(Gradient Magnitude)를 계산하고 시각화  

---

## 🧰 사용 라이브러리

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
```

---

## 📝 코드 설명

### 1. 이미지 불러오기
```python
image_path = "./image/edgeDetectionImage.jpg"
image = cv2.imread(image_path)
```

### 2. 그레이스케일로 변환
```python
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
```

### 3. Sobel 필터로 엣지 검출
```python
sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
```

### 4. 엣지 강도 계산
```python
magnitude = cv2.magnitude(sobel_x, sobel_y)
```

### 5. 이미지 포맷 변환 (uint8)
```python
edge_image = cv2.convertScaleAbs(magnitude)
```

### 6. 결과 시각화
```python
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.title("Grayscale Image")
plt.imshow(gray, cmap='gray')
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("Edge Strength Image")
plt.imshow(edge_image, cmap='gray')
plt.axis("off")

plt.tight_layout()
plt.show()
```

---

## ✅ 참고 사항

- `ksize`는 3, 5 등 홀수값 사용 가능  
- `cv2.Sobel()`은 노이즈에 민감 → `cv2.GaussianBlur()`로 선처리 가능  

---

# 📌 2. Canny 엣지 및 허프(Hough) 변환을 이용한 직선 검출

이 실습은 OpenCV를 이용해 Canny 엣지 검출기를 사용하고, Hough 변환을 통해 이미지 내의 직선을 검출 및 시각화하는 프로젝트입니다.


![image](https://github.com/user-attachments/assets/54a78e1b-bf66-430c-a079-b8d9dbb14e4d)

---

## 🧾 실습 목표

- Canny 엣지 알고리즘을 사용하여 엣지 맵 생성  
- 허프 변환(Hough Transform)을 통해 직선 검출  
- 검출된 직선을 원본 이미지에 빨간색으로 시각화  

---

## 📌 핵심 함수 설명

### `cv2.Canny()`
- 목적: 엣지를 검출하여 윤곽선 맵 생성  
- 임계값: `threshold1=100`, `threshold2=200`

### `cv2.HoughLinesP()`
- 목적: 엣지 맵을 기반으로 직선을 검출하는 확률적 허프 변환  
- 주요 파라미터:
  - `rho=1` (픽셀 거리 단위)  
  - `theta=np.pi/180` (라디안 각도 단위)  
  - `threshold=100` (허프 누적값 기준)  
  - `minLineLength=50` (최소 직선 길이)  
  - `maxLineGap=10` (최대 허용 간격)

### `cv2.line()`
- `(0, 0, 255)` (빨간색)으로 검출된 선을 시각화  
- 두께는 `2`

---

## 🖼 출력 결과 예시

- **왼쪽**: Canny로 추출된 엣지 맵  
- **오른쪽**: 원본 이미지 위에 검출된 직선을 빨간색으로 시각화  

---

## ✅ 참고 사항

- Canny 임계값 조정으로 민감도 조절 가능  
- Hough 변환 파라미터 조절로 직선 검출 정확도 개선  
- 노이즈 제거용 `cv2.GaussianBlur()` 사용 권장  

---

# 📌 3. GrabCut을 이용한 대화식 영역 분할 및 객체 추출

OpenCV의 **GrabCut 알고리즘**을 사용하여 사용자가 지정한 사각형 영역을 기반으로 객체를 분리합니다.


![image](https://github.com/user-attachments/assets/22309077-293e-40ed-8ef8-4c4859b244c1)

---

## 🧾 실습 목표

- GrabCut을 이용한 반자동 객체 분리 수행  
- 마스크 형태로 결과 시각화  
- 원본 이미지에서 배경 제거 후 객체만 추출  

---

## 📌 주요 단계

1. 이미지 불러오기 (`cv2.imread`)  
2. 마스크 및 배경/전경 모델 초기화  
3. 사각형 ROI 설정 (`x, y, width, height`)  
4. `cv2.grabCut()` 수행  
5. `np.where()`로 최종 마스크 생성  
6. 마스크로 원본에서 객체만 추출  

---

## 💡 주요 함수 설명

### `cv2.grabCut()`
- GrabCut 알고리즘 수행  
- 모드: `cv2.GC_INIT_WITH_RECT`  
- 입력값: 이미지, 마스크, 사각형, bgdModel, fgdModel, 반복횟수 등  

### `np.where()`
- 전경(`cv2.GC_FGD`) 또는 유사 전경(`cv2.GC_PR_FGD`) 픽셀을 1로 설정  
- 나머지는 0으로 설정하여 배경 제거  

---

## 🖼 출력 결과

- **Original Image**: 원본 이미지  
- **GrabCut Mask**: 생성된 마스크 시각화 (흑백)  
- **Foreground Extracted**: 배경 제거 후 객체만 남긴 이미지  

> 💡 **Tip**: 사각형 ROI는 객체를 충분히 감싸도록 설정하면 결과가 좋아집니다.

---

## ✅ 참고 사항

- `bgdModel`, `fgdModel`: `np.zeros((1, 65), np.float64)`로 초기화  
- GrabCut은 반자동 → 추가 마스크 보정도 가능  
- 마우스 입력으로 ROI 직접 지정하면 성능 개선 가능


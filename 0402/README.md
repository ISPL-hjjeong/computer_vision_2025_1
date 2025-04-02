# 📌 SIFT를 이용한 특징점 검출 및 시각화
![image](https://github.com/user-attachments/assets/63de966e-d2ee-4b63-ab7c-987f24efa693)

## 📷 개요
OpenCV의 SIFT (Scale-Invariant Feature Transform) 알고리즘을 사용하여 이미지의 특징점을 검출하고 이를 시각화하는 방법을 다룹니다.

## 🧰 코드 설명

1. **이미지 로드 및 전처리**
   - `cv2.imread(image_path)`: 지정된 경로에서 이미지를 로드합니다.
   - `cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)`: 컬러 이미지를 그레이스케일로 변환합니다.

2. **SIFT 객체 생성 및 특징점 검출**
   - `cv2.SIFT_create()`: SIFT 알고리즘을 생성합니다.
   - `sift.detectAndCompute(gray, None)`: 입력 이미지에서 특징점을 검출하고 기술자를 계산합니다.

3. **특징점 시각화**
   - `cv2.drawKeypoints()`: 검출된 특징점을 원본 이미지 위에 그립니다.

4. **이미지 출력**
   - `matplotlib`을 이용하여 원본 이미지와 특징점이 시각화된 이미지를 나란히 출력합니다.

5. **원의 크기**
   - 특징점이 존재하는 이미지 영역의 스케일(크기) 을 의미합니다.
   - 작은 원은 작은 특징을, 큰 원은 큰 특징을 나타냅니다.


## 📝 전체 코드

``` python
import cv2
import matplotlib.pyplot as plt


image_path = "./image/mot_color70.jpg"  
image = cv2.imread(image_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# SIFT 객체 생성
sift = cv2.SIFT_create()

# 특징점 검출 및 기술자 계산
keypoints, descriptors = sift.detectAndCompute(gray, None)

# 특징점을 원본 이미지에 시각화
drawn_image = cv2.drawKeypoints(image, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# 결과 출력
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("SIFT Keypoints")
plt.imshow(cv2.cvtColor(drawn_image, cv2.COLOR_BGR2RGB))
plt.axis("off")

plt.show()
```

# 📌 SIFT 기반 특징점 매칭

![image](https://github.com/user-attachments/assets/ca304baf-a337-49d1-97a4-8aca88e65c08)

## 📷 개요

SIFT (Scale-Invariant Feature Transform) 알고리즘을 사용하여 두 개의 이미지 간 특징점을 검출하고 매칭하는 과정을 구현. OpenCV 라이브러리를 활용하여 특징점 추출, 기술자 계산, 그리고 BFMatcher를 사용한 매칭을 수행.

## 🧰 코드 설명

1. **이미지 불러오기 및 그레이스케일 변환**
   - `cv2.imread()` : 이미지를 불러옵니다.
   - `cv2.cvtColor(..., cv2.COLOR_BGR2GRAY)` : SIFT는 흑백 이미지에서 동작하므로 그레이스케일 변환을 수행합니다.
     
2. **SIFT 특징점 검출 및 기술자(Descriptor) 계산**
  - `cv2.SIFT_create()` : SIFT 객체를 생성합니다.
  - `detectAndCompute()` : 각 이미지에서 특징점(Keypoints)과 기술자(Descriptors) 를 추출합니다.

3. **BFMatcher를 사용한 특징점 매칭**
- `cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)` : Brute-Force 매처를 생성합니다.

- `NORM_L2` : SIFT는 L2 거리(유클리드 거리)를 사용하여 특징점을 비교합니다.

- `crossCheck=True` : 서로 일치하는 특징점만 유지하여 정확도를 향상시킵니다.

- `matches = sorted(matches, key=lambda x: x.distance)` : 매칭된 특징점을 거리가 짧은 순으로 정렬하여 신뢰도 높은 매칭을 우선적으로 표시합니다.

4. **실행 결과**
- SIFT 알고리즘을 활용하여 두 이미지 간의 특징점이 매칭된 결과를 확인할 수 있습니다.
매칭된 선이 많을수록 두 이미지가 유사하다는 의미이며, 이는 다양한 영상 분석 및 객체 인식 분야에서 활용될 수 있습니다.

## 📝 전체 코드

``` python
import cv2
import matplotlib.pyplot as plt

image1_path = "./image/mot_color70.jpg"
image2_path = "./image/mot_color83.jpg" 
image1 = cv2.imread(image1_path)
image2 = cv2.imread(image2_path)
gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

# SIFT 객체 생성
sift = cv2.SIFT_create()

# 특징점 검출 및 기술자 계산
keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)

# BFMatcher를 사용한 특징점 매칭
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
matches = bf.match(descriptors1, descriptors2)
matches = sorted(matches, key=lambda x: x.distance)  # 거리순 정렬

# 매칭 결과 시각화
drawn_matches = cv2.drawMatches(image1, keypoints1, image2, keypoints2, matches[:50], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# 결과 출력
plt.figure(figsize=(12, 6))
plt.imshow(cv2.cvtColor(drawn_matches, cv2.COLOR_BGR2RGB))
plt.title("SIFT Feature Matching")
plt.axis("off")
plt.show()

```


# 📌 호모그래피를 이용한 이미지 정합 (Image Alignment)
![image](https://github.com/user-attachments/assets/ce09cc49-e129-431a-9b8b-88960233d9b5)
## 📷 개요
두 장의 이미지를 SIFT(Scale-Invariant Feature Transform)를 이용하여 정합(Alignment)한 뒤, 두 이미지가 feature 기반으로 겹치도록 합성하는 Python 코드를 설명합니다. 두 이미지 간의 공통된 특징점을 찾아 호모그래피(Homography) 행렬을 계산하고, 이를 통해 한 이미지를 다른 이미지 위에 정렬 후 블렌딩하여 시각적으로 겹치는 이미지를 생성

## 🧰 코드 설명
1. **SIFT를 사용한 특징점 검출 및 기술자 계산**
``` python
sift = cv2.SIFT_create()
keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)
```
- SIFT 알고리즘을 생성하고 두 이미지에서 특징점(Keypoints)과 기술자(Descriptors)를 추출

  2. **특징점 매칭 및 좋은 매칭 필터링**
``` python
bf = cv2.BFMatcher()
matches = bf.knnMatch(descriptors1, descriptors2, k=2)

good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)
```

- ```BFMatcher```를 통해 최근접 이웃(2개)을 찾고, Lowe's ratio test를 적용하여 좋은 매칭만 추려냄

 3. **호모그래피 계산 및 이미지 정합**
``` python
src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
```
- 좋은 매칭점을 바탕으로 소스 이미지(image1)에서 대상 이미지(image2)로의 변환 행렬(Homography)을 계산합니다.

  4. **이미지 변환 및 블렌딩**
``` python
warped_image1 = cv2.warpPerspective(image1, H, (w2, h2))
```
- 변환 행렬을 이용하여 image1을 image2와 동일한 좌표계로 정합합니다.

``` python
blended = np.where(mask_warped == 0, image2, cv2.addWeighted(warped_image1, 0.5, image2, 0.5, 0))
```
- 마스크를 통해 두 이미지를 단순 평균 블렌딩하여 겹칩니다.

## 📝 전체 코드

``` python
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 이미지 로드
image1_path = "./image/img1.jpg"
image2_path = "./image/img2.jpg"
image1 = cv2.imread(image1_path)
image2 = cv2.imread(image2_path)
gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

# SIFT 객체 생성
sift = cv2.SIFT_create()

# 특징점 검출 및 기술자 계산
keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)

# BFMatcher를 사용한 특징점 매칭
bf = cv2.BFMatcher()
matches = bf.knnMatch(descriptors1, descriptors2, k=2)

# 좋은 매칭점 선택
good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)

# 최소 매칭점 개수 설정
MIN_MATCH_COUNT = 10
if len(good_matches) > MIN_MATCH_COUNT:
    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    
    # 호모그래피 계산
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    
    # 이미지2의 크기 기준으로 이미지1 변환
    h2, w2 = image2.shape[:2]
    warped_image1 = cv2.warpPerspective(image1, H, (w2, h2))

    # 두 이미지를 블렌딩하여 겹치기
    # 마스크 생성
    mask_warped = (warped_image1 > 0).astype(np.uint8) * 255
    mask_image2 = (image2 > 0).astype(np.uint8) * 255

    # 이미지 오버랩 - 간단한 평균 블렌딩
    blended = np.where(mask_warped == 0, image2, 
                       cv2.addWeighted(warped_image1, 0.5, image2, 0.5, 0))

    # 결과 출력
    plt.figure(figsize=(10, 6))
    plt.imshow(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB))
    plt.title("Overlapped Image")
    plt.axis("off")
    plt.show()

else:
    print("Not enough matches are found - {}/{}".format(len(good_matches), MIN_MATCH_COUNT))

```

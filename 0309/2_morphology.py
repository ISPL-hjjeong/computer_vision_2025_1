import cv2
import numpy as np
import matplotlib.pyplot as plt

# 이미지 로드
image_path = "./image/mistyroad.jpg"
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# 이진화 수행
threshold_value = 127
_, binary = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY)

# 사각형 커널 생성 (5x5)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

# 모폴로지 연산 적용
dilation = cv2.morphologyEx(binary, cv2.MORPH_DILATE, kernel)
erosion = cv2.morphologyEx(binary, cv2.MORPH_ERODE, kernel)
opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
closing = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

# 결과 출력
result = np.hstack([binary, dilation, erosion, opening, closing])

plt.figure(figsize=(12, 6))
plt.imshow(result, cmap='gray')
plt.title("Binary | Dilation | Erosion | Opening | Closing")
plt.axis("off")
plt.show()

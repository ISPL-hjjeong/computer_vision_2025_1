import cv2
import numpy as np
import matplotlib.pyplot as plt

# 이미지 로드
image_path = "./image/rose.png"
image = cv2.imread(image_path, cv2.IMREAD_COLOR)


# 그레이스케일 변환
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 이진화 수행
threshold_value = 127
_, binary = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)

# 히스토그램 계산
hist = cv2.calcHist([gray], [0], None, [256], [0, 256])

# 결과 출력
plt.figure(figsize=(12, 5))

# 원본 이미지 출력
plt.subplot(1, 4, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("Original Image")
plt.axis("off")

# 그레이스케일 이미지 출력
plt.subplot(1, 4, 2)
plt.imshow(gray, cmap="gray")
plt.title("Grayscale Image")
plt.axis("off")

# 이진화 이미지 출력
plt.subplot(1, 4, 3)
plt.imshow(binary, cmap="gray")
plt.title("Binary Image")
plt.axis("off")

# 히스토그램 출력
plt.subplot(1, 4, 4)
plt.plot(hist, color='black')
plt.title("Histogram")
plt.xlabel("Pixel Value")
plt.ylabel("Frequency")

plt.tight_layout()
plt.show()

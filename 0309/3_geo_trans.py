import cv2
import numpy as np
import matplotlib.pyplot as plt

# 이미지 로드
image_path = "./image/mistyroad.jpg"
image = cv2.imread(image_path, cv2.IMREAD_COLOR)

# 이미지 크기 가져오기 (cols: 너비, rows: 높이)
rows, cols = image.shape[:2]

# 회전 변환 행렬 생성 (중심: (cols/2, rows/2), 각도: 45도, 배율: 1.5)
M = cv2.getRotationMatrix2D((cols/2, rows/2), 45, 1.5)

# 회전 및 확대 적용
rotated_scaled_image = cv2.warpAffine(image, M, (int(cols*1.5), int(rows*1.5)), flags=cv2.INTER_LINEAR)

# 결과 출력
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

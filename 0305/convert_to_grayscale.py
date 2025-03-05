import cv2
import numpy as np

img = cv2.imread('./image/soccer.jpg', cv2.IMREAD_COLOR)
if img is None:
    print("이미지를 불러올 수 없습니다.")
    exit()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray_3channel = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

combined = np.hstack((img, gray_3channel))

cv2.namedWindow('Original and Grayscale', cv2.WINDOW_NORMAL)


cv2.imshow('Original and Grayscale', combined)
cv2.waitKey(0)
cv2.destroyAllWindows()

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

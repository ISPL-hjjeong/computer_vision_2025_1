<!DOCTYPE html>
<html lang="ko">
<head>
  
</head>
<body>

  <h1>📌1. Sobel 필터를 활용한 엣지(Edge) 검출 및 시각화</h1>

  <p>
    이 프로젝트는 OpenCV를 활용하여 <strong>Sobel 필터를 이용한 엣지 검출</strong>을 수행하고,
    검출된 엣지 강도를 시각화하는 실습 과제입니다.
  </p>

  <hr>

  <h2>📷 실습 개요</h2>
  <ul>
    <li>입력 이미지를 Grayscale로 변환</li>
    <li>Sobel 필터를 통해 X축, Y축 방향의 엣지 검출</li>
    <li>엣지 강도(Gradient Magnitude)를 계산하고 시각화</li>
  </ul>

  <hr>

  <h2>🧰 사용 라이브러리</h2>
  <pre><code>import cv2
import numpy as np
import matplotlib.pyplot as plt</code></pre>

  <hr>

  <h2>📝 코드 설명</h2>

  <h3>1. 이미지 불러오기</h3>
  <pre><code>image_path = "./image/edgeDetectionImage.jpg"
image = cv2.imread(image_path)</code></pre>
  <p>- <code>cv2.imread()</code>로 이미지를 읽어옵니다.</p>

  <h3>2. 그레이스케일로 변환</h3>
  <pre><code>gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)</code></pre>
  <p>- 엣지 검출은 색상 정보보다는 <span class="highlight">명암의 차이</span>를 기반으로 하기 때문에 Grayscale 변환이 필요합니다.</p>

  <h3>3. Sobel 필터로 엣지 검출</h3>
  <pre><code>sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)</code></pre>
  <ul>
    <li><code>ksize=3</code>: Sobel 커널의 크기를 3으로 설정</li>
    <li><code>dx=1, dy=0</code>: X축 방향 엣지 검출</li>
    <li><code>dx=0, dy=1</code>: Y축 방향 엣지 검출</li>
    <li><code>cv2.CV_64F</code>: 연산 결과의 정밀도를 위해 64비트 float 사용</li>
  </ul>

  <h3>4. 엣지 강도 계산</h3>
  <pre><code>magnitude = cv2.magnitude(sobel_x, sobel_y)</code></pre>
  <p>- 각 픽셀의 X, Y 방향 그래디언트를 사용하여 <strong>엣지의 크기(강도)</strong>를 계산합니다.</p>
  <p>- 수식: <code>magnitude = sqrt(sobel_x² + sobel_y²)</code></p>

  <h3>5. 이미지 포맷 변환 (uint8)</h3>
  <pre><code>edge_image = cv2.convertScaleAbs(magnitude)</code></pre>
  <p>- <code>cv2.convertScaleAbs()</code>를 이용해 float 형식의 엣지 강도 이미지를 <code>uint8</code>로 변환하여 시각화합니다.</p>

  <h3>6. 결과 시각화</h3>
  <pre><code>plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.title("Grayscale Image")
plt.imshow(gray, cmap='gray')
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("Edge Strength Image")
plt.imshow(edge_image, cmap='gray')
plt.axis("off")

plt.tight_layout()
plt.show()</code></pre>
  <p>- <strong>matplotlib</strong>를 사용하여 원본 Grayscale 이미지와 엣지 강도 이미지를 나란히 출력합니다.</p>
  <p>- <code>cmap='gray'</code>를 통해 흑백 이미지로 시각화합니다.</p>

  <hr>

  <h2>✅ 참고 사항</h2>
  <ul>
    <li><code>ksize</code>는 3, 5 등 홀수값 사용 가능. 커질수록 더 부드럽고 큰 영역을 탐지함.</li>
    <li><code>cv2.Sobel()</code>은 노이즈에 민감할 수 있으므로, 필요시 <code>cv2.GaussianBlur()</code>로 노이즈 제거 후 적용 가능</li>
  </ul>
  
<hr>

<h1>📌2. Canny 엣지 및 허프(Hough) 변환을 이용한 직선 검출</h1>

  <p>이 실습은 OpenCV를 이용해 Canny 엣지 검출기를 사용하고, Hough 변환을 통해 이미지 내의 직선을 검출 및 시각화하는 프로젝트입니다.</p>

  <hr>

  <h2>🧾 실습 목표</h2>
  <ul>
    <li>Canny 엣지 알고리즘을 사용하여 엣지 맵 생성</li>
    <li>허프 변환(Hough Transform)을 통해 직선 검출</li>
    <li>검출된 직선을 원본 이미지에 빨간색으로 시각화</li>
  </ul>

<h2>📌 핵심 함수 설명</h2>

  <h3>cv2.Canny()</h3>
  <ul>
    <li>목적: 엣지를 검출하여 윤곽선 맵 생성</li>
    <li><code>threshold1=100</code>, <code>threshold2=200</code> 사용</li>
  </ul>

  <h3>cv2.HoughLinesP()</h3>
  <ul>
    <li>목적: 엣지 맵을 기반으로 직선을 검출하는 확률적 허프 변환 함수</li>
    <li>매개변수 설명:</li>
    <ul>
      <li><code>rho=1</code>: 거리 해상도 (픽셀 단위)</li>
      <li><code>theta=np.pi/180</code>: 각도 해상도 (라디안 단위)</li>
      <li><code>threshold=100</code>: 직선으로 판단할 최소 교차 수</li>
      <li><code>minLineLength=50</code>: 검출할 최소 직선 길이</li>
      <li><code>maxLineGap=10</code>: 직선으로 간주할 최대 간격</li>
    </ul>
  </ul>

  <h3>cv2.line()</h3>
  <ul>
    <li>각 직선을 원본 이미지에 <code>(0, 0, 255)</code> (빨간색)으로 시각화</li>
    <li>두께는 <code>2</code>로 설정</li>
  </ul>

  <h2>🖼 출력 결과 예시</h2>
  <ul>
    <li><strong>왼쪽</strong>: Canny로 추출된 엣지 맵</li>
    <li><strong>오른쪽</strong>: 원본 이미지 위에 검출된 직선을 빨간색으로 표시</li>
  </ul>
  <p>(결과 이미지는 코드 실행 시 출력됨)</p>

  <hr>

  <h2>✅ 참고 사항</h2>
  <ul>
    <li>Canny 임계값(<code>threshold1</code>, <code>threshold2</code>)를 조절하면 민감도 조절 가능.</li>
    <li>허프 파라미터도 조정하여 더 정확한 직선 검출.</li>
    <li>이미지에 노이즈가 많을 경우, <code>GaussianBlur</code>를 사전 적용.</li>
  </ul>

<hr>

<h1>📌 3. GrabCut을 이용한 대화식 영역 분할 및 객체 추출</h1>

  <p>이 실습은 사용자가 지정한 사각형 영역을 기반으로 OpenCV의 <strong>GrabCut 알고리즘</strong>을 사용하여 객체를 분리하는 예제입니다.</p>

  <hr>

  <h2>🧾 실습 목표</h2>
  <ul>
    <li>GrabCut을 이용한 반자동 객체 분리 수행</li>
    <li>객체 추출 결과를 마스크 형태로 시각화</li>
    <li>원본 이미지에서 배경을 제거하고 객체만 남긴 결과 생성</li>
  </ul>

  <h2>📌 주요 단계</h2>
  <ol>
    <li>이미지 불러오기 (<code>cv2.imread()</code>)</li>
    <li>초기 마스크 및 모델(bgdModel, fgdModel) 초기화</li>
    <li>사각형 영역 설정 (<code>(x, y, width, height)</code>)</li>
    <li><code>cv2.grabCut()</code> 함수로 마스크 업데이트</li>
    <li><code>np.where()</code>를 통해 최종 객체 마스크 생성</li>
    <li>원본 이미지에 마스크를 곱해 배경 제거</li>
  </ol>

  <h2>💡 주요 함수 설명</h2>

  <h3>cv2.grabCut()</h3>
  <ul>
    <li>GrabCut 알고리즘 수행</li>
    <li>모드: <code>cv2.GC_INIT_WITH_RECT</code> 사용</li>
    <li>입력: 이미지, 마스크, 사각형, 배경/전경 모델, 반복 횟수, 초기화 모드</li>
  </ul>

  <h3>np.where()</h3>
  <ul>
    <li>마스크에서 전경(<code>cv2.GC_FGD</code>) 또는 가능성 있는 전경(<code>cv2.GC_PR_FGD</code>)을 추출</li>
    <li>값이 1인 부분만 남기고 나머지는 0으로 처리</li>
  </ul>

  <h2>🖼 출력 결과</h2>
  <ul>
    <li><strong>Original Image</strong>: 원본 이미지</li>
    <li><strong>GrabCut Mask</strong>: GrabCut으로 생성된 마스크 (Gray 시각화)</li>
    <li><strong>Foreground Extracted</strong>: 배경이 제거된 최종 객체 이미지</li>
  </ul>

  <div class="note">
    🔍 <strong>Tip:</strong> 사각형 위치는 이미지 내 객체를 최대한 감싸도록 설정하면 더욱 정확한 분할 결과를 얻을 수 있습니다.
  </div>

  <hr>

  <h2>✅ 참고 사항</h2>
  <ul>
    <li><code>bgdModel</code>과 <code>fgdModel</code>은 <code>np.zeros((1, 65), np.float64)</code>로 초기화해야 합니다.</li>
    <li>GrabCut은 반자동 방식이므로 추가 마스크 보정으로 성능 향상 가능</li>
    <li>인터랙티브한 분할은 마우스로 ROI를 지정하여 개선할 수 있습니다.</li>
  </ul>


</body>
</html>

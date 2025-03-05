![image](https://github.com/user-attachments/assets/b094b287-5433-4de5-a348-82561221efbf)# 과제 1: 이미지 불러오기 및 그레이스케일 변환

## 개요 
- OpenCV를 사용하여 이미지를 불러온 뒤, 컬러 이미지를 그레이스케일로 변환함.
- 원본 이미지와 그레이스케일 이미지를 가로로 연결(hstack)하여 하나의 창에 표시함.

## 주요 함수
- cv2.imread(path, cv2.IMREAD_COLOR): 컬러 모드로 이미지 읽기
- cv2.cvtColor(img, cv2.COLOR_BGR2GRAY): BGR → 그레이스케일 변환
- np.hstack(): 두 이미지를 가로 방향으로 연결
- cv2.imshow(): 창에 이미지 표시
- cv2.waitKey(0): 키 입력 대기



# 과제 2: 웹캠 영상에서 에지(Canny) 검출
## 개요
- 웹캠에서 실시간으로 영상을 받아와 각 프레임에 Canny 에지 검출을 적용.
- 원본 영상과 에지 검출 영상을 가로로 연결(hstack)하여 실시간으로 화면에 표시
- q 키를 누르면 종료

##주요 함수
- cv2.VideoCapture(0): 기본 웹캠(0번) 연결
- cv2.Canny(gray, 100, 200): 그레이스케일 영상에 Canny 에지 검출
- cv2.waitKey(1): 1ms 동안 키 입력 대기 (실시간 스트리밍)
- cap.release(): 웹캠 자원 해제

# 과제 3: 마우스로 영역 선택 및 ROI 추출

## 개요
- 이미지를 불러온 뒤, 마우스로 드래그하여 관심 영역(ROI)을 선택
- 선택한 영역에 사각형을 그려 표시하고, 해당 영역을 별도 창에 표시하거나 파일로 저장
- s 키를 누르면 ROI가 파일로 저장되고, q 키를 누르면 프로그램이 종료

## 주요 함수
- cv2.setMouseCallback(windowName, callbackFunction): 특정 창에 마우스 이벤트 콜백 설정
### 마우스 이벤트 상수:
- cv2.EVENT_LBUTTONDOWN: 마우스 왼쪽 버튼 누름
- cv2.EVENT_MOUSEMOVE: 마우스 이동
- cv2.EVENT_LBUTTONUP: 마우스 왼쪽 버튼 뗌
- cv2.rectangle(img, pt1, pt2, color, thickness): 사각형 그리기
- ROI 추출: roi = img[y1:y2, x1:x2]


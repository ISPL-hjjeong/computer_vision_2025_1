# 🎯 과제1. SORT 알고리즘을 활용한 다중 객체 추적기 구현

이 프로젝트는 [YOLOv4](https://github.com/AlexeyAB/darknet)와 [SORT (Simple Online and Realtime Tracking)](https://github.com/abewley/sort) 알고리즘을 결합하여 비디오 내 다중 객체를 실시간으로 추적하는 시스템을 구현합니다.

![image](https://github.com/user-attachments/assets/439f1ac5-a7d3-4bcd-9a35-fa50c444c256)

## 📌 프로젝트 설명

이 실습에서는 SORT 알고리즘을 사용하여 비디오에서 다중 객체를 실시간으로 추적하는 프로그램을 구현합니다.  
이를 통해 객체 추적의 기본 개념과 SORT 알고리즘의 적용 방법을 학습할 수 있습니다.

## ✅ 요구사항

- **객체 검출기 구현**: YOLOv4와 같은 사전 훈련된 모델을 사용하여 각 프레임에서 객체를 검출합니다.
- **SORT 추적기 초기화**: 검출된 객체의 경계 상자를 입력으로 받아 SORT를 초기화합니다.
- **객체 추적 유지**: 각 프레임마다 검출된 객체와 기존 추적 객체를 연관시켜 추적을 유지합니다.
- **결과 시각화**: 추적된 객체에 고유 ID를 부여하고, 해당 ID와 경계 상자를 비디오에 실시간으로 표시합니다.

## 💡 힌트

- OpenCV의 DNN 모듈로 YOLOv4 모델을 로드합니다.
- SORT는 칼만 필터 및 헝가리안 알고리즘을 활용하여 추적 상태를 예측하고 연관시킵니다.
- 정확한 appearance 추적이 필요할 경우 [Deep SORT](https://github.com/nwojke/deep_sort) 같은 확장 버전을 고려해볼 수 있습니다.

## 🧠 주요 라이브러리

- `OpenCV`: 비디오 처리 및 DNN 기반 객체 검출
- `NumPy`: 수치 연산
- `SORT`: 객체 추적 알고리즘


## 🧪 코드 설명 (main.py)

### YOLOv4 설정
```
config_path = "yolov4.cfg"
weights_path = "yolov4.weights"
names_path = "coco.names"
```
### 클래스 이름 불러오기
```
with open(names_path, 'r') as f:
    classes = f.read().strip().split('\n')
```
### YOLOv4 모델 로드
```
net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
```
### 출력 레이어 이름 추출
```
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
```
### SORT 추적기 초기화
```
tracker = Sort()
```
### 비디오 열기
```
cap = cv2.VideoCapture("video.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
```
### YOLOv4 객체 검출
```
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)

    boxes, confidences, class_ids = [], [], []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.4:
                cx, cy, bw, bh = detection[0:4] * np.array([w, h, w, h])
                x, y = int(cx - bw/2), int(cy - bh/2)
                boxes.append([x, y, int(bw), int(bh)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.4, 0.3)
    detections = []

    if len(indices) > 0:
        for i in indices.flatten():
            x, y, bw, bh = boxes[i]
            detections.append([x, y, x + bw, y + bh, confidences[i]])

    dets = np.array(detections)
    tracks = tracker.update(dets)

    for track in tracks:
        x1, y1, x2, y2, track_id = track.astype(int)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'ID {int(track_id)}', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("SORT Tracker", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

## 🎥 실행 결과
실행 시, 추적된 객체에 고유 ID가 할당되고 비디오 프레임에 사각형과 함께 실시간으로 표시됩니다.

----


# 👤 과제2. Mediapipe를 활용한 얼굴 랜드마크 추출 및 시각화

이 프로젝트는 Google의 [Mediapipe](https://google.github.io/mediapipe/) 라이브러리를 사용하여 **실시간으로 얼굴의 468개 랜드마크를 추출하고 시각화**하는 프로그램을 구현한 예제입니다.

![image](https://github.com/user-attachments/assets/d92a80b3-4a32-475c-abe4-f2aa544838dc)

## 📌 프로젝트 설명

- Mediapipe의 FaceMesh 모듈을 사용해 웹캠으로부터 입력되는 영상에서 얼굴을 검출하고, 468개의 정밀한 랜드마크를 실시간으로 추출합니다.
- 추출된 랜드마크는 `OpenCV`의 그리기 유틸을 사용해 화면에 점과 선으로 시각화됩니다.
- ESC 키를 누르면 프로그램이 종료됩니다.

## ✅ 요구사항

- Mediapipe FaceMesh 모듈을 초기화하고 얼굴 랜드마크를 실시간으로 검출합니다.
- OpenCV를 통해 웹캠 영상 스트리밍 및 화면 출력 기능을 구현합니다.
- 검출된 468개 얼굴 랜드마크를 점 또는 연결선으로 시각화합니다.

## 🧠 사용 라이브러리

- [`mediapipe`](https://pypi.org/project/mediapipe/): 얼굴 랜드마크 검출
- [`opencv-python`](https://pypi.org/project/opencv-python/): 영상 처리 및 UI

## 🧪 코드 설명

### FaceMesh 모듈 초기화
```
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
```
### 랜드마크 스타일 정의
```
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1, color=(0, 255, 0))
```
### 웹캠 영상 스트리밍
```
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # 좌우 반전
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
```
### 얼굴 랜드마크 검출
```
results = face_mesh.process(rgb_frame)
```
### 얼굴에 랜드마크 표시
```
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=drawing_spec,
                connection_drawing_spec=drawing_spec
            )

    cv2.imshow('FaceMesh Landmarks', frame)
```

## 🖼️ 실행 결과

얼굴에 468개의 랜드마크가 점과 연결선 형태로 실시간으로 표시됩니다.

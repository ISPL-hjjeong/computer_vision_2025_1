# 🎯 SORT 알고리즘을 활용한 다중 객체 추적기 구현

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

import cv2
import numpy as np
from sort.sort import Sort


# YOLOv4 설정
config_path = "yolov4.cfg"
weights_path = "yolov4.weights"
names_path = "coco.names"

# 클래스 이름 불러오기
with open(names_path, 'r') as f:
    classes = f.read().strip().split('\n')

# DNN 모델 로드
net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# 출력 레이어 이름 추출
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# SORT 객체 초기화
tracker = Sort()

# 비디오 열기
cap = cv2.VideoCapture("slow_traffic_small.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    h, w = frame.shape[:2]

    # YOLOv4 객체 검출
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)

    # 검출된 객체 저장
    boxes = []
    confidences = []
    class_ids = []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.4:
                cx, cy, bw, bh = detection[0:4] * np.array([w, h, w, h])
                x = int(cx - bw / 2)
                y = int(cy - bh / 2)
                boxes.append([x, y, int(bw), int(bh)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Non-Maximum Suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.4, 0.3)
    detections = []

    if len(indices) > 0:
        for i in indices.flatten():
            x, y, bw, bh = boxes[i]
            detections.append([x, y, x + bw, y + bh, confidences[i]])

    # numpy array 형식으로 변환
    dets = np.array(detections)

    # SORT 추적
    tracks = tracker.update(dets)

    # 결과 시각화
    for track in tracks:
        x1, y1, x2, y2, track_id = track.astype(int)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'ID {int(track_id)}', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # 결과 출력
    cv2.imshow("SORT Tracker", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

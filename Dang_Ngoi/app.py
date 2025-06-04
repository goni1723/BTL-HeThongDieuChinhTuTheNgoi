import cv2
from ultralytics import YOLO
import mediapipe as mp
import os
from datetime import datetime

# Tạo thư mục lưu ảnh nếu chưa tồn tại
output_dir = "captures"
os.makedirs(output_dir, exist_ok=True)

# Load YOLOv8 model
model = YOLO('runs/detect/yolov8_dang_ngoi/weights/best.pt')

# Khởi tạo MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1,
                    enable_segmentation=False, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Mở webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Không thể mở webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Không thể đọc frame từ webcam.")
        break

    results = model(frame)
    annotated_frame = frame.copy()

    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            label = model.names[cls]

            if conf < 0.7:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            person_roi = frame[y1:y2, x1:x2]

            # Bỏ qua nếu vùng crop không hợp lệ
            if person_roi.size == 0:
                continue

            # Xử lý MediaPipe
            person_rgb = cv2.cvtColor(person_roi, cv2.COLOR_BGR2RGB)
            results_pose = pose.process(person_rgb)

            if results_pose.pose_landmarks:
                mp_drawing.draw_landmarks(person_roi, results_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            annotated_frame[y1:y2, x1:x2] = person_roi

            # Vẽ bounding box và label
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated_frame, f"{label} ({conf:.2f})", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Nếu là tư thế bad thì chụp và lưu ảnh
            if label.lower() == "bad":
                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                filename = f"{output_dir}/bad_{timestamp}.jpg"
                cv2.imwrite(filename, annotated_frame)
                print(f"📸 Đã lưu ảnh cảnh báo: {filename}")

    cv2.imshow('🎯 YOLOv8 + MediaPipe Pose', annotated_frame)
    if cv2.waitKey(1) & 0xFF == 27:  # Nhấn ESC để thoát
        break

cap.release()
cv2.destroyAllWindows()

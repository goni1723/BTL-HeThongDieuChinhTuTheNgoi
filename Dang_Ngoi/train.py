from ultralytics import YOLO

model = YOLO("yolov8s.pt")  # hoặc yolov8n.pt nếu bạn muốn nhẹ hơn

model.train(
    data="C:/dang_ngoi/data.yaml",  # Đường dẫn đến file yaml
    epochs=10,
    imgsz=640,
    batch=16,
    name="yolov8_dang_ngoi",
    workers=4
)
# model = YOLO("runs/detect/yolov8_dang_ngoi/weights/best.pt")
# results = model.predict("nhua_ne.jpg", save=True)

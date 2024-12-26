from ultralytics import YOLO

model = YOLO("yolov11.pt")

model.predict(source = "4.jpg", show=True, save=True, conf=0.4)
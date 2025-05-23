from ultralytics import YOLO

# Load a pretrained YOLO11n model
# model = YOLO("yolo11n.pt")

model = YOLO(r"runs\detect\train15\weights\best.pt")
# Run inference on 'bus.jpg' with arguments
model.predict("https://ultralytics.com/images/bus.jpg", save=True, imgsz=320, conf=0.5)
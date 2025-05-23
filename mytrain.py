from ultralytics import YOLO

model = YOLO("yolo11s.pt")

model.train(data = "ultralytics/cfg/datasets/mydata.yaml",
            epochs = 50,batch = 12,workers=0)
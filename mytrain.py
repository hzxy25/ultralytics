import multiprocessing

# Classification
from ultralytics import YOLO


# Object detection
# model = YOLO("yolo11s.pt")
# model.train(data = "ultralytics/cfg/datasets/mydata.yaml",
#             epochs = 50,batch = 12,workers=0)
def main():

    # Load a model
    model = YOLO("yolo11n-cls.pt")  # load a pretrained model (recommended for training)
    # Train the model
    results = model.train(
        data="E:\下载\Compressed\WebFG-496-yolo",
        epochs=40,
        imgsz=640,
        workers=0)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
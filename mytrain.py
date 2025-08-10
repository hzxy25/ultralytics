import multiprocessing

# Classification
from ultralytics import YOLO


# Object detection
# model = YOLO("yolo11s.pt")
# model.train(data = "ultralytics/cfg/datasets/mydata.yaml",
#             epochs = 50,batch = 12,workers=0)
def main():
    # Load a model
    # model = YOLO("yolo11s-cls.yaml",task='classify')  # load a pretrained model (recommended for training)
    model = YOLO("cgyolo11s-cls.yaml",task='classify')  # load a pretrained model (recommended for training)
    # Train the model
    results = model.train(
        # data="/root/autodl-tmp/data/webfg400",
        # data=r"E:\tool\Data\WebFG-496-yolo",
        data=r"E:\tool\Data\testClassification",

        epochs=200,
        imgsz=384,
        save_period=10,
        batch=4,
        workers=0
    )


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()

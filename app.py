from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
from ultralytics import YOLO
import cv2
import numpy as np
import io

app = FastAPI()
model = YOLO("/ultralytics/models/best.pt")  # 模型路径


@app.post("/predict")
async def predict(file: UploadFile = File(...), return_image: bool = True):
    # 1. 读取并解码图片
    image_bytes = await file.read()
    image = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(image, cv2.IMREAD_COLOR)

    # 2. 模型推理
    results = model(img)

    # 3. 生成带标注的图片
    annotated_img = results[0].plot()  # 自动绘制检测框
    # annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
    # 4. 将图片转换为字节流
    _, encoded_img = cv2.imencode('.jpg', annotated_img)
    img_bytes = encoded_img.tobytes()

    # 5. 根据参数决定返回类型
    if return_image:
        # 返回标注后的图片
        return StreamingResponse(io.BytesIO(img_bytes), media_type="image/jpeg")
    else:
        # 返回检测结果 + 图片（可选）
        return {
            "detections": results[0].boxes.data.cpu().numpy().tolist(),
            "classes": model.names,
            "image": StreamingResponse(io.BytesIO(img_bytes), media_type="image/jpeg")
        }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=5000)
import os
import csv
from ultralytics import YOLO


def predict_and_save():
    # 加载训练好的模型
    model = YOLO("best.pt")  # 替换为你的模型路径

    # 待预测图片所在目录（根据实际情况修改）
    image_dir = r"E:\tool\Data\test_A"  # 测试图片目录

    # 预测结果保存路径
    output_csv = "pred_results_web400.csv"

    # 获取目录下所有图片文件（排序确保顺序一致）
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
    image_files = sorted(
        [f for f in os.listdir(image_dir) if os.path.splitext(f)[1].lower() in image_extensions]
    )

    # 批量预测整个目录
    results = model(image_dir)  # verbose=False关闭冗余输出

    # 检查结果数量是否匹配
    if len(results) != len(image_files):
        print(f"警告: 预测结果数量({len(results)})与图片数量({len(image_files)})不匹配")

    # 写入CSV文件
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['image_name', 'class'])  # 可选：添加表头

        # 遍历每张图片的结果
        for img_file, result in zip(image_files, results):
            # 获取最可能的类别（取置信度最高的）
            if hasattr(result, 'probs') and result.probs is not None:
                cls = result.probs.top1
            else:
                # 处理没有预测结果的情况（例如输出默认值）
                cls = -1

            # 格式化类别为四位数字（不满四位前面补0）
            cls_str = f"{cls:04d}"
            writer.writerow([img_file, cls_str])


if __name__ == "__main__":
    predict_and_save()
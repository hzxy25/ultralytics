import os
import shutil
import random
from pathlib import Path


def split_dataset(train_dir, val_dir, val_ratio=0.2):
    """
    从训练集分割出验证集

    参数:
    train_dir: 原始训练集目录路径 (包含按类别分类的子文件夹)
    val_dir: 要创建的验证集目录路径
    val_ratio: 验证集比例 (默认0.2即20%)
    """
    # 创建验证集目录
    Path(val_dir).mkdir(parents=True, exist_ok=True)

    # 遍历每个类别文件夹
    for class_name in os.listdir(train_dir):
        class_train_dir = os.path.join(train_dir, class_name)
        class_val_dir = os.path.join(val_dir, class_name)

        # 跳过非目录文件
        if not os.path.isdir(class_train_dir):
            continue

        # 创建验证集的类别目录
        Path(class_val_dir).mkdir(parents=True, exist_ok=True)

        # 获取当前类别的所有图片文件
        all_images = [f for f in os.listdir(class_train_dir)
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]

        # 计算需要移动到验证集的数量
        num_val = int(len(all_images) * val_ratio)

        # 随机选择验证集图片
        val_images = random.sample(all_images, num_val)

        # 移动文件到验证集
        for img in val_images:
            src = os.path.join(class_train_dir, img)
            dst = os.path.join(class_val_dir, img)
            shutil.move(src, dst)

        print(f"类别 '{class_name}': 移动了 {len(val_images)} 张图片到验证集")


if __name__ == "__main__":
    # 配置路径 - 根据实际情况修改这些路径
    original_train_dir = r"E:\下载\Compressed\WebFG-496-yolo\train"  # 原始训练集目录
    new_train_dir = r"E:\下载\Compressed\WebFG-496-yolo\train_new"  # 新的训练集目录
    val_dir = r"E:\下载\Compressed\WebFG-496-yolo\val"  # 验证集目录

    # 第一步：复制原始训练集（避免修改原始数据）
    print("正在复制原始训练集...")
    if os.path.exists(new_train_dir):
        shutil.rmtree(new_train_dir)
    shutil.copytree(original_train_dir, new_train_dir)

    # 第二步：从复制的训练集中分割验证集
    print("\n正在分割验证集...")
    split_dataset(new_train_dir, val_dir, val_ratio=0.2)

    print("\n操作完成！")
    print(f"新训练集: {new_train_dir}")
    print(f"验证集: {val_dir}")
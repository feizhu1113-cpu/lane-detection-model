import cv2
import numpy as np


def apply_roi(image, keep_ratio=0.6):
    """
    保留图片底部区域（路面部分）
    keep_ratio: 保留图片高度的比例（0.6表示保留底部60%）
    """
    height, width = image.shape[:2]
    start_y = int(height * (1 - keep_ratio))
    roi_image = image[start_y:height, :]
    return roi_image


def preprocess_image(image, target_size=(224, 224)):
    """
    完整预处理：ROI + 缩放 + 归一化 + 转Tensor格式
    """
    # 1. 应用ROI裁剪
    roi = apply_roi(image)

    # 2. 缩放到模型输入大小
    resized = cv2.resize(roi, target_size)

    # 3. 归一化到 [0, 1]
    normalized = resized / 255.0

    # 4. 转换维度：HWC → CHW（PyTorch要求的格式）
    tensor = np.transpose(normalized, (2, 0, 1))

    return tensor.astype(np.float32)


# 测试代码
if __name__ == "__main__":
    # 创建一张假图片（720x1280，3通道）
    fake_image = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
    result = preprocess_image(fake_image)
    print(f"输入假图尺寸: {fake_image.shape}")
    print(f"预处理后形状: {result.shape}")  # 应该是 (3, 224, 224)
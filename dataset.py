import json
import os
import cv2
import torch
from torch.utils.data import Dataset
from utils.roi import preprocess_image
import numpy as np


class LaneDataset(Dataset):
    """TuSimple车道数据集，用于车道内/车道外二分类"""

    def __init__(self, data_root, is_train=True, transform=None):
        """
        Args:
            data_root: 数据集根目录，比如 "D:/Intelligent Driving/tusimple"
            is_train: True=训练集，False=验证集
            transform: 数据增强（可选）
        """
        self.data_root = data_root
        self.is_train = is_train
        self.transform = transform

        # 加载标注文件
        label_file = os.path.join(data_root, "train_label.json")
        self.annotations = []

        with open(label_file, 'r') as f:
            for line in f:
                self.annotations.append(json.loads(line))

        # 划分训练集和验证集（80%训练，20%验证）
        split_idx = int(len(self.annotations) * 0.8)
        if is_train:
            self.annotations = self.annotations[:split_idx]
        else:
            self.annotations = self.annotations[split_idx:]

        print(f"加载 {'训练' if is_train else '验证'} 集: {len(self.annotations)} 张图片")

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        # 1. 获取图片路径和标注
        ann = self.annotations[idx]
        img_path = os.path.join(self.data_root, ann['raw_file'])

        # 2. 读取图片
        image = cv2.imread(img_path)
        if image is None:
            print(f"警告: 无法读取图片 {img_path}")
            # 返回一个假数据避免崩溃
            image = np.zeros((720, 1280, 3), dtype=np.uint8)

        # 3. 预处理：ROI + 缩放 + 归一化
        processed = preprocess_image(image)

        # 4. 生成标签（车道内/车道外）
        label = self.get_label(ann, image.shape)

        # 5. 转换为 PyTorch Tensor
        processed = torch.from_numpy(processed).float()
        label = torch.tensor(label, dtype=torch.long)

        # 6. 数据增强（如果启用）
        if self.transform and self.is_train:
            processed = self.transform(processed)

        return processed, label

    def get_label(self, ann, img_shape):
        """
        根据标注判断车辆是否在车道内
        规则：假设车辆中心在图片底部中心 (x = 图片宽度的一半)
        """
        h, w = img_shape[:2]
        car_x = w // 2  # 车辆中心的 x 坐标

        lanes = ann['lanes']
        h_samples = ann['h_samples']

        # 取离车辆最近的有效 y 位置（图片底部附近）
        # 选择 y 坐标最大的几个点来判段车辆位置
        valid_x_left = None
        valid_x_right = None

        # 通常第 0 条是左车道线，第 1 条是右车道线
        if len(lanes) >= 2:
            left_lane = lanes[0]
            right_lane = lanes[1]

            # 找到非空的点（坐标 ≠ -2）
            for i, (lx, rx) in enumerate(zip(left_lane, right_lane)):
                if lx != -2 and rx != -2:
                    valid_x_left = lx
                    valid_x_right = rx

        # 如果找不到有效的车道线，默认认为在车道内
        if valid_x_left is None or valid_x_right is None:
            return 1

        # 判断：车辆中心是否在左右车道线之间
        if valid_x_left <= car_x <= valid_x_right:
            return 1  # 车道内
        else:
            return 0  # 车道外
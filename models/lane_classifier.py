import torch
import torch.nn as nn
from torchvision import models


class LaneClassifier(nn.Module):
    """车道内/车道外二分类模型"""

    def __init__(self):
        super(LaneClassifier, self).__init__()
        # 加载预训练的MobileNetV2
        self.backbone = models.mobilenet_v2(pretrained=True)
        # 修改最后一层：原本输出1000类，改成输出2类
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier[1] = nn.Linear(in_features, 2)

    def forward(self, x):
        return self.backbone(x)


# 测试代码：右键 → Run 可以直接运行
if __name__ == "__main__":
    model = LaneClassifier()
    print("✅ 模型创建成功！")
    # 用随机数据测试前向传播
    dummy_input = torch.randn(1, 3, 224, 224)
    output = model(dummy_input)
    print(f"输入形状: {dummy_input.shape}")
    print(f"输出形状: {output.shape}")
    print(f"输出内容（未训练，随机初始化）: {output}")
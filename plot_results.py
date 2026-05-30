import matplotlib.pyplot as plt
import numpy as np

# 你训练中的原始数据（从输出结果中提取）
epochs = list(range(1, 11))
train_loss = [0.0696, 0.0330, 0.0336, 0.0130, 0.0069, 0.0030, 0.0073, 0.0122, 0.0129, 0.0066]
train_acc = [97.08, 99.15, 99.01, 99.73, 99.91, 99.96, 99.87, 99.64, 99.87, 99.87]
val_acc = [99.46, 97.85, 97.67, 98.92, 99.10, 99.64, 99.64, 99.46, 99.64, 99.64]

# 1. 绘制 Loss 曲线
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(epochs, train_loss, 'b-o', linewidth=2, markersize=6)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Train Loss', fontsize=12)
plt.title('Training Loss Curve', fontsize=14)
plt.grid(True, alpha=0.3)

# 2. 绘制准确率曲线
plt.subplot(1, 2, 2)
plt.plot(epochs, train_acc, 'g-o', linewidth=2, markersize=6, label='Train Acc')
plt.plot(epochs, val_acc, 'r-s', linewidth=2, markersize=6, label='Val Acc')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Accuracy (%)', fontsize=12)
plt.title('Accuracy Curve', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_curves.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ 曲线图已保存为 training_curves.png")
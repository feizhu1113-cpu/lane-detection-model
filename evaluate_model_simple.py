import torch
import numpy as np
import matplotlib.pyplot as plt
from dataset import LaneDataset
from models.lane_classifier import LaneClassifier

# 1. 加载模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LaneClassifier().to(device)
model.load_state_dict(torch.load("lane_classifier.pth", map_location=device))
model.eval()

# 2. 加载验证集
val_dataset = LaneDataset("D:/Intelligent Driving/tusimple", is_train=False)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)

# 3. 收集所有预测结果
all_preds = []
all_labels = []
bad_cases = []

print("正在评估模型...")
with torch.no_grad():
    for batch_idx, (images, labels) in enumerate(val_loader):
        images = images.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        # 收集错误样本
        for i in range(len(labels)):
            if preds[i] != labels[i]:
                bad_cases.append((images[i].cpu(), labels[i].item(), preds[i].item()))

# 4. 转换为numpy数组
all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

# 5. 手动计算混淆矩阵
cm = np.zeros((2, 2), dtype=int)
for true, pred in zip(all_labels, all_preds):
    cm[true][pred] += 1

print("\n=== 混淆矩阵 ===")
print(f"           预测车道外  预测车道内")
print(f"真实车道外:     {cm[0][0]:6d}      {cm[0][1]:6d}")
print(f"真实车道内:     {cm[1][0]:6d}      {cm[1][1]:6d}")

# 6. 绘制混淆矩阵图
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
ax.figure.colorbar(im, ax=ax)
ax.set(xticks=np.arange(2), yticks=np.arange(2),
       xticklabels=['车道外', '车道内'],
       yticklabels=['车道外', '车道内'],
       title='Confusion Matrix',
       ylabel='True Label',
       xlabel='Predicted Label')

# 在格子中显示数字
for i in range(2):
    for j in range(2):
        ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                color='white' if cm[i, j] > cm.max() / 2 else 'black', fontsize=14)

plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ 混淆矩阵已保存为 confusion_matrix.png")

# 7. 可视化 Bad Cases
print(f"\n发现 {len(bad_cases)} 个错误样本")

if len(bad_cases) > 0:
    n_show = min(5, len(bad_cases))
    fig, axes = plt.subplots(1, n_show, figsize=(15, 3))
    if n_show == 1:
        axes = [axes]

    for i, (img, true_label, pred_label) in enumerate(bad_cases[:n_show]):
        # 转换图片格式
        img_np = img.numpy().transpose(1, 2, 0)
        img_np = (img_np * 255).astype(np.uint8)

        axes[i].imshow(img_np)
        true_text = "车道内" if true_label == 1 else "车道外"
        pred_text = "车道内" if pred_label == 1 else "车道外"
        axes[i].set_title(f'True: {true_text}\nPred: {pred_text}', fontsize=10)
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig('bad_cases.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("✅ Bad Case 可视化已保存为 bad_cases.png")
else:
    print("🎉 验证集上没有发现错误样本！")

# 8. 统计信息
total = len(all_labels)
correct = np.sum(all_preds == all_labels)
print(f"\n=== 最终评估结果 ===")
print(f"总样本数: {total}")
print(f"正确数: {correct}")
print(f"错误数: {total - correct}")
print(f"准确率: {100 * correct / total:.2f}%")
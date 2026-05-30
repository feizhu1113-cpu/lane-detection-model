import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from dataset import LaneDataset
from models.lane_classifier import LaneClassifier
import os


def train():
    # 1. 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 2. 加载数据
    train_dataset = LaneDataset("D:/Intelligent Driving/tusimple", is_train=True)
    val_dataset = LaneDataset("D:/Intelligent Driving/tusimple", is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # 3. 创建模型
    model = LaneClassifier().to(device)

    # 4. 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 5. TensorBoard 记录
    writer = SummaryWriter("runs/lane_classifier")

    # 6. 训练循环
    num_epochs = 10
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, labels)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 统计
            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        train_acc = 100 * train_correct / train_total
        train_loss_avg = train_loss / len(train_loader)

        # 验证阶段
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_acc = 100 * val_correct / val_total

        # 记录到 TensorBoard
        writer.add_scalar('Loss/train', train_loss_avg, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)

        print(f"Epoch [{epoch + 1}/{num_epochs}] "
              f"Train Loss: {train_loss_avg:.4f}, Train Acc: {train_acc:.2f}%, "
              f"Val Acc: {val_acc:.2f}%")

    # 7. 保存模型
    torch.save(model.state_dict(), "lane_classifier.pth")
    print("✅ 模型已保存到 lane_classifier.pth")
    writer.close()


if __name__ == "__main__":
    train()
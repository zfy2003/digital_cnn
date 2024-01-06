import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset, random_split
import matplotlib.pyplot as plt
import pandas as pd


# 定义网络
class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.25)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)  # 移除额外的stride参数
        self.dropout2 = nn.Dropout(0.25)
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(64)
        self.dropout3 = nn.Dropout(0.25)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 7 * 7, 256)
        self.bn6 = nn.BatchNorm1d(256)
        self.dropout4 = nn.Dropout(0.25)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        x = self.dropout1(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool2(x)
        x = self.dropout2(x)
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.dropout3(x)
        x = self.flatten(x)
        x = F.relu(self.bn6(self.fc1(x)))
        x = self.dropout4(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


# 定义训练函数
def train(epoch):
    loss_train = []
    total, correct = 0, 0
    for batch_id, data in enumerate(train_dataloader, 0):
        # 计算梯度并更新参数
        inputs, target = data
        inputs, target = inputs.to(device), target.to(device)
        optimer.zero_grad()
        outputs = model(inputs)
        # 计算训练时的准确率
        _, predicted = torch.max(outputs.data, dim=1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
        loss = criterion(outputs, target)
        loss.mean().backward()
        optimer.step()
        loss_train.append(loss.mean().item())
    loss = sum(loss_train) / len(loss_train)
    acc = correct / total
    print('epoch:%d loss:%.3f train_acc:%f ' % (epoch + 1, loss, acc))
    return loss, acc


# 定义测试函数
def test():
    correct = 0
    total = 0
    with torch.no_grad():
        for data in val_dataloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            # 模型预测
            outputs = model(images)
            # 将置信度最高的为当前预测值
            _, predicted = torch.max(outputs.data, dim=1)
            total += labels.size(0)
            # 获得预测准确的样本数
            correct += (predicted == labels).sum().item()
    acc = correct / total
    print('test_acc:%f ' % acc)
    return acc


# 绘制准确率和损失图像
def paint(title, train_l, train_a, test_a, e):
    xl = [i + 1 for i in range(e)]
    # 创建包含两个子图的图形
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    # 第一个子图
    axes[0].plot(xl, train_l, color='r', linestyle='-', label='train_loss')
    axes[0].plot(xl, train_a, color='g', linestyle=':', label='train_acc')
    axes[0].set_xlabel('epoch')
    axes[0].set_ylabel('loss/acc')
    axes[0].set_title('train loss/acc')
    # 第二个子图
    axes[1].plot(xl, train_a, color='g', linestyle=':', label='train_acc')
    axes[1].plot(xl, test_a, color='b', linestyle='--', label='test_acc')
    axes[1].set_xlabel('epoch')
    axes[1].set_ylabel('acc')
    axes[1].set_title('test acc')
    # 为整个图形添加大标题
    fig.suptitle(title + ' Training and Testing Result', fontsize=16)
    # 调整布局，防止重叠
    plt.tight_layout()
    # 显示图例
    axes[0].legend()
    axes[1].legend()
    # 显示图形
    plt.show()


if __name__ == '__main__':
    # 加载数据集
    total_data = pd.read_csv("./Datasets/digit-recognizer/train.csv")
    # 提取标签和图像数据
    total_labels = total_data['label'].values
    total_images = total_data.drop('label', axis=1).values
    # 转换为PyTorch的Tensor
    total_images = torch.tensor(total_images, dtype=torch.float32)
    total_labels = torch.tensor(total_labels, dtype=torch.long)
    # 转换为适当的形状
    total_images = total_images.view(-1, 1, 28, 28)  # MNIST图像大小是28x28
    # 创建图像转换
    transform_train = transforms.Compose([
        transforms.RandomRotation(10),  # 随机旋转最多±10度
        transforms.RandomAffine(0, translate=(0.1, 0.1)),  # 随机水平和垂直平移最多10%的图像大小
        transforms.RandomAffine(0, scale=(0.9, 1.1)),  # 随机缩放图像大小为90%-110%
    ])
    batch_size = 64
    # 创建数据集和数据加载器
    total_dataset = TensorDataset(total_images, total_labels)
    train_size = int(0.8 * len(total_dataset))
    val_size = int(0.2 * len(total_dataset))
    # 利用 random_split 函数划分数据集
    train_dataset, val_dataset = random_split(total_dataset, [train_size, val_size])
    # 对训练集进行数据增强
    # train_dataset = [(transform_train(img), label) for img, label in train_dataset]
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 初始化模型
    model = MyNet().to(device)
    title = 'MyNet'

    epochs = 3#20
    lr = 0.001
    # 优化算法——随机梯度下降
    optimer = torch.optim.Adam(model.parameters(), lr=lr)
    # 损失函数——交叉熵
    criterion = nn.CrossEntropyLoss()
    loss = []
    train_acc = []
    test_acc = []
    # 训练+测试
    for epoch in range(epochs):
        los, train_ac = train(epoch)
        test_ac = test()
        loss.append(los)
        train_acc.append(train_ac)
        test_acc.append(test_ac)

    paint(title, loss, train_acc, test_acc, epochs)

    # 读取测试数据集并预测
    test_data = pd.read_csv("./Datasets/digit-recognizer/test.csv")
    sub = pd.read_csv('./Datasets/digit-recognizer/sample_submission.csv')
    test_images = test_data.values
    test_images = torch.tensor(test_images, dtype=torch.float32)
    test_images = test_images.view(-1, 1, 28, 28)

    test_set = TensorDataset(test_images)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    pred_test = []
    for data in test_loader:
        image = data[0]
        output = model(image.to(device))
        pred_t = torch.argmax(output, dim=1).tolist()
        pred_test += pred_t

    sub['Label'] = pred_test
    sub.to_csv('submission_.csv', index=False)
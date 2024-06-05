import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet50  # 修改为导入ResNet-50
import matplotlib.pyplot as plt
import pandas as pd
from cc_attention import CrissCrossAttention  # 引入CrissCrossAttention模块

# 确保结果文件夹存在
if not os.path.exists('resnet50_ccnet_result'):
    os.makedirs('resnet50_ccnet_result')

# 1. 数据预处理
transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
])

# 2. 加载CIFAR-100数据集
trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

# 3. 定义ResNet-50模型，并添加CrissCrossAttention模块
class ResNet50_CIFAR100_CCA(nn.Module):
    def __init__(self):
        super(ResNet50_CIFAR100_CCA, self).__init__()
        self.model = resnet50(weights=None)  # 使用ResNet-50，不使用预训练权重
        self.model.fc = nn.Linear(self.model.fc.in_features, 100)
        self.cca = CrissCrossAttention(self.model.layer4[-1].conv3.out_channels)  # 修改为适配ResNet-50

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        x = self.cca(x)  # 在layer4之后应用CrissCrossAttention模块

        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.model.fc(x)
        return x

net = ResNet50_CIFAR100_CCA()

# 使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net.to(device)

# 4. 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

# 记录训练和测试的损失与准确率
results = {
    "epoch": [],
    "train_loss": [],
    "train_accuracy": [],
    "test_loss": [],
    "test_accuracy": []
}

# 5. 训练函数
def train(epoch):
    net.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(inputs)}/{len(trainloader.dataset)} ({100. * batch_idx / len(trainloader):.0f}%)]\tLoss: {loss.item():.6f}')

    train_loss = running_loss / len(trainloader)
    train_accuracy = 100. * correct / total
    results["epoch"].append(epoch)
    results["train_loss"].append(train_loss)
    results["train_accuracy"].append(train_accuracy)
    print(f'Train Epoch: {epoch}\tLoss: {train_loss:.6f}\tAccuracy: {train_accuracy:.2f}%')

# 6. 测试函数
def test(epoch):
    net.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    test_loss_avg = test_loss / len(testloader)
    test_accuracy = 100. * correct / total
    results["test_loss"].append(test_loss_avg)
    results["test_accuracy"].append(test_accuracy)
    print(f'Test Epoch: {epoch}\tLoss: {test_loss_avg:.6f}\tAccuracy: {test_accuracy:.2f}%')

if __name__ == '__main__':
    # 7. 训练和测试模型
    num_epochs = 100
    best_accuracy = 0
    for epoch in range(1, num_epochs + 1):
        train(epoch)
        test(epoch)
        scheduler.step()

        # 保存最佳模型
        if results["test_accuracy"][-1] > best_accuracy:
            best_accuracy = results["test_accuracy"][-1]
            torch.save(net.state_dict(), 'resnet50_ccnet_result/best_model.pth')

    # 8. 绘制损失和准确率曲线
    epochs = range(1, num_epochs + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, results["train_loss"], label='Training Loss')
    plt.plot(epochs, results["test_loss"], label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, results["train_accuracy"], label='Training Accuracy')
    plt.plot(epochs, results["test_accuracy"], label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy Curve')
    plt.legend()

    plt.savefig('resnet50_ccnet_result/loss_accuracy_curves.png')
    plt.show()

    # 9. 保存结果到Excel文件
    df = pd.DataFrame(results)
    df.to_excel('resnet50_ccnet_result/training_results.xlsx', index=False)

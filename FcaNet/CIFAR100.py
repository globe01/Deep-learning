import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from fcanet import fcanet50
import csv


# 数据集准备
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
])

trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=0)

# 模型定义
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# 定义模型、损失函数和优化器
net = fcanet50(num_classes=100)  # CIFAR-100有100个类别
net = net.to(device)

# 损失函数和优化器
criterion = nn.CrossEntropyLoss()

# weight_decay   1e-3
optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)  # 每30个epoch学习率减少10倍

# CSV 文件名
csv_filename = 'training_results.csv'

# 初始化 CSV 文件，写入标题行
with open(csv_filename, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Epoch', 'Train Loss', 'Train Accuracy', 'Test Loss', 'Test Accuracy'])

# 保存数据到 CSV 的函数
def save_to_csv(filename, data):
    with open(filename, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(data)

# 训练函数
def train(epoch):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    train_accuracy = 100. * correct / total
    train_loss_avg = train_loss / (batch_idx + 1)
    print(f'Epoch: {epoch}, Loss: {train_loss_avg}, Accuracy: {train_accuracy}')
    return train_loss_avg, train_accuracy

# 测试函数
def test(epoch):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    test_accuracy = 100. * correct / total
    test_loss_avg = test_loss / (batch_idx + 1)
    print(f'Epoch: {epoch}, Test Loss: {test_loss_avg}, Test Accuracy: {test_accuracy}')
    return test_loss_avg, test_accuracy

# 训练和测试
for epoch in range(100):
    train_loss, train_accuracy = train(epoch)
    test_loss, test_accuracy = test(epoch)
    scheduler.step()
    # 保存结果到 CSV
    save_to_csv(csv_filename, [epoch, train_loss, train_accuracy, test_loss, test_accuracy])

print('Finished Training')
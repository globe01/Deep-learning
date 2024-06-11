# ============================================================
# Imports
# ============================================================
import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd


# ============================================================
# Criss-Cross Attention Module
# ============================================================
class CrissCrossAttention(nn.Module):
    """ Criss-Cross Attention Module"""

    def __init__(self, in_dim):
        super(CrissCrossAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.softmax = nn.Softmax(dim=3)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        m_batchsize, _, height, width = x.size()
        proj_query = self.query_conv(x)
        proj_query_H = proj_query.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height).permute(0, 2, 1)
        proj_query_W = proj_query.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width).permute(0, 2, 1)
        proj_key = self.key_conv(x)
        proj_key_H = proj_key.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height)
        proj_key_W = proj_key.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width)
        proj_value = self.value_conv(x)
        proj_value_H = proj_value.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height)
        proj_value_W = proj_value.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width)
        energy_H = (torch.bmm(proj_query_H, proj_key_H) + self.INF(m_batchsize, height, width)).view(m_batchsize, width, height, height).permute(0, 2, 1, 3)
        energy_W = torch.bmm(proj_query_W, proj_key_W).view(m_batchsize, height, width, width)
        concate = self.softmax(torch.cat([energy_H, energy_W], 3))

        att_H = concate[:, :, :, 0:height].permute(0, 2, 1, 3).contiguous().view(m_batchsize * width, height, height)
        att_W = concate[:, :, :, height:height + width].contiguous().view(m_batchsize * height, width, width)
        out_H = torch.bmm(proj_value_H, att_H.permute(0, 2, 1)).view(m_batchsize, width, -1, height).permute(0, 2, 3, 1)
        out_W = torch.bmm(proj_value_W, att_W.permute(0, 2, 1)).view(m_batchsize, height, -1, width).permute(0, 2, 1, 3)

        return self.gamma * (out_H + out_W) + x

    def INF(self, B, H, W):
        return -torch.diag(torch.tensor(float("inf")).cuda().repeat(H), 0).unsqueeze(0).repeat(B * W, 1, 1)

# ============================================================
# Basic and Bottleneck Blocks for ResNet
# ============================================================
# Basic Block for ResNet
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out


# Bottleneck Block for ResNet
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = torch.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out



# ============================================================
# RCCAModule，调用CrissCrossAttention
# ============================================================
class RCCAModule(nn.Module):
    def __init__(self, in_channels, out_channels, num_classes):
        super(RCCAModule, self).__init__()
        inter_channels = in_channels // 4
        self.conva = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(inter_channels),
                                   nn.ReLU(inplace=True))
        self.cca = CrissCrossAttention(inter_channels)
        self.convb = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(inter_channels),
                                   nn.ReLU(inplace=True))

        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels + inter_channels, out_channels, kernel_size=3, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            # nn.Conv2d(out_channels, num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        )

    def forward(self, x, recurrence=1):
        output = self.conva(x)
        for _ in range(recurrence):
            output = self.cca(output)
        output = self.convb(output)
        output = self.bottleneck(torch.cat([x, output], 1))
        return output


# ============================================================
# 尝试优化，加一个动态注意力模块
# ============================================================


class DynamicAttention(nn.Module):
    def __init__(self, in_channels):
        super(DynamicAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, C, width, height = x.size()
        proj_query = self.query_conv(x).view(batch_size, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(batch_size, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(batch_size, -1, width * height)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1)).view(batch_size, C, width, height)
        out = self.gamma * out + x
        return out



# ============================================================
# ResNet加上注意力机制cca
# ============================================================
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        #self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        # 去除掉第一个最大池化层，本来CIFAR100数据集很小，只有32*32，不用这个池化层
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.attention1 = DynamicAttention(64 * block.expansion)  # 添加动态注意力模块

        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.attention2 = DynamicAttention(128 * block.expansion)  # 添加动态注意力模块

        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.attention3 = DynamicAttention(256 * block.expansion)  # 添加动态注意力模块

        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=1) # layer4的stride改为1
        self.attention4 = DynamicAttention(512 * block.expansion)  # 添加动态注意力模块


        # 这样的话，最后在平均池化之前的feature map大小为8*8，如果直接用原来的resnet50是1*1，学不到啥东西

        # self.cca = CrissCrossAttention(512 * block.expansion)  # 添加Criss-Cross注意力机制
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(512 * block.expansion, num_classes)

        #---加注意力机制---#
        self.rcca = RCCAModule(512 * block.expansion, 512, 10)  # 添加RCCAModule来使用Criss-Cross注意力机制

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)  # 修改全连接层输入通道数

        #---不加注意力机制---#
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(512 * block.expansion, num_classes)


    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        # 去除掉第一个最大池化层，本来CIFAR100数据集很小，只有32*32，不用这个池化层
        # x = torch.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        x = self.layer1(x)
        x = self.attention1(x)  # 应用动态注意力模块
        x = self.layer2(x)
        x = self.attention2(x)  # 应用动态注意力模块
        x = self.layer3(x)
        x = self.attention3(x)  # 应用动态注意力模块
        x = self.layer4(x)
        x = self.attention4(x)  # 应用动态注意力模块



        # x = self.cca(x) # 注意力机制

        x = self.rcca(x)  # RCCAModule来使用注意力机制

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def ResNet50(num_classes=100):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes)



# ============================================================
# 训练和验证
# ============================================================
# 训练函数
def train(model, trainloader, criterion, optimizer, epoch, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        if i % 100 == 99:  # 每100个batch打印一次
            print(
                f'Train Epoch: {epoch + 1} [{i * len(inputs)}/{len(trainloader.dataset)} ({100. * i / len(trainloader):.0f}%)]\tLoss: {running_loss / 100:.6f}')
            running_loss = 0.0

    train_loss = running_loss / len(trainloader)
    train_accuracy = 100. * correct / total
    print(f'Train Epoch: {epoch + 1}\tLoss: {train_loss:.6f}\tAccuracy: {train_accuracy:.2f}%')
    return train_loss, train_accuracy


# 测试函数
def test(model, testloader, criterion, epoch, device):
    model.eval()
    correct = 0
    total = 0
    test_loss = 0.0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_loss /= len(testloader)
    test_accuracy = 100. * correct / total
    print(f'Test Epoch: {epoch + 1}\tLoss: {test_loss:.6f}\tAccuracy: {test_accuracy:.2f}%')
    return test_loss, test_accuracy



# ============================================================
# 主函数
# ============================================================
if __name__ == '__main__':
    # 创建结果保存文件夹
    os.makedirs('optimize_result', exist_ok=True)

    mean = [0.5070751592371323, 0.48654887331495095, 0.4409178433670343]
    std = [0.2673342858792401, 0.2564384629170883, 0.27615047132568404]

    # 定义数据增强和预处理
    transform = transforms.Compose([
        transforms.RandomCrop(size=(32, 32), padding=4),
        transforms.RandomHorizontalFlip(p=0.5), # 增加p参数，使得随机水平翻转的概率为0.5
        transforms.AutoAugment(policy=torchvision.transforms.AutoAugmentPolicy.CIFAR10),  # 新增自动数据增强
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        transforms.RandomErasing(p=0.25)  # 新增随机擦除，p=0.25
    ])

    # 加载CIFAR-100数据集
    trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)

    classes = trainset.classes

    # 创建模型实例
    num_classes = 100
    model = ResNet50(num_classes)

    # 选择设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    # SGD优化器，学习率为0.1，动量为0.9，权重衰减为5e-4
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

    # 学习率调度器
    # StepLR适用于大多数基础场景，每隔step_size个epoch，学习率乘以gamma
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    # MultiStepLR适用于对训练过程有一定了解并希望在特定点进行学习率调整的场景

    # 训练200个epoch，前5个epoch学习率线性预热到0.1
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.2)

    # 训练100个epoch，前5个epoch学习率线性预热到0.1
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60], gamma=0.2)

    # 换成CosineAnnealingWarmRestarts调度器，T_0为初始周期，T_mult为周期倍数，eta_min为最小学习率
    # CosineAnnealingWarmRestarts适用于需要长时间训练且可能存在多个局部最优的场景，通过周期性重启学习率来避免模型陷入局部最优
    # scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=2, eta_min=0.0001)

    # 加一个学习率预热
    def linear_warmup(step, warmup_steps, base_lr):
        return base_lr * step / warmup_steps


    warmup_epochs = 5
    base_lr = 0.1
    min_lr = 1e-6
    warmup_steps = warmup_epochs * len(trainloader)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=2, eta_min=min_lr)



    # 记录每个epoch的loss和accuracy
    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []

    # 训练和评估模型
    for epoch in range(230):  # 训练230个epoch

        #---学习率预热---#
        if epoch < warmup_epochs:
            lr = linear_warmup(epoch * len(trainloader), warmup_steps, base_lr)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        else:
            scheduler.step()

        train_loss, train_accuracy = train(model, trainloader, criterion, optimizer, epoch, device)
        test_loss, test_accuracy = test(model, testloader, criterion, epoch, device)

        #---不预热---#
        # train_loss, train_accuracy = train(model, trainloader, criterion, optimizer, epoch, device)
        # test_loss, test_accuracy = test(model, testloader, criterion, epoch, device)
        # scheduler.step()

        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)

    # 保存训练和测试结果到Excel文件
    results_df = pd.DataFrame({
        'Epoch': range(1, 231),
        'Train Loss': train_losses,
        'Train Accuracy': train_accuracies,
        'Test Loss': test_losses,
        'Test Accuracy': test_accuracies
    })
    #results_df.to_excel('optimize_result/training_results.xlsx', index=False)
    results_df.to_csv('optimize_result/training_results.csv', index=False)

    # 绘制Loss和Accuracy曲线并保存
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, 231), train_losses, label='Train Loss')
    plt.plot(range(1, 231), test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss vs. Epochs')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(1, 231), train_accuracies, label='Train Accuracy')
    plt.plot(range(1, 231), test_accuracies, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs. Epochs')
    plt.legend()

    plt.tight_layout()
    plt.savefig('optimize_result/loss_accuracy_plot.png')
    plt.show()

# import torch
# import torchvision
# import torchvision.transforms as transforms
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader
#
# from networks import ccnet
#
# # 数据加载和预处理
# transform_train = transforms.Compose([
#     transforms.RandomCrop(32, padding=4),
#     transforms.RandomHorizontalFlip(),
#     transforms.ToTensor(),
#     transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
# ])
#
# transform_test = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
# ])
#
# trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
# trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
#
# testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
# testloader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
#
# # 模型定义
# class CIFAR100CCNet(nn.Module):
#     def __init__(self, num_classes=100):
#         super(CIFAR100CCNet, self).__init__()
#         self.ccnet = ResNet(num_classes=num_classes)
#         self.fc = nn.Linear(2048, num_classes)  # 2048是一个假设值，实际需要根据CCNet的输出调整
#
#     def forward(self, x):
#         x = self.ccnet(x)
#         x = x.view(x.size(0), -1)
#         x = self.fc(x)
#         return x
#
# # 定义网络，损失函数和优化器
# net = CIFAR100CCNet(num_classes=100)
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
#
# # 训练函数
# def train(epoch):
#     net.train()
#     running_loss = 0.0
#     for i, data in enumerate(trainloader, 0):
#         inputs, labels = data
#         optimizer.zero_grad()
#         outputs = net(inputs)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
#         running_loss += loss.item()
#         if i % 100 == 99:  # 每100个batch打印一次
#             print(f'[Epoch {epoch+1}, Iter {i+1}] loss: {running_loss / 100:.3f}')
#             running_loss = 0.0
#
# # 测试函数
# def test():
#     net.eval()
#     correct = 0
#     total = 0
#     with torch.no_grad():
#         for data in testloader:
#             images, labels = data
#             outputs = net(images)
#             _, predicted = torch.max(outputs.data, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()
#     print(f'Accuracy of the network on the 10000 test images: {100 * correct / total:.2f}%')
#
# # 训练和测试循环
# for epoch in range(200):  # 训练200个epoch
#     train(epoch)
#     test()
#
# # 模型保存
# PATH = './cifar100_ccnet.pth'
# torch.save(net.state_dict(), PATH)
#
# # 加载模型
# # net = CIFAR100CCNet(num_classes=100)
# # net.load_state_dict(torch.load(PATH))

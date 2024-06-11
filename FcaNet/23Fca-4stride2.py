import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import CIFAR100
import torchvision.transforms as transforms
import torch.optim as optim
import os
import argparse
import pandas as pd
from layer import MultiSpectralAttentionLayer

train_loss_list = []
train_acc_list = []
loss_list = []
acc_list = []


def load_data(args):
    mean = [0.5070751592371323, 0.48654887331495095, 0.4409178433670343]
    std = [0.2673342858792401, 0.2564384629170883, 0.27615047132568404]

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    train_dataset = CIFAR100(root='./data', train=True, transform=train_transform, download=True)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_dataset = CIFAR100(root='./data', train=False, transform=test_transform, download=True)

    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    return train_loader, test_loader


def get_n_params(model):
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn * s
        pp += nn
    return pp


class MHSA(nn.Module):
    def __init__(self, n_dims, width=14, height=14, heads=4):
        super(MHSA, self).__init__()
        self.heads = heads

        self.query = nn.Conv2d(n_dims, n_dims, kernel_size=1)
        self.key = nn.Conv2d(n_dims, n_dims, kernel_size=1)
        self.value = nn.Conv2d(n_dims, n_dims, kernel_size=1)

        self.rel_h = nn.Parameter(torch.randn([1, heads, n_dims // heads, 1, height]), requires_grad=True)
        self.rel_w = nn.Parameter(torch.randn([1, heads, n_dims // heads, width, 1]), requires_grad=True)

        self.softmax = nn.Softmax(dim=-1)
        self.gamma = nn.Parameter(torch.zeros(1))  # 添加可学习的gamma参数

    def forward(self, x):
        n_batch, C, width, height = x.size()
        q = self.query(x).view(n_batch, self.heads, C // self.heads, -1)
        k = self.key(x).view(n_batch, self.heads, C // self.heads, -1)
        v = self.value(x).view(n_batch, self.heads, C // self.heads, -1)
        content_content = torch.matmul(q.permute(0, 1, 3, 2), k)

        content_position = (self.rel_h + self.rel_w).view(1, self.heads, C // self.heads, -1).permute(0, 1, 3, 2)
        content_position = torch.matmul(content_position, q)

        energy = content_content + content_position
        attention = self.softmax(energy)

        out = torch.matmul(v, attention.permute(0, 1, 3, 2))
        out = out.view(n_batch, C, width, height)

        # return out
        return self.gamma * out + x


# Criss-Cross Attention module
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
        proj_query_H = proj_query.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height).permute(0, 2,
                                                                                                                 1)
        proj_query_W = proj_query.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width).permute(0, 2,
                                                                                                                 1)
        proj_key = self.key_conv(x)
        proj_key_H = proj_key.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height)
        proj_key_W = proj_key.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width)
        proj_value = self.value_conv(x)
        proj_value_H = proj_value.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height)
        proj_value_W = proj_value.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width)
        energy_H = (torch.bmm(proj_query_H, proj_key_H) + self.INF(m_batchsize, height, width)).view(m_batchsize, width,
                                                                                                     height,
                                                                                                     height).permute(0,
                                                                                                                     2,
                                                                                                                     1,
                                                                                                                     3)
        energy_W = torch.bmm(proj_query_W, proj_key_W).view(m_batchsize, height, width, width)
        concate = self.softmax(torch.cat([energy_H, energy_W], 3))

        att_H = concate[:, :, :, 0:height].permute(0, 2, 1, 3).contiguous().view(m_batchsize * width, height, height)
        att_W = concate[:, :, :, height:height + width].contiguous().view(m_batchsize * height, width, width)
        out_H = torch.bmm(proj_value_H, att_H.permute(0, 2, 1)).view(m_batchsize, width, -1, height).permute(0, 2, 3, 1)
        out_W = torch.bmm(proj_value_W, att_W.permute(0, 2, 1)).view(m_batchsize, height, -1, width).permute(0, 2, 1, 3)

        return self.gamma * (out_H + out_W) + x

    def INF(self, B, H, W):
        return -torch.diag(torch.tensor(float("inf")).cuda().repeat(H), 0).unsqueeze(0).repeat(B * W, 1, 1)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, heads=4, mhsa=False, resolution=None, reduction=16, Fca=False):
        super(Bottleneck, self).__init__()

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        if not mhsa:
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, stride=stride, bias=False)
        else:
            self.conv2 = nn.ModuleList()
            self.conv2.append(MHSA(planes, width=int(resolution[0]), height=int(resolution[1]), heads=heads))
            if stride == 2:
                self.conv2.append(nn.AvgPool2d(2, 2))
            self.conv2 = nn.Sequential(*self.conv2)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)
        c2wh = dict([(64, 32), (128, 16), (256, 8), (512, 4)])  # 根据 CIFAR-100 特征图尺寸进行调整
        self.Fca = Fca  # 保存 Fca 参数状态
        if Fca:
            self.att = MultiSpectralAttentionLayer(planes * 4, c2wh[planes], c2wh[planes], reduction=reduction,
                                                   freq_sel_method='low16')
        # self.gamma = nn.Parameter(torch.ones(1))
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.Fca:  # 检查 Fca 参数状态
            out = self.att(out)
        out += self.shortcut(x)
        # out = out * self.gamma + self.shortcut(x)
        out = F.relu(out)
        return out


# https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=100, resolution=(32, 32), heads=4):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.resolution = list(resolution)

        # self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        if self.conv1.stride[0] == 2:
            self.resolution[0] /= 2
        if self.conv1.stride[1] == 2:
            self.resolution[1] /= 2
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # for ImageNet
        # if self.maxpool.stride == 2:
        #     self.resolution[0] /= 2
        #     self.resolution[1] /= 2

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2,Fca=True)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, Fca=True)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, heads=heads, mhsa=False)
        # self.cca = CrissCrossAttention(in_dim=2048)  # 添加Criss-Cross注意力机制

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Dropout(0.3),  # All architecture deeper than ResNet-200 dropout_rate: 0.2
            nn.Linear(512 * block.expansion, num_classes)
        )

    def _make_layer(self, block, planes, num_blocks, stride=1, heads=4, mhsa=False, Fca=False):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for idx, stride in enumerate(strides):
            layers.append(block(self.in_planes, planes, stride, heads, mhsa, self.resolution, Fca))
            if stride == 2:
                self.resolution[0] /= 2
                self.resolution[1] /= 2
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        # out = self.maxpool(out)  # for ImageNet

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        # out = self.cca(out)

        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out


def ResNet50(num_classes=100, resolution=(32, 32), heads=4):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, resolution=resolution, heads=heads)


def save_checkpoint(best_acc, model, optimizer, args, epoch):
    print('Best Model Saving...')
    if args.device_num > 1:
        model_state_dict = model.module.state_dict()
    else:
        model_state_dict = model.state_dict()

    torch.save({
        'model_state_dict': model_state_dict,
        'global_epoch': epoch,
        'optimizer_state_dict': optimizer.state_dict(),
        'best_acc': best_acc,
    }, os.path.join('checkpoints', 'checkpoint_model_best.pth'))


def _train(epoch, train_loader, model, optimizer, criterion, args):
    model.train()

    losses = 0.
    acc = 0.
    total = 0.
    for idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()

        output = model(data)
        _, pred = F.softmax(output, dim=-1).max(1)
        acc += pred.eq(target).sum().item()
        total += target.size(0)

        optimizer.zero_grad()
        loss = criterion(output, target)
        losses += loss.item()
        loss.backward()
        if args.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
        optimizer.step()

        if idx % args.print_intervals == 0 and idx != 0:
            print('[Epoch: {0:4d}], Loss: {1:.3f}, Acc: {2:.3f}, Correct {3} / Total {4}'.format(epoch,
                                                                                                 losses / (idx + 1),
                                                                                                 acc / total * 100.,
                                                                                                 acc, total))
    train_loss_list.append(losses / total)
    train_acc_list.append(acc / total * 100.)


def _eval(epoch, test_loader, model, args):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.
    acc = 0.
    with torch.no_grad():
        for data, target in test_loader:
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            output = model(data)
            _, pred = F.softmax(output, dim=-1).max(1)

            acc += pred.eq(target).sum().item()
            total_loss += criterion(output, target).item()
        print('[Epoch: {0:4d}], Acc: {1:.3f}'.format(epoch, acc / len(test_loader.dataset) * 100.))
        acc_list.append(acc / len(test_loader.dataset) * 100.)
        loss_list.append(total_loss / len(test_loader.dataset))
    return acc / len(test_loader.dataset) * 100.


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', type=str, default="读取额外的参数")

    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=1e-3)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--cuda', type=bool, default=True)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--print_intervals', type=int, default=40)
    parser.add_argument('--evaluation', type=bool, default=False)
    parser.add_argument('--checkpoints', type=str, default=None, help='model checkpoints path')
    parser.add_argument('--device_num', type=int, default=1)
    parser.add_argument('--gradient_clip', type=float, default=2.)

    args = parser.parse_args()

    train_loader, test_loader = load_data(args)
    model = ResNet50()

    optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)

    if not os.path.isdir('checkpoints'):
        os.mkdir('checkpoints')

    if args.checkpoints is not None:
        checkpoints = torch.load(os.path.join('checkpoints', args.checkpoints))
        model.load_state_dict(checkpoints['model_state_dict'])
        optimizer.load_state_dict(checkpoints['optimizer_state_dict'])
        start_epoch = checkpoints['global_epoch']
    else:
        start_epoch = 1

    if args.cuda:
        model = model.cuda()

    if not args.evaluation:
        criterion = nn.CrossEntropyLoss()
        lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=2, eta_min=0.0001)
        # lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

        global_acc = 0.
        for epoch in range(start_epoch, args.epochs + 1):
            _train(epoch, train_loader, model, optimizer, criterion, args)
            best_acc = _eval(epoch, test_loader, model, args)
            if global_acc < best_acc:
                global_acc = best_acc
                save_checkpoint(best_acc, model, optimizer, args, epoch)

            lr_scheduler.step()
            print('Current Learning Rate: {}'.format(lr_scheduler.get_last_lr()))
    else:
        _eval(start_epoch, test_loader, model, args)


if __name__ == '__main__':
    main()
    data = {'Train_Loss': train_loss_list,
            'Train_Accuracy': train_acc_list,
            'Loss': loss_list,
            'Accuracy': acc_list}

    df = pd.DataFrame(data)
    df.to_excel('accuracy_scores_Fca23.xlsx', index=False)

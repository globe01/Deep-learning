# 使用PyTorch框架和CPU/GPU实现CIFAR100分类模型搭建

以resnet18和resnet50为基础网络，基于参考论文[CCNet: Criss-Cross Attention for Semantic Segmentation](https://arxiv.org/abs/1811.11721)和相应代码[GitHub - CCNet](https://github.com/speedinghzl/CCNet)的方法进行训练。



- Resnet为基于resnet18进行训练
- Resnet+CCNet为resnet18+论文新增的注意力机制进行训练
- Resnet50为基于resnet50进行训练
- Resnet+CCNet为resnet50+论文新增的注意力机制进行训练

训练结果的loss和accuracy曲线以及完整结果的excel表格在相应名称的文件夹中。

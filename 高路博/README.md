# 使用PyTorch框架和CPU/GPU实现CIFAR100分类模型搭建

以resnet18和resnet50为基础网络，基于参考论文[CCNet: Criss-Cross Attention for Semantic Segmentation](https://arxiv.org/abs/1811.11721)和相应代码[GitHub - CCNet](https://github.com/speedinghzl/CCNet)的方法进行训练。



- `Resnet.py`为基于resnet18进行训练
- `Resnet+CCNet.py`为resnet18+论文新增的注意力机制进行训练
- `Resnet50.py`为基于resnet50进行训练
- `Resnet+CCNet.py`为resnet50+论文新增的注意力机制进行训练，是直接调用库的resnet50，准确率60.51%，结果位于`resnet50_ccnet_result`。
- `myResnet50.py`是手动实现的resnet50，`myResnet50+CCNet.py`是手动实现的resnet50+ccnet，准确率则是50.9%，进行卷积核等优化后达到68.85%，结果位于`myresnet50_ccnet_result`。
- MultiStepLR调度器+注意力机制.py是换成该调度器并用resnet50结合ccnet进行训练，100epoch后准确率77.41%，结果位于`new_try_result`。
- 余弦调度器+数据增强+注意力机制+预热.py是换新调度器并用resnet50结合ccnet进行训练，有数据增强和预热部分的代码，取消注释即可添加，100epoch最高达到79.88%，几个结果均位于`new_try_result`。
- 优化.py是在上一个的基础上新增动态注意力机制模块，100epoch最高达到80.20%，结果位于`optimize_result`。

训练结果的loss和accuracy曲线以及完整结果的excel表格在相应名称的文件夹中。

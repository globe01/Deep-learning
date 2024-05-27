'''
This code is borrowed from Serge-weihao/CCNet-Pure-Pytorch
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Softmax

# 辅助函数INF，该函数返回一个形状为(B*W, H, H)的张量，主要用于生成包含负无穷大的对角矩阵
# 用于注意力计算中的掩码操作，确保对角线元素被忽略
# 原本的INF函数必须使用cuda，现在改成了cpu也可以使用
# def INF(B,H,W):
#      return -torch.diag(torch.tensor(float("inf")).cuda().repeat(H),0).unsqueeze(0).repeat(B*W,1,1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def INF(B, H, W):
    return -torch.diag(torch.tensor(float("inf"), device=device).repeat(H), 0).unsqueeze(0).repeat(B * W, 1, 1)

# 论文所说的Criss-Cross注意力机制的模块，包括生成查询、键、值的卷积操作和Softmax操作
class CrissCrossAttention(nn.Module):
    """ Criss-Cross Attention Module"""
    def __init__(self, in_dim):
        super(CrissCrossAttention,self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.softmax = Softmax(dim=3)
        self.INF = INF # 调用辅助函数INF，生成掩码矩阵
        self.gamma = nn.Parameter(torch.zeros(1))# 缩放注意力输出


    def forward(self, x):
        # 先获取输入x的批量大小、高度和宽度
        m_batchsize, _, height, width = x.size()
        # 对输入应用query_conv、key_conv和value_conv卷积层，分别生成查询、键和值的特征图
        # 生成后这些特征图后要进行重排和变形，以便于后续的矩阵乘法操作
        proj_query = self.query_conv(x)
        proj_query_H = proj_query.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height).permute(0, 2, 1)
        proj_query_W = proj_query.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width).permute(0, 2, 1)
        proj_key = self.key_conv(x)
        proj_key_H = proj_key.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height)
        proj_key_W = proj_key.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width)
        proj_value = self.value_conv(x)
        proj_value_H = proj_value.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height)
        proj_value_W = proj_value.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width)
        
        # 计算水平和垂直方向上的能量矩阵

        # 计算水平方向上的注意力能量，并添加负无穷大的掩码矩阵
        energy_H = (torch.bmm(proj_query_H, proj_key_H)+self.INF(m_batchsize, height, width)).view(m_batchsize,width,height,height).permute(0,2,1,3)
        # 计算垂直方向上的注意力能量
        energy_W = torch.bmm(proj_query_W, proj_key_W).view(m_batchsize,height,width,width)
        # 将两个能量矩阵沿最后一个维度拼接，并应用softmax操作，得到注意力权重
        concate = self.softmax(torch.cat([energy_H, energy_W], 3))


        # 将注意力权重应用到值特征图上：
        # 1 提取水平方向的注意力权重，并进行变形
        att_H = concate[:,:,:,0:height].permute(0,2,1,3).contiguous().view(m_batchsize*width,height,height)
        #print(concate)
        #print(att_H) 
        # 2 提取垂直方向的注意力权重，并进行变形
        att_W = concate[:,:,:,height:height+width].contiguous().view(m_batchsize*height,width,width)
        # 3 计算水平方向的输出特征
        out_H = torch.bmm(proj_value_H, att_H.permute(0, 2, 1)).view(m_batchsize,width,-1,height).permute(0,2,3,1)
        # 4 计算垂直方向的输出特征
        out_W = torch.bmm(proj_value_W, att_W.permute(0, 2, 1)).view(m_batchsize,height,-1,width).permute(0,2,1,3)
        #print(out_H.size(),out_W.size())
        return self.gamma*(out_H + out_W) + x



if __name__ == '__main__':
    model = CrissCrossAttention(64) # 实例化CrissCrossAttention模块，输入通道数为64，测试其前向传播
    x = torch.randn(2, 64, 5, 6)# 生成输入张量x，大小为(2, 64, 5, 6)
    out = model(x)# 传入模型，计算输出
    print(out.shape)# 输出out的形状

# 加载dataset(使用torchvision 的 dataset), dataloader
import torch
from torch.utils.tensorboard import SummaryWriter
from torch import tensor, nn, stack, reshape, optim
from collections import OrderedDict
from torch.nn import functional as F
import numpy as np
from torchvision.datasets import CIFAR10, MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os
# 导入模型
from torchvision import models

# 单神经元下的logistics回归的定义
class Bin_Class(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.relu = nn.ReLU()
        # 此处使用一个神经元, 先映射, 再分类
        self.linear = nn.Linear(1, 1)
        self.sig = nn.Sigmoid()
    def forward(self, x):
        x = self.linear(x)
        x = self.sig(x)
        return x
# 样本
x_data = tensor([[1.], [2.], [3],], dtype=torch.float32)
y_date = tensor([[0.], [0.], [1.],], dtype=torch.float32)
std_x = x_data / len(x_data)
# 实例化模型
binary_class = Bin_Class()
# 实例化损失函数
criterion = nn.BCELoss()
# 优化器
# sigmid求导数时的值比较小, 可以增加学习率,加速梯度下降, 或增加轮数.
opt = optim.SGD(binary_class.parameters(), lr=1e-2)
for _ in range(10000):
    # 输出的预测结果
    res = binary_class(std_x)
    # # 梯度清零
    opt.zero_grad()
    # 损失计算
    loss = criterion(res, y_date)
    # 计算梯度
    loss.backward()
    # 更新梯度
    opt.step()
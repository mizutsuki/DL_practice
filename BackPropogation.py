import torch
from torch.utils.tensorboard import SummaryWriter
from torch import tensor, nn, stack, reshape, optim
from collections import OrderedDict
from torch.nn import functional as F
import numpy as np
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os
# 导入模型
from torchvision import models
# 设有数据集
y_data = [2, 4, 6]
x_data = [1, 2, 3]

lr = 0.01
# 初始化w 与 学习率
w = tensor([.0])
# 类方法required_grad = True, 设为求梯度的变量
w.requires_grad = True

# 前向传播
def forward(w, x):
    # 生成计算图
    res = w * x
    return res

# 损失计算, 生成计算图
def loss_func(w, x, y):
    hypo = forward(w, x)
    loss = (hypo - y)**2
    return loss

# 求梯度, 记录
sw = SummaryWriter("runs/exp2")
counter = 0
for _ in range(30):
    for x, y in zip(x_data, y_data):
        # 偏i/偏w 求梯度
        i = loss_func(w, x, y)
        i.backward()
        # 更新梯度 现参数 == 原参数 - 梯度 * 学习率
        # 梯度由backward()方法求出
        w.data = w.data - w.grad.data * lr
        counter += 1
        if not counter % 3:
            sw.add_scalar("loss", i.data.item(), counter)
            sw.add_scalar("grad", w.grad.data.item(), counter)
        # 归零
        w.grad.zero_()
sw.close()

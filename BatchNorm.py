import pandas as pd
import numpy as np
import torch
from torch import nn
from torch import tensor
from torch.utils.data import DataLoader
#
# 使用归一化公示的未知数
# 在使用torch时, 需要预先输入特征个数, 为每条样本都分配微调参数γ 和 β
def batch_norm2d_4d(x, shape, momentum = 0.01, moving_mea = 0, moving_std = 1, epsilon = 1e-5):
    # 检查是 训练 还是 推理(with torch.no_grad()执行这里)
    assert len(x.shape) in (2, 4)
    # 初始化， 只需一个array， 自动广播至匹配
    gamma = torch.ones(1, shape,1, 1)
    beta = torch.zeros(1, shape, 1, 1)
    # 推理模式
    if not torch.is_grad_enabled():
        # 对整批x标准化
        x = (x - moving_mea)/(moving_std + epsilon)
    else:
        if len(x.shape) == 2:
            # 2D tensor， 取得一个批次下所有相同特征的平均
            std, mean = torch.std_mean(x, dim=0, keepdim=True)
            gamma = torch.ones(1, shape)
            beta = torch.zeros(1, shape)
        else:
            # n个通道的均值
            std, mean = torch.std_mean(x, dim=(0, 2, 3), keepdim=True)
            gamma = torch.ones(1, shape, 1, 1)
            beta = torch.zeros(1, shape, 1, 1)
        # 标准化
        x = (x - mean) / (std + epsilon)
        # 更新动量的值
        moving_mea = momentum*moving_mea + (1-moving_mea) * mean
        moving_std = momentum * moving_std + (1 - moving_std) * mean
    # 使用补偿值进行伸缩移动
    y = gamma * x + beta
    # 为每个feature附加r和b调整.
    print(y.shape)
    return y, moving_std, moving_mea
# 3个batch, 10个通道, 每个通道内求平均
x = torch.ones((3, 10))
# tensor，特征数
res = batch_norm2d_4d(x, 10)

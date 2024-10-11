# 加载dataset(使用torchvision 的 dataset), dataloader
import torch
from torch.utils.tensorboard import SummaryWriter
from torch import tensor, nn, stack, reshape, optim, float32
from collections import OrderedDict
from torch.nn import functional as F
import numpy as np
from torchvision.datasets import CIFAR10, MNIST
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os
# RNN网络类定义
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers) -> None:
        super().__init__()
        # 显式输入尺寸: 18, hidden_layer:18, layer循环次数:输出字数
        self.rn = nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
        self.seq = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 8),
            nn.Linear(hidden_size * 8, hidden_size)
        )

    def forward(self, x):
        # 使用默认隐藏输入, 两层RNN + 残差值
        out, h_n1 = self.rn(x)
        _, h_n2 = self.rn(out, h_n1)
        x = self.seq(h_n2)
        return h_n1 + x
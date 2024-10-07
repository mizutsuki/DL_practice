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
# 导入模型
from torchvision import models
from PIL import Image

# 定义transform
transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=.1307, std=.3081)
])
# 数据集
train_set = MNIST("dataset/MNIST", train=True, download=False, transform=transforms.ToTensor())
test_set = MNIST("dataset/MNIST", train=False, download=False, transform=transforms.ToTensor())


# 设计网络
class Mod(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.seq = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.seq(x)
        return x


mod = Mod()
# 优化器
opt = optim.SGD(mod.parameters(), lr=0.01)
# 损失函数
criterion = nn.CrossEntropyLoss()
# 初始化训练
train_seq = DataLoader(train_set, batch_size=16, shuffle=True)
# 初始化测试
test_seq = DataLoader(train_set, batch_size=16, shuffle=True)

counter = 0
loss_sum = 0
for loop in range(1):
    for x in train_seq:
        counter += 1
        img, label = x
        # 送入模型
        res = mod(img)
        # 这里的loss是每一批样本的loss
        loss = criterion(res, label)
        opt.zero_grad()
        loss.backward()
        opt.step()
        # 使用类属性不生成计算图
        loss_sum += loss.data
        # if not counter % 100:
        #     loss_sum = 0
    # 每个loop统计一次
    with torch.no_grad():
        for counter, y in enumerate(test_seq, 0):
            img, tar = y
            pred = mod(img)
            # max()对于2d-tensor, dim = H, dim2 = W
            if not counter%100:
                _, accu = torch.max(pred, dim=1)
                accuracy = (accu.data == tar.data).sum()
                print(accuracy/16)
#
# for y in test_set:
#     img, label = y
#     r += torch.mean(img)
#
# print(r/70000)
# print(len(test_set))

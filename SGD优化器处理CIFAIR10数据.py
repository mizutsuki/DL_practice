# 加载dataset(使用torchvision 的 dataset), dataloader
import torch
from torch.utils.tensorboard import SummaryWriter
from torch import tensor, nn, stack, reshape, optim
from collections import OrderedDict
from torch.nn import functional as F
import numpy as np
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader


# 实例化数据集, datase序列中每个元素包含一个img && target
train_set = CIFAR10("dataset/train", download=False, transform=ToTensor())

# 实例化
class module(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.seq = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=(5, 5), padding=2),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(16, 32, kernel_size=(5, 5), padding=2),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=(5, 5), padding=2),
            nn.MaxPool2d(2, 2),
            # 输出的图像尺寸为4 * 4, 共64个channel
            nn.Flatten(),
            # 输入1024个参数, 输出64个参数
            nn.Linear(1024, 64),
            nn.Linear(64, 10)
        )
    #  更新参数
    def forward(self, x):
        x = self.seq(x)
        return x

# dataloader实例化
dataloader = DataLoader(train_set, batch_size=16, shuffle=True)
# 模型实例化
seq_mod = module()
# 优化器实例化, 指定parameters && 学习率超参数
gsd_opt = optim.SGD(seq_mod.parameters(), lr=0.01)
# 指定损失函数
loss_func = nn.CrossEntropyLoss()
# 实例化summary_writer
sw = SummaryWriter("runs/exp1")
counter = 0
for y in range(3):
    for x in dataloader:
        # 提取图像
        ipt, target = x
        # 对图像卷积
        res = seq_mod(ipt)
        # 10个被flatten的数据分别对应10个target
        res_loss = loss_func(res, target)
        # 记录
        sw.add_scalar("cross_entropy_loss", res_loss, counter)
        counter += 10
        # 梯度归零0
        gsd_opt.zero_grad()
        # 利用损失函数进行反向传播
        res_loss.backward()
        # 进入下一轮
        gsd_opt.step()
print("over")
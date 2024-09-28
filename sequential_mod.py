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
import os
# 导入模型
from torchvision import models

# 网络模型, 假设有这样一个神经网络
class model(nn.Module):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.seq = nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=(5, 5), padding=2),
                nn.MaxPool2d(kernel_size=(2, 2)),
                nn.Conv2d(16, 64, kernel_size=(3, 3), padding=2),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(64, 256, kernel_size=(3, 3), padding=2),
                nn.MaxPool2d(2, 2),
                nn.Flatten(),
                # 输入1024个参数, 输出64个参数
                nn.Linear(6400, 64),
                nn.Linear(64, 10)
            )
        #  更新参数
        def forward(self, x):
            x = self.seq(x)
            return x

# 数据集实例化
train_set = CIFAR10("dataset/train", download=False, transform=ToTensor())
test_set = CIFAR10("dataset/test",train=False, download=False, transform=ToTensor())

# dataloader实例化, (train_set, test_set)
dataloader = DataLoader(train_set, batch_size=64, shuffle=True)
test_loader = DataLoader(test_set, batch_size=32)

# 模型实例化
seq_mod = model()
# 转移至cuda上
seq_mod = seq_mod.cuda()
# 优化器实例化, 指定parameters && 学习率超参数
gsd_opt = optim.Adam(seq_mod.parameters())
# 指定损失函数
loss_func = nn.CrossEntropyLoss()
loss_func = loss_func.cuda()
# 实例化summary_writer
sw = SummaryWriter("runs/exp1")
# step
counter, test_counter = 0, 0

# 样本训练
for y in range(20):
    for x in dataloader:
        # 提取图像
        ipt, target = x
        ipt = ipt.cuda()
        target = target.cuda()
        # 对图像卷积
        res = seq_mod(ipt)
        # 16个被flatten的batch的数据分别对应16个target
        batch_loss = loss_func(res, target)
        # 梯度归零0
        gsd_opt.zero_grad()
        # 利用损失函数进行反向传播
        batch_loss.backward()
        # 进入下一轮
        gsd_opt.step()
        if not (counter % 500):
            # 记录batch loss
            sw.add_scalar("cross_entropy_loss", batch_loss, counter)
            counter += 1
            print(f"当前轮数: {counter} 损失为: {batch_loss.item()}")

    # 每轮训练结束后, 对样本的loss进行测试.
    correct_sample = 0
    # 无梯度模式, 加快计算速度
    with torch.no_grad():
        # 测试集 dataloader
        for test_data in test_loader:
            # 求损失
            test_img, test_target = test_data
            test_img, test_target = test_img.cuda(), test_target.cuda()
            test_img_calc = seq_mod(test_img)
            # 计算一个batch中损失的情况, 与target对比
            # dim = 1, 代表计算每一行
            predict_res = test_img_calc.argmax(1)
            # 利用 布尔张量, 计算正确个数的数量
            single_correct_sample = (predict_res == test_target).sum()
            # 记录正确个数
            correct_sample += single_correct_sample
        # 所有样本结束一次训练后, 验证正确率
        accuracy = correct_sample / len(test_set)
        print(f"第: {test_counter}次样本循环, 正确率为: {accuracy}")
        print(f"正确样本数为{correct_sample}")
        # sw记录正确率
        sw.add_scalar("test_loss_sum", accuracy, test_counter)
        test_counter += 1

sw.close()
torch.save(seq_mod.state_dict(), "Model/CIFAR10_class.pth")
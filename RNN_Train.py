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
from Rnn_Net import RNN
# 数据程序
from Rnn_DataSet import pair, input_size, hidden_size, num_iter, mapping, L

# 字符提取, 分类
def extract(x: list):
    pair = x
    # 合并
    content = ""
    for x in pair:
        content += x[0] + x[1]
    # 转tensor → one-hot编码
    counter = 0
    torch.empty(0, dtype=torch.int64)
    mapping = {'\0': 0}
    for i in content:
        # 找不到, 使用哈希索引法, 加快处理速度
        if i not in mapping:
            counter += 1
            mapping[i] = counter

    # 初始化长整型做标签, 在数据很多时, 可以考虑用numpy + csv or 数据库
    # mapping 用于使用 One-hot标签, 查找数据, 预测输出,找tensor.max()
    return mapping


# 样本x进行编码
def str_encoder(x: str):
    container = []
    # 规范化输入
    sub_x = len(x)
    # 接收形参
    x_ = x
    # 补全
    if sub_x < num_iter:
        # 输出补全, 由RNN迭代数确定
        x_ = x + '\0' * (num_iter - sub_x)

    # 单个字符转码
    def encoder(ele: str):
        char = ele
        # 初始如空表
        cod_list = []
        # 当字符 == mapping.keys, 此处设为1, 否则为0
        for x in mapping.keys():
            cod_list.append(1 if char == x else 0)
        return cod_list

    # 对于字符串编码
    for ele in x_:
        # 加入
        container.extend([encoder(ele)])
    container = tensor(container, dtype=torch.float32)
    return container


# 字符串转Label:
def cvt_2_long(y: str):
    container = []
    # 规范化target
    sub_y = len(y)
    y_ = y
    if sub_y < num_iter:
        # 输出补全, 由RNN迭代数确定
        y_ = y + '\0' * (num_iter - sub_y)
    for lable in y_:
        container.append([mapping.get(lable)])
    container = tensor(container, dtype=torch.int64)
    return container

def train():
    # 实例化
    rnn = RNN(input_size, hidden_size, num_layers=num_iter)
    # 损失函数
    criterion = nn.CrossEntropyLoss()
    # 优化器
    opti = optim.Adam(rnn.parameters())
    # 轮计数器
    counter = 0
    # 统计
    loss_counter = None
    # 存储
    pth = os.path.join(r"C:\Project\20240916Pytorch\Model", "chat_rnn.pth")
    for loop in range(500):
        # 每次取一条样本, 使输入 = 8, 输出 = 8
        for seq, tar in pair:
            # 编码为one-hot
            cod_seq = str_encoder(seq).view(L, 1, input_size)
            # LongTensor → 标签码
            tar_seq = cvt_2_long(tar)
            # 传入数据: input: L, H(in), hidden采用默认全0
            res = rnn(cod_seq)
            # 梯度清零
            opti.zero_grad()
            # 损失计算
            loss = tensor(0, dtype=torch.float32)
            for i in range(L):
                loss += criterion(res[i], tar_seq[i])
            loss.backward()
            opti.step()
            # 统计模块
            counter += 1
            if not counter % 50:
                sum_loss = loss.data
                print(sum_loss)
                # 权重保存
                if not loss_counter:
                    loss_counter = sum_loss
                else:
                    if loss_counter > sum_loss:
                        torch.save(rnn.state_dict(), pth)
                        loss_counter = sum_loss
                    else:
                        pass

if __name__ == '__main__':
    train()
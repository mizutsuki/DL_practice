import torch
from torch import tensor, optim
import torch.nn as nn
from matplotlib import pyplot as plt
import numpy as np
from numpy import random
from torch.utils.tensorboard import SummaryWriter
from typing import Optional
# y
def y(x, noise: Optional = 0):
    return 2*np.sin(x) ** 2 + x ** .8 + noise


# 生成50个随机的点, 训练集都是假定正确的样本
x_train = np.sort(random.rand(50)) * 6
# 高斯噪音
noise = random.normal(0, 0.4, 50)
y_train = y(x_train, noise)
# 测试集 start stop step. 取其中50个
x_test = np.arange(0, 6.28, .12)[0:50]
y_test = y(x_test)
# 目的: 使用注意力机制, 使模型使用训练样本建模, 能够拟合测试集数据

# 方法1: 平均汇聚
def average_pool(y_train):
    # 求和 → [] 转为list → 乘len(个数) 变回向量
    return np.array([np.sum(y_train) / len(y_train)] * len(y_train))

y_pred = average_pool(x_train)

# 方法2:
# 无参Nadaraya-Watson, query && key-value对
# 形参x_train, y_train, 测试集:query = x_test,
# 这里的query是样本的集合, 每个x都是单独的query,
# 对这些query计算, 得到query中每一个样本的softmax, 返回后,求sum
# softmax只是一种计算权重的方法, 将最大占比 0 < 占比 <1
def softmax(x):
    res = np.exp(x)
    res = res/np.sum(res)
    return res
# input: 记录exp(-1/2(x-xi)**2)所有结果的集合
# output: softmax过的权重向量
def nw_pool(x_train, y_train, query):
    key = x_train
    value = y_train
    res = [0] * 50
    for i in range(50):
        # 所有query计算的集合
        r = -1 / 2 * (query[i] - key) ** 2
        # 所有结果的集合
        # (exp(x1)/exp(Σxn)), exp(x2)/exp(Σxn)), ...)*value
        # 求和, 返回的值
        soft_out = np.sum(softmax(r) * value)
        # 对于所查询的每个query元素, 最终有
        res[i] = soft_out
    return np.array(res)
y_pred2 = nw_pool(x_train, y_train, x_test)
# 方法3
# 使用神经元的含参注意力汇聚, 自适应, 可学习
class nw_pool_adaptive(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # 调节每次输出的值, 平滑, 防止过大 or 过小.
        self.w = nn.Sequential(nn.Linear(50, 128, bias=False),
                               nn.Linear(128, 256, bias=False),
                               nn.Linear(256, 50, bias=False))
        # self.w =
    def forward(self, x_train, y_train, query):
        size = 50
        key = x_train
        value = y_train
        # 转矩阵
        a = tensor(key, dtype=torch.float32)
        a = a.repeat_interleave(size).reshape(-1, size)

        # 转矩阵
        query = tensor(query, dtype=torch.float32)
        query = query.expand(size, size)
        # query = query.expand(size, size)
        # 这里可以使用广播机制
        # 如果上面的a reshape成矩阵, 则query可以是向量
        # 注意这里是行向量, 因为repeat_interleave的时候,
        # 变成成了[[a, a, ...], [n, n, ...]]
        query = query.reshape(1, -1)

        # 相乘得到结果
        value = tensor(value, dtype=torch.float32)
        res = torch.softmax(-1 / 2 * (a - query) ** 2, dim=-1)

        out = self.w(res)@value
        return out

# 实例化模型
nwp = nw_pool_adaptive()
# 优化器
opti = optim.SGD(nwp.parameters(), lr=.02)
# 实例化损失函数
criterion = nn.MSELoss()
y_test = tensor(y_test, dtype=torch.float32)
for x in range(500):
    # 计算值
    res = nwp(x_train, y_train, x_test)
    loss = criterion(res, y_test)
    opti.zero_grad()
    loss.backward()
    opti.step()
    print(loss)

rrr = nwp(x_train, y_train, x_test)
y_pred3 = rrr.detach().numpy()

# 图例
def plot_img():
    train_ = plt.scatter(x=x_train, y=y_train, color="red")
    test_, = plt.plot(x_test, y_test, color="b")
    plt.plot(x_train, y_pred, color = "g")
    plt.plot(x_test, y_pred2, color="purple")
    plt.plot(x_test, y_pred3, color="orange")
    plt.legend(handles=[train_, test_], labels=["train_data", "sin_function"], loc="best")
    plt.show()

plot_img()

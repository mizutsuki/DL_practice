import torch
from torch import tensor,nn
from typing import Optional
import numpy as np

# 自注意力位置编码
class PosistionEncoding(nn.Module):
    # p为初始化一个全0矩阵(从0开始), 记录位置编码, 与采样后的值结合, 作为输入.
    # number_hiddens: 隐藏层数量(字符编码), max_len: 字符数
    # pos && i 共同决定  正弦 / 余弦波的密度
    def __init__(self, seq_lens, num_hiddens) -> None:
        super().__init__()
        # H = seq_lens, W = num_hiddens
        self.p = torch.zeros(1, seq_lens, num_hiddens)
        # pos / 10000^(2i/d)
        # R ∈ n*1(所有的i)
        pos = torch.arange(seq_lens, dtype=torch.float32).reshape(-1, 1)
        # 拿到所有的 1000^j/d
        denomiantor = torch.pow(10000, torch.arange(0, num_hiddens, step=2, dtype=torch.float32)/num_hiddens)
        # 分别拿到所有的 偶数/奇数 位置编码, 计算sin, cos.
        # pos / denominator = 一个矩阵
        x = pos / denomiantor
        self.p[:, :, 0::2] = torch.sin(x)
        self.p[:, :, 1::2] = torch.cos(x)
        self.drop = nn.Dropout(0.1)
    def forward(self, x):
        # 加性注意力, 不管p的lenth多长, 为保证可加性, len截取到x.shape[1]
        # 还可以根据batch来变化.
        x = x + self.p[:, :x.shape[1], :]
        return self.drop(x)
# 需要是偶数, 否则还要调整, 不便
a = PosistionEncoding(12, 10)
test = torch.randn(1, 12, 10)
res = a(test)
print(res.shape)
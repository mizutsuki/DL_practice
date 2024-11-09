import torch
from torch import tensor,nn
from typing import Optional
import numpy as np

# 位置编码 PE(pos, 2i/2i+1) = sin/cos(pos/10000^(2i/d))
class PosistionEncoding(nn.Module):
    # p为初始化一个全0矩阵(从0开始), 记录位置编码, 与采样后的值结合, 作为输入.
    # number_hiddens: 隐藏层数量(字符编码), max_len: 字符数
    # pos && i 共同决定  正弦 / 余弦波的密度
    # parameter: 编码长度, hidden层
    def __init__(self, seq_len, num_hidden) -> None:
        super().__init__()
        self.max_len = seq_len
        # 位置编码
        pos = tensor(list(range(seq_len))).reshape(-1, 1)
        # step = 2
        i = tensor(list(range(0, num_hidden, 2)), dtype=torch.float32)
        # 位置编码矩阵, 1用于广播
        self.pos_encod_mat = torch.zeros(1, seq_len, num_hidden)
        # 正弦/余弦编码(一次性计算所有pos的编码)
        sin_mat = torch.sin(pos / 10000 ** (i / num_hidden))
        cos_mat = torch.cos(pos / 10000 ** (i / num_hidden))
        self.pos_encod_mat[:, :, 0::2] = sin_mat
        self.pos_encod_mat[:, :, 1::2] = cos_mat

    def forward(self, x):
        # 最大附加位置编码到max_len位置, 增强鲁棒性
        x = x + self.pos_encod_mat[:, : x.shape[1], :]
        return x
# 需要是偶数, 否则还要调整, 不便
a = PosistionEncoding(12, 10)
test = torch.randn(1, 12, 10)
res = a(test)
print(res.shape)
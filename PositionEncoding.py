import pylab as p
import torch
from torch import tensor,nn
# 自注意力位置编码
class PosistionEncoding(nn.Module):
    # p为初始化一个全0矩阵(从0开始), 记录位置编码, 与采样后的值结合, 作为输入.
    # number_hiddens: 隐藏层数量(字符编码), max_len: 字符数
    def __init__(self, num_hiddens, max_len) -> None:
        super().__init__()
        # HW
        self.p = torch.zeros(1, max_len, num_hiddens)
        # i / 10000^(2j/d)
        # R ∈ n*1(所有的i)
        i = torch.arange(max_len, dtype=torch.float32).reshape(-1, 1)
        # 先拿到所有的1000^j/d, 再分奇偶
        denomiantor = pow(10000, torch.arange(0, num_hiddens, dtype=torch.float32))
        # 分别拿到所有的 偶数/奇数 位置编码, 计算sin, cos.
        # i / denominator = 一个矩阵
        self.p[:, :, 0::2] = torch.sin(i / denomiantor)
        self.p[:, :, 1:2] = torch.cos(i / denomiantor)
        self.drop = nn.Dropout(0.2)
        print(p.shape)
    def forward(self, x):
        # 加性注意力, 不管p的lenth多长, 为保证可加性, len截取到x.shape[1]
        x = x + self.p[:, :x.shape[1], :]
        return self.drop(x)

import pylab as p
import torch
from torch import tensor, nn
from typing import Optional
import numpy as np

# 一个样本
batch = 2
seq_lenth = 2
# dim = 词向量维度
dim = 4
# set of K V Q into a matrix
# seq_len.Q = seq_len.V
Q = torch.rand(batch, seq_lenth, dim)
V = torch.rand(batch, seq_lenth, dim)
# dim.Q = dim.K
K = torch.rand(batch, seq_lenth, dim)


# 点积注意力 [KQ/sqrt(d)]*v
class DotAtt(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # 新版本不支持随机随机dropout
        self.sftmax = nn.Softmax(dim=-1)
        self.dp = nn.Dropout(p=.1)

    # softmax遮罩, 可以选择性遮蔽下文文字(对seq_len起效).
    def softmax_masker(self, x, valid_len: [Optional] = None):
        # 转tensor
        valid_len = (valid_len if torch.is_tensor(valid_len) else tensor(valid_len))
        if not valid_len:
            return self.sftmax(x)
        else:
            aim_mask = valid_len.reshape(-1)
            # 获取shape
            x_shape = x.shape
            # 变为2Dtensor
            x = x.reshape(-1, x.shape[-1])
            # 插入
            aim_mask = torch.repeat_interleave(aim_mask, x_shape[-1])
            # 从flatten重建 n * hidden_layer
            aim_mask = aim_mask.reshape(-1, x_shape[-1])
            # indices_matrix
            indices_matrix = tensor(list(range(x.shape[-1]))).expand(x.shape)
            # masker, 索引 > indices的设为true, 最小为index = 1之后全部遮蔽
            masker = indices_matrix > aim_mask
            x[masker] = -1e6
            # 恢复形状
            x = x.reshape(x_shape)
            return self.sftmax(x)

    def forward(self, K, V, Q, valid_len: [Optional] = None):
        # (Q*K.T/sqrt(d)) * V
        score = Q @ K.transpose(1, 2) / (dim ** .5)
        # get a problematic answer code,
        # evaluated with Cross_Entropy_Loss
        att = self.softmax_masker(score, valid_len) @ V
        return self.dp(att)


# multiHead_ATT
class MulAtt(nn.Module):
    # 显式输入, 隐藏层尺寸, 头数
    def __init__(self, q_len, k_len, v_len, num_hidden) -> None:
        super().__init__()
        # hidden_layer
        self.num_hidden = num_hidden
        # 点积注意力
        self.attion = DotAtt()
        self.w_q = nn.Linear(q_len, num_hidden)
        self.w_k = nn.Linear(k_len, num_hidden)
        self.w_v = nn.Linear(v_len, num_hidden)
        # 为融合输出增加线性.
        self.w_o = nn.Linear(num_hidden, num_hidden)

    # transforrmer多头输入预处理
    def transpose_qkv(self, x, num_heads):
        # 将最后的hidden_layer拆分,
        x = x.reshape(x.shape[0], x.shape[1], num_heads, -1)
        # batch, 头数, seq_len, head_dim
        x = x.permute(0, 2, 1, 3)
        # 头数与batch合并, 变成batch*头数, seq_len不变, head_num不变
        # 可以使每个头关注不同的信息(已做embedding), 使感受野变为2D
        return x.reshape(-1, x.shape[2], x.shape[3])

    # transformer输出处理
    def transpose_out(self, x, num_heads):
        x = x.reshape(-1, num_heads, x.shape[1], x.shape[2])
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        return x

    def forward(self, K, V, Q, head):
        self.num_head = head
        trans_k = self.transpose_qkv(K, self.num_head)
        trans_v = self.transpose_qkv(V, self.num_head)
        trans_q = self.transpose_qkv(Q, self.num_head)
        # 一次计算, 拿到所有的头
        # 注意力计算, 遮蔽num_head后面的字符
        # 因为这里是多头并行的, 需要遮蔽头数以后的值(即遮蔽全部后文)
        # 有了embedding,相当于一次只遮蔽几个字.
        # k, v, q, valid_len
        out_put = self.attion(trans_k, trans_v, trans_q, self.num_head)
        # transform还原输入的张量
        out_cat = self.transpose_out(out_put, self.num_head)
        # 换增加非线性
        return self.w_o(out_cat)


# 实例化样本, 计算
v = torch.randn(2, 4, 8)
a = MulAtt(seq_lenth, seq_lenth, seq_lenth, dim)
res = a(K, V, Q, 2)
print(res)
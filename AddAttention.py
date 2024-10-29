from torch import tensor, LongTensor, nn, optim
import string
from typing import Optional

import torch
import torch.nn.functional as F

import pandas as pd
import numpy as np


# 通过最后一个轴上遮蔽元素实现softmax操作
# 形参: 待遮蔽元素, 合法长度, 每一行合法长度的值
def masked_softmax(x, valid_lens:Optional[torch.tensor] = None):
    # 未指定遮罩, 则对全体求softmax
    if valid_lens is None:
        return nn.functional.softmax(x, dim=1)
    else:
        # 获取shape, 保存
        shape = x.shape
        # flatten
        valid_lens = valid_lens.reshape(-1)
        valid_lens = valid_lens.repeat_interleave(shape[-1])
        # reshape, 变成竖的
        valid_lens = valid_lens.reshape(-1)
        # 2D tensor
        x = x.reshape(-1, shape[-1])
        # print(valid_lens.shape, x.shape)
        # 保持H,W不变
        masked_val = masker(x, valid_lens, -1e6)
        # 还原
        masked_val = masked_val.reshape(shape)
        # 对所有元素计算softmax
        return F.softmax(masked_val, dim=-1)
def masker(x, val_lens, value):
    # 获取shape
    shape = x.shape
    # 索引矩阵
    ind = torch.arange(shape[-1])
    # value的h与x等高
    val_lens = val_lens.reshape(-1, shape[-1])
    # 生成mask矩阵, 利用torch的广播特性
    mask = ind < val_lens
    # 对遮罩部分赋值一无穷小值, 使其在softmax时 == 0
    x[~mask] = value
    return x

# 遮罩
# 加注意力
class AddAtt(nn.Module):
    def __init__(self,keysize, numhidden,  quesize) -> None:
        super().__init__()
        # 之前说过, nn.Linear可以进行矩阵运算
        self.W_k = nn.Linear(keysize, numhidden)
        self.W_q = nn.Linear(quesize, numhidden)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()
        # 最后 * vi， 将加权得到的值 * vi
        self.w_v = nn.Linear(numhidden, 1)

    def forward(self, key, que, value, valid_lens: Optional[torch] = None):
        out1 = self.W_k(key)
        out2 = self.W_q(que)
        # key && query 融合, 保留特征. 使用tensor的广播机制
        # out1 → batch, h, w → batch, h, 1, w
        # out2 → batch, h, w → batch, 1, h, w
        # 得到所有 key && que 的组合, 类似于SQL中两个表组合
        features = out1.unsqueeze(2) + out2.unsqueeze(1)
        # 映射
        features = self.tanh(features)
        print(features.shape)
        scores = self.w_v(features).squeeze(-1)
        # 计算权重, 这里没有遮挡, 计算所有
        self.attention_weights = masked_softmax(scores, valid_lens)
        return torch.matmul(self.attention_weights, value)


# key = 10 * 15， value = 10 * 8，q = 10*15
keysize = 20
quesize = 15
valuesize = 8
pair = 10
numhidden = 12
# batch == 2
key = torch.randn(1, pair, keysize)
que = torch.randn(1, pair, quesize)
value = torch.randn(1, pair, valuesize)
addatt = AddAtt(keysize, numhidden, quesize)
res = addatt(key, que, value)
print(res.shape)
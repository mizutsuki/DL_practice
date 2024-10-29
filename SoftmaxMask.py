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
        # 获取shape
        shape = x.shape
        # flatten
        valid_lens = valid_lens.reshape(-1)
        valid_lens = valid_lens.repeat_interleave(shape[-1])
        # reshape
        valid_lens = valid_lens.reshape(-1, shape[-1])
        x = x.reshape(-1, shape[-1])
        # 保持H,W不变
        masked_val = masker(x.reshape(-1, shape[-1]), valid_lens, -1e6)
        # 还原
        masked_val = masked_val.reshape(shape)
        # 对所有元素计算softmax
        return F.softmax(masked_val, dim=-1)
# 遮罩
def masker(x, val_lens, value):
    # 获取shape
    shape = x.shape
    # 索引矩阵
    ind = torch.arange(shape[-1])
    # 生成mask矩阵, 利用torch的广播特性
    mask = ind < val_lens
    # 对遮罩部分赋值一无穷小值, 使其在softmax时 == 0
    x[~mask] = value
    return x

# 测试样本x, 遮罩长度, 希望遮蔽前两个
x = torch.randn(1, 1, 4)
val_lens = tensor([2])

# batch = 2, h = 2, w = 4
x2 = torch.randn(2, 2, 5)
# 希望1, 2, 3, 4行分别屏蔽1, 2, 3, 4个元素, 剩下的变成-1e6
val_lens2 = tensor([[1, 2], [3, 4]])
res = masked_softmax(x2, val_lens2)
print(res)

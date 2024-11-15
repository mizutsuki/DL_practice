import torch
from torch import tensor
# 序列遮罩: 对输入的seq_len进行屏蔽,支持2d与3d tensor:
def batch_seq_msk(x, valid_len, replaced_value=0):
    # get_size
    org_shape = x.shape
    if (len(org_shape) == 3):
        batch, seq_len, hidden_size = org_shape
    else:
        batch, seq_len, hidden_size = x.unsqueeze(dim=0).shape
    # reshape x size
    x = x.reshape(1, -1, hidden_size)
    # flattened_size
    flattened_seq_len = seq_len * batch
    # 不管valid_len为何种形式, 都flatten到1-D张量
    flattened_valid_len = valid_len.reshape(-1)
    # 如果只有一个元素, 扩展到全部长度
    if (len(flattened_valid_len) == 1):
        flattened_valid_len = flattened_valid_len.repeat(flattened_seq_len)
    # 输入一个batch的遮罩, 则扩展到全部天涯不属于
    elif (len(flattened_valid_len) == seq_len):
        flattened_valid_len = flattened_valid_len.repeat(batch)
    # 对每个batch单独设置一个元素的遮罩, 拓展到该batch所有行
    elif (len(flattened_valid_len) == batch):
        flattened_valid_len = flattened_valid_len.repeat_interleave(seq_len)

    masker = flattened_valid_len.reshape(-1, 1)
    indices = tensor(list(range(hidden_size)))
    msk = (masker <= indices)
    x[:, msk] = replaced_value
    X = x.reshape(org_shape)
    return X
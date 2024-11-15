import torch
from torch.utils.data import DataLoader
from PIL import Image
from torch import tensor, LongTensor, nn, optim, Tensor
import string
from typing import Optional
import torch.nn.functional as F
from matplotlib import pyplot as plt
import pandas as pd
from transformers import BertTokenizer
from typing import Optional

class seq2seqEnco(nn.Module):
    # 参数: vocabulary_size, embedding size
    def __init__(self, vocab_size, embedding_size,
                 hidden_size, hidden_layers,
                 drop_out = 0, **kwargs) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.gru = nn.GRU(embedding_size, hidden_size, hidden_layers,
                          dropout=drop_out)
    def forward(self, x):
        x = self.embedding(x)
        # 非Cell的RNN网络中, 时间步和batch对调, 加速运算.
        # 注: 只针对分batch数据, 未分batch, 默认batch为时间步
        if len(x.shape) == 3:
            x = x.permute(1, 0, 2)
        x, state = self.gru(x)
        return x, state


class seq2seqDeco(nn.Module):
    def __init__(self, vocab_size, embedding_size,
                 hidden_size, hidden_layers, drop_out=0) -> None:
        super().__init__()
        # decode输入编码
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        # 解码器同时接受 embedding_size && hidden_size的输入
        self.gru = nn.GRU(embedding_size + hidden_size, hidden_size, hidden_layers,
                          dropout=drop_out)
        # 还原
        self.fn = nn.Linear(embedding_size, vocab_size)
    # 接受encoder的输出[output, hidden]
    def init_state(self, enco_opts, *args):
        # 索引1为: 拿出其中的隐藏状态
        # shape = D * num_layers or D * num_layers, N, Hout
        # 注意:
        # 进入RNN时选择了batch_first, hidden随之变化
        # enco_opts[1], 为含有最后一个时间步所有隐藏状态的列表
        return enco_opts[1]

    def forward(self, x, state, *args):
        # 针对于batched input x 的调整
        x = self.embedding(x)
        if(len(x.shape) == 3):
            x = x.permute(1, 0, 2)
        x_shape = x.shape
        # 拿到最后一刻最后一层的隐藏状态,
        # 为每个时间步都加一个encoder的隐藏层
        hnn = state[-1].repeat(x_shape[0], 1, 1)
        mixed_ipt = torch.cat((x, hnn), dim=-1)
        out, state = self.gru(mixed_ipt)
        # step, batch, hidden → batch → step, hidden
        vocab = self.fn(out.permute(1, 0, 2))
        return vocab


class Enco_Deco(nn.Module):
    def __init__(self, vocab_size, embedding_size,
                 hidden_size, hidden_layer) -> None:
        super().__init__()
        self.seq2seq_encoder = seq2seqEnco(vocab_size, embedding_size,
                                           hidden_size, hidden_layer)
        self.seq2seq_decoder = seq2seqDeco(vocab_size, embedding_size,
                                           hidden_size, hidden_layer)
    def forward(self, enco_x, deco_x):
        enco_x = self.seq2seq_encoder(enco_x)
        ipt_state = self.seq2seq_decoder.init_state(enco_x)
        x = self.seq2seq_decoder(deco_x, ipt_state)
        return x


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

# 测试时使用,
# 使终止标记之后的值不参与CrossEntorphyLoss损失运算
class MskedCrossEntroLoss(nn.CrossEntropyLoss):
    def __init__(self, valid_len, reduction) -> None:
        super().__init__()
        self.valid_len = valid_len
        self.reduction = reduction
    def forward(self, inpt, label):
        weights_one = torch.ones_like(label)
        # 输出序列遮蔽矩阵
        weights = batch_seq_msk(weights_one, self.valid_len)
        # 使用父类方法
        res = super().forward(
            # permute为: number_of_classes, vector
            inpt.permute(0, 2, 1), label
        )
        # 使输出的部分交叉熵变为0,不参与梯度运算,
        # 取mean作为一个batch的CrossEntrophyLoss
        return (res * weights).mean(dim=1)

'''# 实例化
valid_len = tensor([0])
mce = MskedCrossEntroLoss(valid_len=valid_len, reduction="none")
# batch = 1, seq_len = 2, hidden = 2
x = tensor([[[.2, .4],
              [.2, .4]]])
label = tensor([[1, 1]])
res = mce(x, label)
print(res)'''

model= Enco_Deco(4, 4, 4, 1)
for x in model.modules():
    if type(x) == nn.GRU or type(x) == nn.Linear:
        for name, para in x.named_parameters():
            if "weight" in name:
               # 直接在原地址修改: _
               nn.init.xavier_uniform_(para)

enco_x = tensor(list(range(4))).reshape(1, 4)
deco_x = tensor(list(range(4))).reshape(1, 4)
res = model(enco_x, deco_x)
print(res.shape)

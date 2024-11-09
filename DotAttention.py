import math

import torch
from torch import tensor,nn
class Dotatt(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.softmax = nn.Softmax(dim=-1)
    def forward(self, K, V, Q, d):
        # 找与K最相近的Q
        score = torch.bmm(Q, K) / math.sqrt(d)
        print(score)
        # 这里可以使用遮罩softmax
        att = self.softmax(score)
        res = torch.bmm(att, V)
        return res
# 当query只有一条, key只有一个, value只有一个, 遮蔽也只能遮蔽一个
n = 2
m = 2
# key && query has same W
d = 10
# key && value has equal pair
v = 12
# question, key, value
Q = torch.randn(1, n, d)
V = torch.randn(1, m, v)
K = torch.randn(1, m, d)
K.transpose_(1, 2)

# 实例化
dot = Dotatt()
res = dot(K, V, Q, d)
# print(res)

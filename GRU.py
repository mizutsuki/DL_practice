import torch
from torch import nn
from typing import Optional
class GruCell(nn.Module):
    def __init__(self,x_size, hidden_size) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.x_size = x_size
        # ResetGate_weight
        self.Wxr = nn.Linear(x_size, hidden_size)
        self.Whr = nn.Linear(hidden_size, hidden_size)
        # UpdatGate_weight
        self.Wxz = nn.Linear(x_size, hidden_size)
        self.Whz = nn.Linear(hidden_size, hidden_size)
        # candadate hidden state weight
        self.Wxh = nn.Linear(x_size, hidden_size)
        self.Whh = nn.Linear(hidden_size, hidden_size)
        # activate_func
        self.softmax = nn.Softmax(dim=-1)
        self.tanh = nn.Tanh()
    def forward(self, x, h:Optional = None):
        if h is None:
            x_shape = list(x.shape)
            x_shape[-1] = self.hidden_size
            h = torch.zeros(x_shape)
        # ResetGate
        Rt = self.softmax(self.Wxr(x) + self.Whr(h))
        # UpdateGate
        Zt = self.softmax(self.Wxz(x) + self.Whz(h))
        # candidate hidden state
        H_ = self.tanh(self.Wxh(x) + self.Whh(Rt*h))
        # Real hidden state
        Ht = Zt*h + (1-Zt)*H_
        return Ht

grucell = GruCell(3, 4)
x = torch.randn(1, 3)
res = grucell(x)
print(res)
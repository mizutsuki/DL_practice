import torch
from torch import tensor, nn
from typing import Optional

class RnnCell(nn.Module):
    def __init__(self, input_size, hidden_size: Optional) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.input_weight = nn.Linear(input_size, hidden_size)
        self.hidden_weight = nn.Linear(hidden_size, hidden_size)
        self.tanh = nn.Tanh()
    def forward(self, x, hidden:Optional = None):
        x_trans = self.input_weight(x)
        if hidden is None:
            x_shape = list(x.shape)
            x_shape[-1] = self.hidden_size
            hidden = torch.zeros(x_shape)
        h_trans = self.hidden_weight(hidden)
        return self.tanh(x_trans + h_trans)
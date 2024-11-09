from torch import nn
# PositionWiseFFN(Feed-Forward Neural Network)
# 这其实只是一个前馈网络
class PositionWiseFFN(nn.Module):
    def __init__(self, ffn_input, ffn_hidden, ffn_out) -> None:
        super().__init__()
        self.linear1 = nn.Linear(ffn_input, ffn_hidden)
        self.linear2 = nn.Linear(ffn_hidden, ffn_out)
        self.relu = nn.ReLU()
        self.seq1 = nn.Sequential(
            self.linear1,
            self.relu,
            self.linear2
        )
    def forward(self, x):
        x = self.seq1(x)
        return x
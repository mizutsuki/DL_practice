import torch
from torch import tensor, nn,randn
# 构造残差块
class residual_block(nn.Module):
    # 输入的channel
    def __init__(self, channel: int) -> None:
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=(3, 3), padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(3, 3, kernel_size=(3, 3), padding=1)
        )
        self.relu = nn.LeakyReLU()
    def forward(self, x):
        # 正常梯度下降
        y = self.block1(x)
        # 加上上次求梯度的结果
        return self.relu(x + y)

# 测试
rand_tensor = randn(1, 3, 28, 28)
channel_num = rand_tensor.size(dim=1)
res = residual_block(channel_num)(rand_tensor)
print(res)
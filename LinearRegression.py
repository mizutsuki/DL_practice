from torch.utils.tensorboard import SummaryWriter
from torch import tensor, nn, stack, reshape, optim
import torch
# 一个batch element =  3的数据集.
# x ∈ R(3 * 1)
x1 = tensor([[1], [2], [3]], dtype=torch.float)
# y ∈ R(3 * 1)
y = tensor([[2], [4], [6]], dtype=torch.float)
# 单单元神经网络
class LR(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # 一个神经元: w1
        self.ln = nn.Linear(1, 1)
    def forward(self, x):
        # 前向传播方法
        x = self.ln(x)
        return x
# 模型实例化
lr_mod = LR()
# 损失函数实例化
criterion = nn.MSELoss()
# 优化器实例化, parameter代表权重
sgd_op = optim.SGD(lr_mod.parameters(), lr=1e-3)
sw = SummaryWriter("runs/exp3")
# 多轮训练
for i in range(100):
    # 清零上一轮梯度
    sgd_op.zero_grad()
    # 计算损失
    loss = criterion(lr_mod(x1), y)
    # 反传(执行梯度计算过程)
    loss.backward()
    # 更新权重
    sgd_op.step()
    # 记录
    sw.add_scalar("loss", loss, i)
sw.close()

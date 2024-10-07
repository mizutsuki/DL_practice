from typing import Optional, Union, Iterable, List
from torch import nn, from_numpy, optim, tensor, randn, zeros, ones
from torch.utils.data import DataLoader, Dataset, Sampler
from torch.utils.data.dataloader import T_co, _collate_fn_t, _worker_init_fn_t
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
import numpy as np
class Myset(Dataset):
    # 初始化, 路径, 可选: 传入函数, 默认
    def __init__(self, path, transform: Optional[callable] = None):
        file_stream = np.loadtxt(path, delimiter=",", dtype=np.float32)
        self.feature = file_stream[:, :-1]
        self.transform = transform
        self.label = file_stream[:, [-1]]
        self.l = len(self.label)

    def __getitem__(self, index) -> T_co:
        if self.transform:
            self.feature = self.transform(self.feature)
            self.label = self.label[index]
        feature = self.feature[index]
        label = self.label[index]
        return feature, label
    def __len__(self):
        return self.l
path = "dataset/DIABETES/diabetes.csv"
train_ = Myset(path)
# 使用Dataloader加载数据
train_set = DataLoader(train_, batch_size=16)
# 构建网络
class Mluti_class(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # 对n个feature, 一个输出的二分类问题, 网络可设为为
        self.seq = nn.Sequential(
            # 使用两个sigmoid函数进行非线性空间变换
            nn.Linear(8, 64),
            nn.Sigmoid(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        x = self.seq(x)
        return x
# 实例化
model = Mluti_class()
# TODO 取一条数据测试
# test = convert_x[0]
# print(model(test))

# 损失函数
criterion = nn.BCELoss()
# 优化器
opt = optim.SGD(model.parameters(), lr=0.1)
# 训练
counter = 0
sw = SummaryWriter("runs/exp4")
for _ in range(6000):
    for x in train_set:
        feature, label = x
        # 梯度归零
        opt.zero_grad()
        # 预测计算
        pred = model(feature)
        # 计算损失(预测 vs 实际结果)
        loss = criterion(pred, label)
        # 计算梯度
        loss.backward()
        # 计算出的loss通过优化器更新到权重上
        opt.step()
        counter += 1
        if not counter % 1000:
        #     # if (pred > 0.5) = T && Y = T → pred == Y == T
        #     # if (pred < 0.5) = F && Y = F → pred == Y == T
            res = ((pred >= 0.5) == label)
            accuracy = res.sum()
            accuracy = accuracy/16
            print(accuracy.item(), loss)
            sw.add_scalar("accuracy", accuracy.item(), int(counter / 1000))
            sw.add_scalar("loss", accuracy.item(), int(counter / 1000))
sw.close()
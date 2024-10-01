import torch
from torch import nn, tensor, optim
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
# 假设有一张 3*32*32的图像
class Inception_a(nn.Module):
    def __init__(self, input_value: int) -> None:
        # 注意, 这里再子类中传参, 所以没有*args 和 *kwards
        super().__init__()
        # branch1 平均池化 + 卷积: 1 * 1 * 24
        self.branch1 = nn.Sequential(
            nn.AvgPool2d(kernel_size=(3, 3), stride=1, padding=1),
            nn.Conv2d(input_value, 24, kernel_size=(1, 1))
                        )
        # branch2 卷积 1 * 1 * 16
        self.branch2 = nn.Sequential(
            nn.Conv2d(input_value, 16, kernel_size=(1, 1))
        )
        # branch3 卷积: 1 * 1 * 16, 卷积: 5 * 5 *24
        self.branch3 = nn.Sequential(
            nn.Conv2d(input_value, 16, kernel_size=(1, 1)),
            nn.Conv2d(16, 24, kernel_size=(5, 5), padding=2)
        )
        # branch4
        self.branch4 = nn.Sequential(
            nn.Conv2d(input_value, 16, kernel_size=(1, 1)),
            nn.Conv2d(16, 24, kernel_size=(3, 3), padding=1),
            nn.Conv2d(24, 24, kernel_size=(3, 3), padding=1)
        )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        # merge
        merge_branch = torch.cat([branch1, branch2, branch3, branch4], dim=1)
        return merge_branch

class Net(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # seq1
        self.seq1 = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(3, 10, kernel_size=(7, 7), padding=1),
            nn.MaxPool2d(2),
            nn.LeakyReLU()
        )
        # seq2
        self.seq2 = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(88, 20, kernel_size=(5, 5), padding=1),
            nn.MaxPool2d(2),
            nn.LeakyReLU()
        )
        # cov层
        self.incep1 = Inception_a(input_value=10)
        self.incep2 = Inception_a(input_value=20)
        # fully connection
        self.full_c1 = nn.Linear(15680, 10)
    # 全连接
    def forward(self, x):
        batch_size = x.size(0)
        # 获取图像batch
        x = self.seq1(x)
        x = self.incep1(x)
        x = self.seq2(x)
        x = self.incep2(x)
        x = self.seq2(x)
        x = x.view(batch_size, -1)
        # print(x.shape)
        x = self.full_c1(x)
        return x

gn = Net()
train_img = torch.randn(1, 3, 32, 32)
res = gn(train_img)

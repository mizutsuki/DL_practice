import pandas as pd
import numpy as np
import torch
from torch import nn, optim
from torch import tensor
from torch.utils.data import DataLoader
from torchvision import transforms, models
from torchvision.datasets import CIFAR10
from PIL import Image
from torchvision.models import VGG16_Weights

img = Image.open("dataset/PILImage/cat.jpg")
# 实例化一个compose变换器
compose = transforms.Compose([
    # 任意图像变换器
    transforms.RandomRotation((0, 360)),
    transforms.RandomResizedCrop((224, 224), scale=(.4, 1.), ratio=(.1, 2.)),
    transforms.ColorJitter(brightness=(.5, 1.), hue=(-.5, .5)),
    transforms.ToTensor(),
])
img_out = compose(img).unsqueeze(dim=0)
img_out2 = compose(img).unsqueeze(dim=0)
# 设定一个
img_out = torch.cat((img_out, img_out2), dim=0)

# 实例化模型, 只需预训练模型的一部分
vgg16 = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features

# 继承
class vgg_ext(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # 插入自己构造的网络
        self.block1 = vgg16
        self.block2 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(25088, 2)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        return x

# 实例化模型
model = vgg_ext()

# Xavier normal distribution 初始化权重
nn.init.xavier_uniform_(model.block2[1].weight)

# 实例化优化器, 对非目标对象设置微小的学习率
opti1 = optim.SGD(model.block1.parameters(), lr=1e-5)
# 对微调目标对象设置较大的学习率
opti2 = optim.SGD(model.block2.parameters(), lr=0.01)

# 损失函数
criterion = nn.CrossEntropyLoss()
# 设置one-hot标签
label = tensor([1, 0], dtype=torch.int64)
# 图像输入
res = model(img_out)
# 损失计算
loss = criterion(res, label)
# 梯度清零
opti1.zero_grad()
opti2.zero_grad()
# bcakward
loss.backward()
# 更新
opti1.step()
opti2.step()
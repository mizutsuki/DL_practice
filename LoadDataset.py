from typing import Optional, Union, Iterable, List
import matplotlib.pyplot as plt
import pandas as pd
import os

import torch
from torchvision import io
from torch import tensor
from torch.utils.data import DataLoader, Dataset, Sampler
def banana(train):
    if train == True:
        # csv肚子鼓
        csv_address = "dataset/banana-detection/bananas_train/label.csv"
        # 获得训练集相对地址
        path_rel = "dataset/banana-detection/bananas_train/images/"
    else:
        csv_address = "dataset/banana-detection/bananas_val/label.csv"
        path_rel = "dataset/banana-detection/bananas_val/images/"
    # 在csv中含有 图像名,物体类别, 边缘框
    csv_data = pd.read_csv(csv_address, delimiter=',')
    csv_data = csv_data.set_index("img_name")
    # tensor图像矩阵
    tensor_image_set = []
    # 标签矩阵
    label = []
    # 迭代行, 读取行 index 和 val 数据
    # interrows(), 行索引 → str, 剩余内容 → numpySeries
    for name, val in csv_data.iterrows():
        path = os.path.join(path_rel, str(name))
        tensor_img = io.read_image(path)
        tensor_image_set.append(tensor_img)
        label.append(list(val))
    label = tensor(label, dtype=torch.float32).unsqueeze(dim=0)
    return tensor_image_set, label

# 定义一个data_set加载器
class banana_Data_Set(DataLoader):
    def __init__(self, istrain):
        super().__init__(istrain)
        self.img, self.label = banana(istrain)
    def __getitem__(self, idx):
        return self.img[idx].float(), self.label[idx]
    def __len__(self):
        return len(self.label)
# 实例化数据集
train_set = banana_Data_Set(istrain=True)
# 打印验证, batch === 1, 数量 == 1000条, feature == 5的数据
img, label = train_set[0]
# 取一条打印验证, class, X1, Y1, X2, Y2

# 查看3张数据集, 验证
# 子图
figure, ax = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
for x in range(3):
    path = rf"C:\Project\20240916Pytorch\dataset\banana-detection\bananas_train\images\{x}.png"
    # 获取数据集中的图片
    bbox = label[x]
    plt_img = plt.imread(path)
    # 中心点
    def get_anchor(tensor):
        x0 = tensor[1].data.item()
        y0 = tensor[2].data.item()
        return x0, y0
    def get_size(tensor):
        width = tensor[3] - tensor[1]
        height = tensor[4] - tensor[2]
        return width, height
    width, height = get_size(bbox)
    # 在plt上生成一个边框, 附加在子图上
    rect = plt.Rectangle(xy=get_anchor(bbox),
                         width=width, height=height,
                         fill=False,
                         edgecolor="red",
                         linewidth=3)
    ax[x].add_patch(rect)
    ax[x].imshow(plt_img)
# 展示
plt.show()
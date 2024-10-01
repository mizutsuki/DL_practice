import torch
from torch import nn, tensor, optim
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from GoogleNet import Net
from LeNet5 import LeNet5
# 数据集实例化
train_set = CIFAR10("dataset/train", download=False, transform=ToTensor())
test_set = CIFAR10("dataset/test",train=False, download=False, transform=ToTensor())
# TODO
# dataloader实例化, (train_set, test_set)
dataloader = DataLoader(train_set, batch_size=32, shuffle=True)
test_loader = DataLoader(test_set, batch_size=16)

# 模型实例化
google_net = LeNet5()
# 转移至cuda上
mod = google_net.cuda()
# 优化器实例化, 指定parameters && 学习率超参数
gsd_opt = optim.SGD(mod.parameters(), lr=0.01,weight_decay=0.0002)
# 指定损失函数
loss_func = nn.CrossEntropyLoss()
loss_func = loss_func.cuda()
# 实例化summary_writer
sw = SummaryWriter("runs/exp1")
# step
counter, test_counter = 0, 0

# 样本训练
for y in range(30):
    for x in dataloader:
        # 提取图像
        ipt, target = x
        ipt = ipt.cuda()
        target = target.cuda()
        # 对图像卷积
        res = mod(ipt)
        # 16个被flatten的batch的数据分别对应16个target
        batch_loss = loss_func(res, target)
        # 梯度归零0
        gsd_opt.zero_grad()
        # 利用损失函数得到的数据进行反向传播
        batch_loss.backward()
        # 进入下一轮
        gsd_opt.step()
        counter += 1
        if not counter % 500:
            # 记录batch loss
            step_counter = int(counter/500)
            sw.add_scalar("cross_entropy_loss", batch_loss, step_counter)

            print(f"当前子轮数: {step_counter} * 500 损失为: {batch_loss.item()}")
    # 将lr减少
    # 每轮训练结束后, 对样本的loss进行测试.
    correct_sample = 0
    # 无梯度模式, 加快计算速度
    with torch.no_grad():
        # 测试集 dataloader
        for test_data in test_loader:
            # 求损失
            test_img, test_target = test_data
            test_img, test_target = test_img.cuda(), test_target.cuda()
            test_img_calc = mod(test_img)
            # 计算一个batch中损失的情况, 与target对比
            # dim = 1, 代表计算每一行
            predict_res = test_img_calc.argmax(1)
            # 利用 布尔张量, 计算正确个数的数量
            single_correct_sample = (predict_res == test_target).sum()
            # 记录正确个数
            correct_sample += single_correct_sample
        # 所有样本结束一次训练后, 验证正确率
        accuracy = correct_sample / len(test_set)
        highest_accuracy = 0
        print(f"第: {test_counter}次样本循环, 正确率为: {accuracy}")
        print(f"正确样本数为{correct_sample}")
        # sw记录正确率
        sw.add_scalar("accuracy", accuracy, test_counter)
        test_counter += 1
        # 择优存储
        if accuracy > highest_accuracy:
            highest_accuracy = accuracy
            torch.save(mod.state_dict(), "Model/CIFAR10_class.pth")


sw.close()

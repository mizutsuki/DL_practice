import torch
# 训练集
pair = [["你好", "你好喵~"],
        ["你喜欢什么学校", "江西农业大学"],
        ["我是谁", "你是我的主人喵~"],
        ["你是谁", "我是猫娘女仆喵~"],
        ["今年学费收多少", "今年学费是八千元"],
        ["再见", "主人再见喵~"]]
# 自动提取样本, 编码预处理
def extract(x: list):
    pair = x
    # 合并
    content = ""
    for x in pair:
        content += x[0] + x[1]
    # 转tensor → one-hot编码
    counter = 0
    torch.empty(0, dtype=torch.int64)
    mapping = {'\0': 0}
    for i in content:
        # 使用哈希索引法, 加快处理速度
        if i not in mapping:
            counter += 1
            mapping[i] = counter

    # 初始化长整型做标签, 在数据很多时, 可以考虑用numpy + csv or 数据库
    # mapping 用于使用 One-hot标签, 查找数据, 预测输出,找tensor.max()
    return mapping

# 得到编码表, key为字符, value为对应索引
mapping = extract(pair)
# one-hot编码数, 传入数据集确定编码长度
input_size = len(mapping)
# hidden_size
hidden_size = input_size
# sequence_size
L = 8
# batch_size
N = 1
# 迭代次数, 最多生成8个字回复
num_iter = 8
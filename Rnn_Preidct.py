import torch
from Rnn_Net import RNN
from Rnn_DataSet import input_size, hidden_size, num_iter, mapping, L
from RNN_Train import str_encoder
# 加载模型
weight = torch.load("Model/chat_rnn.pth", weights_only=True)
rnn = RNN(input_size, hidden_size, num_iter)
rnn.load_state_dict(weight)

# 测试
print("欢迎使用聊天机器人, 输入0退出")
while (True):
    que = input("")
    if que == "" or len(que) > 8:
        print("不合法的输入, 要求0-8字")
    elif que == "0":
        exit(0)
    else:
        test_seq = str_encoder(que).view(L, 1, input_size)
        res = rnn(test_seq)
        _, i = torch.max(res, dim=2)
        # 拼接回复
        q = i.data
        response = ""
        # 反向创建dict哈希表
        rev_map = {v:k for k, v in mapping.items()}
        for ind in q:
            if ind != 0:
                response+=rev_map.get(ind.item())
        print(response)
import torch
from torch import tensor, nn
from typing import Optional
import numpy as np

# PositionWiseFFN(Feed-Forward Neural Network)全连接层.
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

# 层正则化, 针对每个hidden_layer进行正则化
# norm层
class Add_Norm(nn.Module):
    def __init__(self, features) -> None:
        super().__init__()
        self.layer_norm = nn.LayerNorm(features)
        self.dp = nn.Dropout(.1)
    def forward(self, x):
        # 接受上一层的输入 + 本层的正则化
        x = x + self.layer_norm(x)
        return self.dp(x)

# DotATT (Q*K.T/sqrt(d)) * V
class DotAtt(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # 新版本不支持随机随机dropout
        self.sftmax = nn.Softmax(dim=-1)

    # softmax遮罩, 可以选择性遮蔽下文文字(对seq_len起效).
    def softmax_masker(self, x, valid_len:Optional = None):
        if valid_len is None:
            return self.sftmax(x)
        else:
            if len(x.shape) == 3:
                # TODO
                valid_len = (valid_len if torch.is_tensor(valid_len) else tensor(valid_len))
                aim_mask = valid_len.reshape(-1)
                # 获取shape
                x_shape = x.shape
                # 变为2Dtensor
                x = x.reshape(-1, x.shape[-1])
                # 插入
                aim_mask = torch.repeat_interleave(aim_mask, x_shape[-1])
                # 从flatten重建 n * hidden_layer
                aim_mask = aim_mask.reshape(-1, x_shape[-1])
                aim_mask = torch.cat([aim_mask]*x_shape[0])
                # indices_matrix
                indices_matrix = tensor(list(range(x.shape[-1]))).expand(x.shape)
                # masker, 索引 > indices的设为true, 最小为index = 1之后全部遮蔽
                masker = indices_matrix > aim_mask
                x[masker] = -1e6
                # 恢复形状
                x = x.reshape(x_shape)
                return self.sftmax(x)
            else:
                print("not 3D-tensor, use a normal softmax instead")
                return self.sftmax(x)

    def forward(self, K, V, Q, valid_len:Optional = None):
        # Q = q * h, K = k*h, V = v*h && v = q
        # (Q*K.T/sqrt(d)) * V && matrix_shape = q*k@v*h → q*h
        dim = K.shape[-1]
        score = Q @ K.transpose(1, 2) / (dim ** .5)
        att = self.softmax_masker(score @ V, valid_len)
        return att

# MultiHead_attention
class MulAtt(nn.Module):
    # 显式输入, 隐藏层尺寸, 头数
    def __init__(self, q_hidden, k_hidden, v_hidden,
                 num_hidden) -> None:
        super().__init__()
        # hidden_layer
        self.num_hidden = num_hidden
        # 点积注意力
        self.attion = DotAtt()
        # 输入尺寸归一化, q && k && v_hidden_size → hidden_size
        self.w_q = nn.Linear(q_hidden, num_hidden)
        self.w_k = nn.Linear(k_hidden, num_hidden)
        self.w_v = nn.Linear(v_hidden, num_hidden)
        # 为融合输出增加线性.
        self.w_o = nn.Linear(num_hidden, num_hidden)

    # transforrmer多头输入预处理
    def transpose_qkv(self, x, num_heads):
        # 将最后的hidden_layer拆分,
        x = x.reshape(x.shape[0], x.shape[1], num_heads, -1)
        # batch, 头数, seq_len, head_dim
        x = x.permute(0, 2, 1, 3)
        # 头数与batch合并, 变成batch*头数, seq_len不变, head_num不变
        # 可以使每个头关注不同的信息(已做embedding), 使感受野变为2D
        return x.reshape(-1, x.shape[2], x.shape[3])

    # transformer输出处理
    def transpose_out(self, x, num_heads):
        x = x.reshape(-1, num_heads, x.shape[1], x.shape[2])
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        return x

    def forward(self, K, V, Q, head, valid_len:Optional = None):
        self.num_head = head
        # 增加非线性变换, 统一输出形状
        K = self.w_k(K)
        V = self.w_v(V)
        Q = self.w_q(Q)
        # 转变为 matrix_shape = (head, seq, head_layer(hidden_layer被拆开))
        trans_k = self.transpose_qkv(K, self.num_head)
        trans_v = self.transpose_qkv(V, self.num_head)
        trans_q = self.transpose_qkv(Q, self.num_head)
        # k, v, q, encoder时可以不遮蔽
        out_put = self.attion(trans_k, trans_v, trans_q, valid_len)
        # transform还原输入的张量
        out_cat = self.transpose_out(out_put, self.num_head)
        # 换增加非线性
        return self.w_o(out_cat)

# 单一EncoderBlock
class EconderBlock(nn.Module):
    def __init__(self, q_hidden, k_hidden, v_hidden, num_hidden,
                 ffn_num_hidden, head) -> None:
        super().__init__()
        self.head_num = head
        self.mulHeadAtt = MulAtt(q_hidden, k_hidden, v_hidden, num_hidden)
        # normalize for each_layer
        features = num_hidden
        # 实例化两个add_norm层
        self.add_norm = Add_Norm(features)
        self.add_norm2 = Add_Norm(features)
        # feed_forward
        ffn_num_input = num_hidden
        ffn_num_out = num_hidden
        self.feed_forward = PositionWiseFFN(ffn_num_input, ffn_num_hidden, ffn_num_out)
    def forward(self, x, valid_len:Optional = None):
        # 编码器可不遮蔽,多头注意直接将x同时作为key,value,query
        # 在编码器自注意力机制中, k, v, q == x
        out1 = self.mulHeadAtt(x, x, x, self.head_num, valid_len)
        res1 = self.add_norm(out1)
        # sub_layer2
        fusion = out1 + res1
        out2 = self.add_norm(self.feed_forward(fusion))
        res2 = self.add_norm2(fusion) + out2
        return out2 + res2
# 位置编码 PE(pos, 2i/2i+1) = sin/cos(pos/10000^(2i/d))

# 如果到达最大长度, 可以选择类重载
class PositionalEconding(nn.Module):
    # parameter: 编码长度, hidden层
    def __init__(self, max_coding_len, num_hidden) -> None:
        super().__init__()
        self.max_len = max_coding_len
        # 位置编码
        pos = tensor(list(range(max_coding_len))).reshape(-1, 1)
        # step = 2
        i = tensor(list(range(0, num_hidden, 2)), dtype=torch.float32)
        # 位置编码矩阵, 1用于广播
        self.pos_encod_mat = torch.zeros(1, max_coding_len, num_hidden)
        # 正弦/余弦编码(一次性计算所有pos的编码)
        sin_mat =torch.sin(pos/10000**(i/num_hidden))
        cos_mat = torch.cos(pos/10000**(i/num_hidden))
        self.pos_encod_mat[:, :, 0::2] = sin_mat
        self.pos_encod_mat[:, :, 1::2] = cos_mat
    def forward(self, x):
        # 最大附加位置编码只到max_len位置
        x = x + self.pos_encod_mat[:, : x.shape[1], :]
        return x

# TransformerEncoder
class TransformerEncoder(nn.Module):
    # 初始化: 子层1: 字典容量, 嵌入层尺寸, 序列长度, 隐藏层大小, ffn隐层, layer
    def __init__(self, dict_capacity, embedding_dim,
                 # maxtirx_size[-1] to num_hidden
                 q_hidden, k_hidden, v_hidden, num_hidden,
                 # 编码长度, 头数, ffn_隐层, transformer_encoder数量
                 max_coding_len, head, ffn_hidden_number, layer_num: int
                 ) -> None:
        super().__init__()
        # 隐藏层参数
        self.num_hidden = num_hidden
        # 头数
        self.num_head = head
        # 初始化嵌入层, 对输入文字编码
        self.embedding = nn.Embedding(num_embeddings=dict_capacity, embedding_dim=embedding_dim)
        # 位置编码, 输出: embedding + position
        self.pos_encod = PositionalEconding(max_coding_len, num_hidden)
        # 单个encoder层
        self.encoderBlock = EconderBlock(q_hidden, k_hidden, v_hidden, num_hidden,
                                         ffn_hidden_number, self.num_head)
        # nn.Sequential放置多个encoder层
        self.multi_encoderBlock = nn.Sequential()
        for i in range(layer_num):
            self.multi_encoderBlock.add_module(f"layer: {i}",
                                               self.encoderBlock)
    # 对任意输入embedding
    def forward(self, x, valid_lenth:Optional = None):
        # 转embedding编码, 对其适当放大, 因为p越大,
        x = self.embedding(x) * self.num_hidden**.5
        # 对自注意力计算后的值附加位置编码
        x = self.pos_encod(x)
        # 因为其中的模块接受多个参数, 所以需要循环赋值
        for unpacked_encoderBlock in self.multi_encoderBlock:
            # 拿到一个实例计算的x, 再赋值给下一个x
            x = unpacked_encoderBlock(x, valid_lenth)
        # x同时做q, k, v
        return x
# DecodeBlock 单层解码器
class DecoderBlock(nn.Module):
    def __init__(self,  q_hidden, k_hidden, v_hidden, num_hidden,
                 ffn_num_hidden, head, i) -> None:
        super().__init__()
        # 头数
        self.num_head = head
        # 开关
        self.i = i
        # 注意力模块
        self.attention1 = MulAtt(q_hidden, k_hidden, v_hidden, num_hidden)
        # 正则化
        self.addnorm1_1 = Add_Norm(num_hidden)
        self.addnorm1_2 = Add_Norm(num_hidden)
        # 下层Attention从OutputEmbedding中取数据.
        self.attention2 = MulAtt(q_hidden, k_hidden, v_hidden, num_hidden)
        # 正则化2
        self.addnorm2_1 = Add_Norm(num_hidden)
        self.addnorm2_2 = Add_Norm(num_hidden)
        # positionwiseFFN
        self.positionwiseFFN = PositionWiseFFN(num_hidden, ffn_num_hidden, num_hidden)
        self.addnorm3_1 = Add_Norm(num_hidden)
        self.addnorm3_2 = Add_Norm(num_hidden)
    def forward(self, x, state,  valid_len:Optional = None):
        # 接受编码器的输出, 编码器的遮罩
        enc_outputs, enc_valid_lens = state[0], state[1]
        # state处理:
        if state[2][self.i] is None:
            key_value_capacitor = x
        else:
            # matrix_size = batc, seq_len, hidden
            # 实现联系前文
            key_value_capacitor = torch.cat((state[2][self.i], x), dim=1)
        if self.training:
            batch_size, num_steps, _ = x.shape
            # 对batch中第1...n行元素, 遮蔽1...n之后的值, 是一个三角矩阵(因果掩码)
            dec_valid_lens = torch.arange(0, num_steps).repeat(batch_size, 1)
            # TODO
        else:   # evaluate模式时
            # 测试模式, 非遮挡
            dec_valid_lens = None
        # sub_layer1: Q, K, V, 头数, 遮罩层输入
        x1 = self.attention1(x, key_value_capacitor, key_value_capacitor, self.num_head, dec_valid_lens)
        # layer_norm
        y = self.addnorm1_1(x1) + self.addnorm1_2(x)
        # sub_layer2: 输入的q由解码器提供, k&&v由编码器提供
        y1 = self.attention2(y, enc_outputs, enc_outputs, self.num_head, enc_valid_lens)
        z = self.addnorm2_1(y) + self.addnorm2_2(y1)
        # sub_layer3: FFN层
        ffn_out = self.positionwiseFFN(z)
        return self.addnorm3_1(z) + self.addnorm3_2(ffn_out), state

class TransformerDecoder(nn.Module):
    # 初始化: 子层1: 字典容量, 嵌入层尺寸, 序列长度, 隐藏层大小, ffn隐层, layer
    def __init__(self, dict_capacity, embedding_dim,
                 # maxtirx_size[-1] to num_hidden
                 q_hidden, k_hidden, v_hidden, num_hidden,
                 # 编码长度, 头数, ffn_隐层, transformer_decoder数量,
                 max_coding_len, head, ffn_hidden_number, layer_num: int,
                 ) -> None:
        super().__init__()
        # decoder层数
        self.layer_num = layer_num
        # 隐藏层参数
        self.num_hidden = num_hidden
        # 头数
        self.num_head = head
        # 初始化嵌入层, 对输入文字编码
        self.embedding = nn.Embedding(num_embeddings=dict_capacity, embedding_dim=embedding_dim)
        # 位置编码, 输出: embedding + position
        self.pos_encod = PositionalEconding(max_coding_len, num_hidden)
        # 使用sequential防止多个decoder层
        self.multi_decoderBlock = nn.Sequential()
        for i in range(layer_num):
            self.multi_decoderBlock.add_module(f"layer: {i}",
                                               DecoderBlock(q_hidden, k_hidden, v_hidden, num_hidden,
                                                            ffn_hidden_number, self.num_head, i))
        # 从embedding中还原为字典编码输出.
        self.dense = nn.Linear(num_hidden, dict_capacity)

    # 解码器初始, 接收n个编码器输出, 编码器遮罩, 返回state数组
    def init_state(self, enc_outputs, enc_valid_lens:Optional = None, *args):
        # 预测输出存储
        return [enc_outputs, enc_valid_lens, [None]*self.layer_num]

    # 形参1: token → embedding输入, 形参2: encoder → decoder_init_state → decoder
    def forward(self, x, state):
        # 索引编码 → embedding编码
        x = self.embedding(x) * self.num_hidden**.5
        # 附加位置编码
        x = self.pos_encod(x)
        # n层decoder循环赋值
        for single_decode_layer in self.multi_decoderBlock:
            x, state = single_decode_layer(x, state)
        # 展开
        x = self.dense(x)
        return x, state
        pass

# encoder的输出 → decoder输入1 && decoderk, v, q输入1
class Enco_Deco(nn.Module):
                # 字典长度, 嵌入层维度
    def __init__(self, dict_capacity, embedding_dim,
                 q_hidden, k_hidden, v_hidden, num_hidden,
                 # 编码长度, 头数, ffn_隐层, transformer_decoder数量
                 max_coding_len, head, ffn_hidden_number, layer_num: int,
                 ) -> None:
        super().__init__()
        self.tsfEncoder = TransformerEncoder(dict_capacity, embedding_dim,
                                        # k, v, q_hidden, num_hidden
                                        q_hidden, k_hidden, v_hidden, num_hidden,
                                        # 位置编码最大长度, head, ffn
                                        max_coding_len, head, ffn_hidden_number, layer_num)
        self.tsfDecoder = TransformerDecoder(dict_capacity, embedding_dim,
                                             # maxtirx_size[-1] to num_hidden
                                             q_hidden, k_hidden, v_hidden, num_hidden,
                                             # 编码长度, 头数, ffn_隐层, transformer_decoder数量,
                                             max_coding_len, head, ffn_hidden_number, layer_num)
        self.sft = nn.Softmax(dim=-1)
    def forward(self, x):
        y = x
        x = self.tsfEncoder(y)
        state = self.tsfDecoder.init_state(x)
        # decoder也要重新编码
        print(x.shape)
        z, _ = self.tsfDecoder(y, state)
        return z

# 注意: embedding_width == hidden_width
seq = tensor([[0, 1]], dtype=torch.int64)
ed = Enco_Deco(2, 8, 8, 8, 8, 8, 1000, 2, 4, 2)
res = ed(seq)
print(res.shape)
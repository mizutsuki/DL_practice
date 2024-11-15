from transformers import BertTokenizer
from typing import Optional

# 实例化分词器, 可以将字符串进行语义划分
tokens = BertTokenizer.from_pretrained("bert-base-chinese", local_files_only=True)
# CLS
strat = 101
# SEP
end = 102
# Padding
padding_token = 102
# 这里展示使用分词器的方法, 也可以逐字编码
def str_encoder(chars,  seq_max_len, padding_token):
    chars = tokens(chars).get("input_ids")
    # 截断 or 填充 文本序列
    if len(chars) > seq_max_len:
        reg_chars = tokens[:seq_max_len]
        return reg_chars
    else:
        pad_len = seq_max_len - len(chars)
        reg_chars = chars + [padding_token] * pad_len
        # print(reg_chars)
        return reg_chars

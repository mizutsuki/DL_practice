# 普通字符转列表:
list_chars = list("你好世界")
list_chars = ["<Start>"] + list_chars + ["<End>"]*4

# 经过填充的字符一定宽相等了
def cvrt_2_batch(coding_list, batch, seq_len):
    batch_list = []
    counter = 0
    for x in range(batch):
        batch_list.append(coding_list[counter:counter+seq_len])
    return batch_list
res = cvrt_2_batch(list_chars, 4, 6)
print(res)
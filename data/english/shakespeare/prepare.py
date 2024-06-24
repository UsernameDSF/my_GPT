import os
import tiktoken
import numpy as np
'''
在字符级别处理数据集
'''

input_file_path = os.path.join(os.path.dirname(__file__), 'shakespeare.txt')

with open(input_file_path, 'r', encoding='utf-8') as f:
    data = f.read()
n = len(data)
train_data = data[:int(n*0.9)]
val_data = data[int(n*0.9):]

chars = sorted(list(set(data)))
vocab_size = len(chars)
# 创建一个字符映射到id和id映射到字符的字典
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
# 根据上面的映射，编写编码解码函数
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])


train_ids = encode(train_data)
val_ids = encode(val_data)

# export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))

# train.bin has 301,966 tokens
# val.bin has 36,059 tokens
import os

'''
在字符级别处理数据集
'''

input_file_path = os.path.join(os.path.dirname(__file__), 'Xiyou.txt')
with open(input_file_path, 'r', encoding='utf-8') as f:
    data = f.read()
n = len(data)
train_data = data[:int(n*0.9)]
val_data = data[int(n*0.9):]

chars = sorted(list(set(data)))
vocab_size = len(chars)
print(vocab_size)
# 创建一个字符映射到id和id映射到字符的字典
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
# 根据上面的映射，编写编码解码函数
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])


train_ids = encode(train_data)
val_ids = encode(val_data)
with open('train.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(map(str, train_ids)))
with open('val.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(map(str, val_ids)))

print('train has', len(train_ids), 'tokens')
print('val has', len(val_ids), 'tokens')
'''
train has 649,892 tokens
val has 72,211 tokens
'''

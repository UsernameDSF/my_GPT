import os
import tiktoken

input_file_path = os.path.join(os.path.dirname(__file__), 'shakespeare.txt')

with open(input_file_path, 'r', encoding='utf-8') as f:
    data = f.read()
n = len(data)
train_data = data[:int(n*0.9)]
val_data = data[int(n*0.9):]

enc = tiktoken.get_encoding("gpt2")

train_ids = enc.encode_ordinary(train_data)
val_ids = enc.encode_ordinary(val_data)
with open('shakespeare/train.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(map(str, train_ids)))
with open('shakespeare/val.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(map(str, val_ids)))

'''
train has 301,966 tokens
val has 36,059 tokens
'''

import torch
import tiktoken
from torch.utils.data import DataLoader, Dataset
import os
import random
import numpy as np

# 模型参数设置位置！
class ModelArgs:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.block_size = 256  # 窗口大小GPT2为1024
        self.batch_size = 16  # 暂定，之后再看显存占用
        self.n_layer = 6
        self.vocab_size = 65
        self.n_head = 6
        self.n_embed = 384
        self.bias = False
        self.dropout = 0.2
        self.dataset_path = r'D:\PythonProject\my_GPT\data\english\shakespeare'
        self.init_from = 'scratch'  # 'scratch' or 'resume' # 从头训练还是继续
        self.checkpoint_save_dir = r'D:\PythonProject\my_GPT\checkpoint\nanogpt'
        self.eval_step = 10  # 每n步eval和保存checkpoint一次
        self.flash_attn = False
        # 学习率衰减
        self.learning_rate = 0.001
        self.warmup_steps = 100
        self.lr_decay_steps = 5000  # 这个意思是，到lr_decay_steps之后，学习率就不再衰减了。一般与训练总步数一样，因此学习率会一直衰减。
        self.min_lr = 0.0001
        # 优化器参数
        self.max_epochs = 5  # 训练多少个epoch
        # self.weight_decay = 1e-1
        # self.betas = (0.9,0.95)
        self.grad_clip = 1.0  # 梯度裁剪,固定阈值进行裁剪。设置为0.0就是关闭
        self.compile = False




args = ModelArgs()

file_path = os.path.join(args.dataset_path, 'shakespeare.txt')

with open(file_path, 'r', encoding='utf-8') as f:
    texts = f.read()

chars = sorted(list(set(texts)))
vocab_size = len(chars)
args.vocab_size = vocab_size


stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string


'''
将训练文本划分成不重复的文本段来训练
'''
# class MyDataset(Dataset):
#     def __init__(self, method):
#         super().__init__()
#         if method == 'train':
#             train_data_path = os.path.join(args.dataset_path, 'train.txt')
#             with open(train_data_path, 'r', encoding='utf-8') as f:
#                 text = f.read()
#                 text = [int(item) for item in text.split('\n')]
#             self.data = text
#
#         elif method == 'val':
#             val_data_path = os.path.join(args.dataset_path, 'val.txt')
#             with open(val_data_path, 'r', encoding='utf-8') as f:
#                 text = f.read()
#                 text = [int(item) for item in text.split('\n')]
#             self.data = text
#
#         self.block_size = args.block_size
#         self.start = [i for i in range(0, len(self.data) - args.block_size, args.block_size)]
#
#     def __len__(self):
#         return len(self.start)
#
#     def __getitem__(self, i):
#         start_idx = self.start[i]
#         end_idx = start_idx + self.block_size
#
#         x = torch.LongTensor(self.data[start_idx: end_idx])
#         y = torch.LongTensor(self.data[start_idx + 1: end_idx + 1])
#         return x, y

'''
从训练文本中随机抽取文本段来训练
'''
class MyDataset(Dataset):
    def __init__(self, method):
        super().__init__()
        self.method = method
        if method == 'train':
            self.data = np.memmap(os.path.join(args.dataset_path, 'train.bin'), dtype=np.uint16, mode='r')
        else:
            self.data = np.memmap(os.path.join(args.dataset_path, 'val.bin'), dtype=np.uint16, mode='r')

    # 指定训练集，验证集大小
    def __len__(self):
        if self.method == 'train':
            return 16000
        else:
            return 160

    def __getitem__(self, idx):
        i = random.randint(0, len(self.data) - args.block_size - 1)
        input_x = np.copy(self.data[i:i + args.block_size])
        input_y = np.copy(self.data[i + 1:i + args.block_size + 1])
        x = torch.LongTensor(input_x)
        y = torch.LongTensor(input_y)
        return x, y


# val_loader = DataLoader(MyDataset('val'), batch_size=args.batch_size, shuffle=True, drop_last=True)
# train_loader = DataLoader(MyDataset('train'), batch_size=args.batch_size, shuffle=True, drop_last=True)
# print(next(iter(train_loader)))
# print(os.path.join(args.dataset_path, 'train.txt'))
# print(os.path.exists(os.path.join(args.dataset_path, 'train.txt')))
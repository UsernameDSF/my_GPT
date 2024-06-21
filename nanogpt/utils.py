import torch
import tiktoken
from torch.utils.data import DataLoader, Dataset
import os
import random


# 模型参数设置位置！
class ModelArgs:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.block_size = 256  # 窗口大小GPT2为1024
        self.batch_size = 16  # 暂定，之后再看显存占用
        self.n_layer = 6
        self.vocab_size = 50304  # gpt2 tokenizer词表大小，使用的gpt2的分词器
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
        self.lr_decay_steps = 5110
        self.min_lr = 0.0001
        # 优化器参数
        self.max_epochs = 5  # 训练多少个epoch
        # self.weight_decay = 1e-1
        # self.betas = (0.9,0.95)
        # self.grad_clip = 1.0 # 梯度裁剪


args = ModelArgs()
enc = tiktoken.get_encoding("gpt2")
decode = lambda x: enc.decode(x)
encode = lambda x: enc.encode(x, allowed_special={"<|endoftext|>"})


# en = encode('print("<|endoftext|>")')
# print(en)
# print(decode(en))


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


class MyDataset(Dataset):
    def __init__(self, method):
        super().__init__()
        self.method = method
        if method == 'train':
            train_data_path = os.path.join(args.dataset_path, 'train.txt')
            with open(train_data_path, 'r', encoding='utf-8') as f:
                text = f.read()
                text = [int(item) for item in text.split('\n')]
            self.data = text

        elif method == 'val':
            val_data_path = os.path.join(args.dataset_path, 'val.txt')
            with open(val_data_path, 'r', encoding='utf-8') as f:
                text = f.read()
                text = [int(item) for item in text.split('\n')]
            self.data = text

    def __len__(self):
        if self.method == 'train':
            return 16000
        else:
            return 160

    def __getitem__(self, idx):
        i = random.randint(0, len(self.data) - args.block_size - 1)
        x = torch.LongTensor(self.data[i:i + args.block_size])
        y = torch.LongTensor(self.data[i + 1:i + args.block_size + 1])
        return x, y


# val_loader = DataLoader(MyDataset('val'), batch_size=args.batch_size, shuffle=True, drop_last=True)
# train_loader = DataLoader(MyDataset('train'), batch_size=args.batch_size, shuffle=True, drop_last=True)

# print(os.path.join(args.dataset_path, 'train.txt'))
# print(os.path.exists(os.path.join(args.dataset_path, 'train.txt')))

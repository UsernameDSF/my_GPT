import torch
import tiktoken
from torch.utils.data import DataLoader, Dataset
import os
import random
import numpy as np
from model import  GPT
# 模型参数设置位置！
class ModelArgs:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.block_size = 256  # 窗口大小GPT2为1024
        self.batch_size = 8  # 暂定，之后再看显存占用
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
        # self.warmup_steps = 100
        # self.lr_decay_steps = 5000  # 这个意思是，到lr_decay_steps之后，学习率就不再衰减了。一般与训练总步数一样，因此学习率会一直衰减。
        self.min_lr = 0.0001
        # 优化器参数
        self.max_epochs = 5  # 训练多少个epoch
        # self.weight_decay = 1e-1
        # self.betas = (0.9,0.95)
        self.grad_clip = 1.0  # 梯度裁剪,固定阈值进行裁剪。设置为0.0就是关闭
        self.compile = False




args = ModelArgs()

enc = tiktoken.get_encoding("gpt2")
encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
decode = lambda l: enc.decode(l)

'''
从训练文本中随机抽取文本段来训练
'''
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, method):
        super().__init__()
        self.method = method
        if method == 'train':
            self.data = np.memmap(os.path.join(args.dataset_path, 'train.bin'), dtype=np.uint16)
        else:
            self.data = np.memmap(os.path.join(args.dataset_path, 'val.bin'), dtype=np.uint16)

    # 指定训练集，验证集大小
    def __len__(self):
        if self.method == 'train':
            return 16000
        else:
            return 160

    def __getitem__(self, idx):
        i = random.randint(0, len(self.data) - args.block_size - 1)
        x = torch.from_numpy((self.data[i:i + args.block_size]).astype(np.int64))
        y = torch.from_numpy((self.data[i + 1:i + args.block_size + 1]).astype(np.int64))
        return x, y


# val_loader = DataLoader(MyDataset('val'), batch_size=args.batch_size, shuffle=True, drop_last=True)
# train_loader = DataLoader(MyDataset('train'), batch_size=args.batch_size, shuffle=True, drop_last=True)
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
        self.block_size = 1024  # 窗口大小GPT2为1024
        self.batch_size = 24  # 暂定，之后再看显存占用
        self.n_layer = 12
        self.vocab_size = 50304 # 50257四舍五入以提高效率
        self.n_head = 12
        self.n_embed = 768
        self.bias = False
        self.dropout = 0  # 对于预训练0是好的，对于微调尝试0.1+
        self.dataset_path = '/data/openwebtext'
        self.init_from = None  # 从头训练还是在已训练的模型上继续
        self.checkpoint_save_dir = '/nvme/file_of_yl/my_GPT/checkpoint/nanogpt'
        self.eval_step = 200  # 每n步eval和保存checkpoint一次
        self.flash_attn = True
        # 学习率衰减
        self.learning_rate = 6e-4
        # self.warmup_steps = 100
        # self.lr_decay_steps = 5000  # 这个意思是，到lr_decay_steps之后，学习率就不再衰减了。一般与训练总步数一样，因此学习率会一直衰减。
        self.min_lr = 6e-5
        # 优化器参数
        self.max_epochs = 1  # 训练多少个epoch
        # self.weight_decay = 1e-1
        # self.betas = (0.9,0.95)
        self.grad_clip = 1.0  # 梯度裁剪,固定阈值进行裁剪。设置为0.0就是关闭
        self.compile = True
        self.resume = '/nvme/file_of_yl/my_GPT/checkpoint/nanogpt/step_99000'
        self.gradient_accumulation_steps = 48 # 模拟每个gpu达到480的batch_size




args = ModelArgs()


'''
从训练文本中随机抽取文本段来训练
'''
class MyDataset(Dataset):
    def __init__(self, method):
        super().__init__()
        self.method = method
        if method == 'train':
            self.data = np.memmap(os.path.join(args.dataset_path, 'train.bin'), dtype=np.uint16)
        else:
            self.data = np.memmap(os.path.join(args.dataset_path, 'val.bin'), dtype=np.uint16)

    # 指定训练集，验证集大小
    def __len__(self):
        if self.method == 'train':
            return 100000*24*6    # 每张gpus各跑60w步
        else:
            return 24*200*6 # batch_size*steps*gpus 在这里相当于每个gpu拿2000份数据测试

    def __getitem__(self, idx):
        i = random.randint(0, len(self.data) - args.block_size - 1)
        x = torch.from_numpy((self.data[i:i + args.block_size]).astype(np.int64))
        y = torch.from_numpy((self.data[i + 1:i + args.block_size + 1]).astype(np.int64))
        return x, y


# val_loader = DataLoader(MyDataset('val'), batch_size=args.batch_size, shuffle=True, drop_last=True)
# train_loader = DataLoader(MyDataset('train'), batch_size=args.batch_size, shuffle=True, drop_last=True)

# print(os.path.join(args.dataset_path, 'train.txt'))
# print(os.path.exists(os.path.join(args.dataset_path, 'train.txt')))
# print(os.path.join(args.dataset_path, 'train.txt'))
# print(os.path.exists(os.path.join(args.dataset_path, 'train.txt')))
m = GPT(args)
m1 = m.transformer
print(m1)

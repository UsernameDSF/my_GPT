from nanogpt.model import GPT
from nanogpt.utils import args,encode,decode
import torch
import os

ckpt_path = os.path.join(args.checkpoint_save_dir, 'best_checkpoint.pt')  # 读取checkpoint路径
checkpoint = torch.load(ckpt_path, map_location=args.device)
checkpoint_model_args = checkpoint['model_args']  # 从checkpoint里面读取模型参数

model = GPT(checkpoint_model_args).to(checkpoint_model_args.device)
state_dict = checkpoint['model']  # 读取模型权重
model.load_state_dict(state_dict)

# generate参数
top_k = 2
tempreture = 1 # 一般都先设置1，想要更random一点就往上调
# num_samples = 1 # sample几次
max_new_tokens = 200

model.eval()
model.to(args.device)

start = "Sherlock Homes"  # 这是最开始的输入
start_ids = encode(start)
x = torch.tensor(start_ids,dtype=torch.long, device=args.device).unsqueeze(0)
idx = model.generate(x, max_new_tokens=max_new_tokens, temperature=tempreture, top_k=top_k)

print(decode(idx[0].tolist()))
from nanogpt.model import GPT
from nanogpt.utils import MyDataset, args
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

best_val_loss = 1e9

assert args.init_from == 'scratch' or args.init_from == 'resume'
if args.init_from == 'scratch':
    print("从头训练模型")
    model = GPT(args).to(args.device)
elif args.init_from == 'resume':
    print("继续训练模型")
    ckpt_path = os.path.join(args.checkpoint_save_dir, 'best_checkpoint.pt')  # 读取checkpoint路径
    checkpoint = torch.load(ckpt_path, map_location=args.device)
    checkpoint_model_args = checkpoint['model_args']  # 从checkpoint里面读取模型参数

    model = GPT(checkpoint_model_args).to(checkpoint_model_args.device)
    state_dict = checkpoint['model']  # 读取模型权重
    model.load_state_dict(state_dict)

    best_val_loss = checkpoint['best_val_loss']
    checkpoint = None  # 释放checkpoint

if args.compile:
    model = torch.compile(model)
    print('使用了torch.compile！')

optim = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
# sched = torch.optim.lr_scheduler.StepLR(optim, step_size=2, gamma=0.5)
val_loader = DataLoader(MyDataset('val'), batch_size=args.batch_size, shuffle=True, drop_last=True)
train_loader = DataLoader(MyDataset('train'), batch_size=args.batch_size, shuffle=True, drop_last=True)

# 初始化混合精度训练
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 检查cuda是否支持bfloat16数据类型
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = torch.amp.autocast(device_type=args.device, dtype=ptdtype)  # torch.amp.autocast混合精度
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))  # 优化：混合精度训练，大部分使用float16，少部分用float32



print('开始训练')
total_steps = len(train_loader) * args.max_epochs
step = 0  # 初始化步数计数器
with tqdm(total=total_steps, desc=f"Training ", unit="step") as pbar:
    while step < total_steps:
        for x, y in train_loader:
            # 更新学习率
            optim.param_groups[0]['lr'] = model.get_lr(step)
            with ctx:
                x, y = x.to(args.device), y.to(args.device)
                logits, loss = model(x, y)
                scaler.scale(loss).backward()
            # 梯度裁剪
            if args.grad_clip != 0.0:
                scaler.unscale_(optim)  # 使用标准，在进行梯度裁剪前，要使用这段代码，unscale梯度回fp32
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)  # 固定阈值剪裁

            scaler.step(optim)
            optim.zero_grad(set_to_none=True)

            step += 1  # 每完成一个批次，步数增加
            pbar.update(1)  # 更新进度条

            # 每args.eval_step步，就计算下验证集的loss
            if step > 0 and step % args.eval_step == 0:
                model.eval()
                val_losses = 0
                for x, y in val_loader:
                    x, y = x.to(args.device), y.to(args.device)
                    _, loss = model(x, y)
                    val_losses += loss.item()
                val_losses = val_losses / len(val_loader)
                lr = optim.param_groups[0]['lr']
                print(f"\n当前进行了{step}步,当前学习率：{lr}, train_loss:{loss.item()},val_loss:{val_losses}")

                model.train()
                # 保存checkpoint
                if val_losses < best_val_loss:
                    best_val_loss = val_losses
                    checkpoint = {
                        'model': model.state_dict(),
                        'optimizer': optim.state_dict,
                        'model_args': model.args,
                        'best_val_loss': best_val_loss
                    }
                    torch.save(checkpoint, os.path.join(args.checkpoint_save_dir, 'best_checkpoint.pt'))
                    print(f"checkpoint保存在{args.checkpoint_save_dir}/best_checkpoint.pt")

            # # 如果达到总步数，结束训练
            # if step >= total_steps:
            #     print(f"已达到总步数 {total_steps}，训练结束。")
            #     break

        # 一轮迭代结束，检查是否需要继续训练
        if step < total_steps:
            # 重新初始化数据加载器以开始新的epoch
            data_loader = DataLoader(MyDataset('train'), batch_size=args.batch_size, shuffle=True, drop_last=True)
        else:
            # 达到总步数，跳出外部循环
            print(f"已达到总步数 {total_steps}，训练结束。")
            break
    # sched.step()
print('结束！')
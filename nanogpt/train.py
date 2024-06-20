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

optim = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
sched = torch.optim.lr_scheduler.StepLR(optim, step_size=2, gamma=0.5)
val_loader = DataLoader(MyDataset('val'), batch_size=args.batch_size, shuffle=True, drop_last=True)
train_loader = DataLoader(MyDataset('train'), batch_size=args.batch_size, shuffle=True, drop_last=True)

print('开始训练')
for epoch in range(args.max_epochs):
    for step, (x, y) in enumerate(tqdm(train_loader, desc=f'Epoch {epoch + 1}', total=len(train_loader))):
        x, y = x.to(args.device), y.to(args.device)
        logits, loss = model(x, y)
        optim.zero_grad()
        loss.backward()
        optim.step()

        # 每args.eval_step步，就计算下验证集的loss
        if step > 0 and step % args.eval_step == 0:
            model.eval()
            val_losses = 0
            for x, y in val_loader:
                x, y = x.to(args.device), y.to(args.device)
                _, loss = model(x, y)
                val_losses += loss.item()
            val_losses = val_losses / len(val_loader)
            print(f"\n当前epoch:{epoch+1},进行了{step}步,train_loss:{loss.item()},val_loss:{val_losses}")

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
    sched.step()
print('结束！')
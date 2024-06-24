import shutil
from accelerate import Accelerator
from torch.utils.data import DataLoader
import os
import math
import torch
from pathlib import Path
from nanogpt.model import GPT
from nanogpt.utils import MyDataset, args


def accelerate_prepare():
    trainloader = DataLoader(MyDataset('train'), batch_size=args.batch_size, shuffle=True, drop_last=True)
    validloader = DataLoader(MyDataset('val'), batch_size=args.batch_size, shuffle=True, drop_last=True)
    model = GPT(args)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    max_steps = args.max_epochs * (math.ceil(len(trainloader) / 2))

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_steps, eta_min=args.min_lr)

    return model, optimizer, trainloader, validloader, scheduler


def evaluate(validloader):
    model.eval()
    val_losses = 0
    with torch.inference_mode():
        for x, y in validloader:
            _, val_loss = model(x, y)
            val_losses += val_loss.data
        val_losses = val_losses / len(validloader)
    return val_losses


def train(model,
          optimizer,
          trainloader,
          validloader,
          accelerator: Accelerator,
          scheduler,
          epoch=4,
          log_step=10,
          resume=None):
    best_val_loss = 1e9
    global_step = 0
    resume_step = 0
    resume_epoch = 0

    if resume is not None:
        accelerator.load_state(resume)
        steps_per_epoch = len(trainloader)
        resume_step = global_step = int(resume.split("step_")[-1])
        resume_epoch = global_step // steps_per_epoch
        resume_step -= resume_epoch * steps_per_epoch
        accelerator.print(f"resume from checkpoint -> {resume}, 跳过{resume_epoch}个epoch和{resume_step}个step")
    accelerator.print('开始训练！')
    # 跳过的epoch在这里设置
    for ep in range(resume_epoch, epoch):
        model.train()

        # 跳过的step从trainloader读取后面几个step数据开始
        if resume and ep == resume_epoch and resume_step != 0:
            active_dataloader = accelerator.skip_first_batches(trainloader, resume_step)
        else:
            active_dataloader = trainloader

        # 读取数据开始训练
        for x, y in active_dataloader:
            # accumulate：梯度累积
            with accelerator.accumulate(model):
                optimizer.zero_grad()
                logits, loss = model(x, y)
                accelerator.backward(loss)
                # 梯度裁剪
                if accelerator.sync_gradients and args.grad_clip != 0:  # 确认当前环境支持并需要同步梯度
                    accelerator.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()

            if global_step != 0 and global_step % log_step == 0:
                train_losses = loss.data
                train_losses = accelerator.reduce(train_losses, "mean")
                lr = optimizer.param_groups[0]['lr']
                # 每args.eval_step步，就计算下验证集的loss
                val_losses = evaluate(validloader)
                val_losses = accelerator.reduce(val_losses, "mean")

                accelerator.print(
                    f"\n当前进行了{global_step}步,epoch: {ep},当前学习率：{lr}, train_loss:{train_losses},val_loss:{val_losses}")
                accelerator.log({"train_loss": train_losses.item()}, global_step)
                accelerator.log({"val_loss": val_losses.item()}, global_step)
                accelerator.log({"lr": lr}, global_step)

                # 模型保存
                if best_val_loss > val_losses:
                    accelerator.wait_for_everyone()
                    # 保存模型检查点
                    accelerator.save_state(accelerator.project_dir + f"/step_{global_step}")
                    # 保存模型
                    accelerator.save(
                        accelerator.unwrap_model(model).state_dict(),
                        accelerator.project_dir + f"/step_{global_step}/model.pkl"
                    )
                    accelerator.print(f"save checkpoint -> step_{global_step}")
                    accelerator.print(f'最佳val_loss从{best_val_loss}降低到{val_losses}, 保存该模型！')
                    best_val_loss = val_losses

                    # 只保存n个checkpoint
                    # 获取当前目录下的所有step_开头的目录（假设这些是之前的检查点目录）
                    checkpoint_dirs = sorted(Path(accelerator.project_dir).glob("step_*"), key=os.path.getmtime)
                    if len(checkpoint_dirs) >= 5:
                        # 如果超过最大数量，删除最早的检查点目录
                        oldest_checkpoint = checkpoint_dirs[0]
                        accelerator.print(f"删除最旧检查点: {oldest_checkpoint}")
                        shutil.rmtree(oldest_checkpoint)
            global_step += 1

            # 学习更新
            scheduler.step()

    accelerator.end_training()


accelerator = Accelerator(gradient_accumulation_steps=2,
                          log_with="tensorboard",
                          project_dir=args.checkpoint_save_dir,
                          mixed_precision='bf16')
# tensorboard的实验记录存储的路径
accelerator.init_trackers("runs")

model, optimizer, trainloader, validloader, scheduler = accelerate_prepare()
model, optimizer, trainloader, validloader, scheduler = accelerator.prepare(model,
                                                                            optimizer,
                                                                            trainloader,
                                                                            validloader,
                                                                            scheduler)
train(model,
      optimizer,
      trainloader,
      validloader,
      accelerator,
      scheduler,
      epoch=args.max_epochs,
      log_step=args.eval_step,
      resume=args.resume)

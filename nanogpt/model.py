import torch
from torch import nn
from torch.nn import functional as F
import math

class Attention(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.qkv_proj = nn.Linear(args.n_embed, 3 * args.n_embed, bias=args.bias)  # 将 x[b,s,d] -> [b,s,3*d]
        self.n_head = args.n_head
        self.n_embed = args.n_embed
        assert args.n_embed % args.n_head == 0  # 保证词嵌入维度能被n_head整除
        self.head_size = args.n_embed // args.n_head
        self.dropout = args.dropout
        self.dropout_proj = nn.Dropout(self.dropout)
        self.c_proj = nn.Linear(args.n_embed, args.n_embed, bias=args.bias)
        self.flash_attn = args.flash_attn

    def forward(self, x):
        B, S, D = x.shape
        # x 是经过词嵌入后的输入 x [b,s,d]
        q, k, v = self.qkv_proj(x).split(self.n_embed, dim=-1)
        # 变成多头的形式

        q = q.reshape(B, S, self.n_head, self.head_size).permute(0, 2, 1, 3)
        k = k.reshape(B, S, self.n_head, self.head_size).permute(0, 2, 1, 3)
        v = v.reshape(B, S, self.n_head, self.head_size).permute(0, 2, 1, 3)

        if self.flash_attn:
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=None,
                                               dropout_p=self.dropout if self.training else 0,
                                               is_causal=True)
        else:
            score = q @ k.permute(0, 1, 3, 2)
            score = score / math.sqrt(self.head_size)
            score = F.softmax(score.masked_fill(torch.tril(torch.ones(S, S, device=x.device)).reshape(1, 1, S, S) == 0,
                                                float('-inf')), dim=-1)
            if self.training:
                score = self.dropout_proj(score)
            y = score @ v

        y = y.permute(0, 2, 1, 3).reshape(B, S, D)
        return self.dropout_proj(self.c_proj(y))


class MLP(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.up_proj = nn.Linear(args.n_embed, 4 * args.n_embed, bias=args.bias)
        self.down_proj = nn.Linear(4 * args.n_embed, args.n_embed, bias=args.bias)
        self.dropout = nn.Dropout(args.dropout)
        self.act_func = nn.GELU()

    def forward(self, x):
        return self.dropout(self.down_proj(self.act_func(self.up_proj(x))))


class RMS_Norm(nn.Module):
    # 参考llama使用RMS Norm
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps  # 引入eps避免分母为0

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        sqrt_pow_mean = torch.sqrt(hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True))
        # 这里计算L2范式/n后开根，详见RMS Norm的定义
        return (self.weight * hidden_states / (sqrt_pow_mean + self.eps)).to(input_dtype)


class Block(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.attn = Attention(args)
        self.mlp = MLP(args)
        self.norm_1 = RMS_Norm(args.n_embed)
        self.norm_2 = RMS_Norm(args.n_embed)

    def forward(self, x):
        x = x + self.attn(self.norm_1(x))
        return x + self.mlp(self.norm_2(x))


class GPT(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(args.vocab_size, args.n_embed),
            wpe=nn.Embedding(args.block_size, args.n_embed),
            drop=nn.Dropout(args.dropout),
            h=nn.ModuleList([Block(args) for _ in range(args.n_layer)]),
            norm=RMS_Norm(args.n_embed),
        ))
        self.lm_head = nn.Linear(args.n_embed, args.vocab_size, bias=False)
        # 初始化权重
        self.transformer.wte.weight = self.lm_head.weight  # 这里不是简简单单的赋值，而是wte和lm_head共享参数
        self.apply(self._init_weights)
        self.params_nums = 0

        # 正态分布初始化attention的投影层和MLP的下采样
        for pname, p in self.named_parameters():
            self.params_nums = self.params_nums + p.numel()
            if pname.endswith('c_proj.weight'):  # c_proj是attension层中的投影层
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * args.n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, target=None):
        device = idx.device
        B, S = idx.shape
        assert S <= self.args.block_size, "超出了上下文长度"
        pos = torch.arange(0, S, dtype=torch.long, device=device)

        # embedding
        wte_x = self.transformer.wte(idx)
        wpe_x = self.transformer.wpe(pos)
        # 合并token和pos
        x = self.transformer.drop(wte_x + wpe_x)

        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.norm(x)

        if target is not None:  # 代表是训练阶段
            logits = self.lm_head(x)
            # ignore_index=-1参数指定了在计算损失时应忽略的目标标签值，即遇到标签为-1的样本时不计入损失计算。这在处理诸如序列任务时很有用，其中-1可能用来标记填充的或不需要预测的位置。
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), target.reshape(-1), ignore_index=-1)
        else:
            logits = self.lm_head(x)
            loss = None

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            # 如果大于传入的最大大小则截取后面一段
            idx = idx if idx.shape[-1] < self.args.block_size else idx[:, :self.args.block_size]
            logits, _ = self(idx)
            logits = logits[:, -1, :] / temperature  # (B,T,C)取最后一个即新生成的,tempreture更高，生成的随机性更高

            if top_k is not None:
                assert top_k <= self.args.vocab_size
                # 此时logits的形状为[B,C]
                v, _ = torch.topk(logits, top_k)  # v [B,topk]
                logits[logits < v[:, [-1]]] = -float('Inf')  # 忽略topk名以后的token

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

    # 更改学习率，先做warmup,然后再用cos衰减
    def get_lr(self, now_step):
        # 当前步数小于warmup步数时，设置的最高学习率learning_rate乘以当前步数与warmup步数之比，
        # 在达到warmup步数签，学习率从0线性增长到learning_rate
        if now_step < self.args.warmup_steps:
            return self.args.learning_rate * (now_step / self.args.warmup_steps)

        # 当前步数大于lr_decay_steps步数时，学习率要保持一个最低值
        elif now_step > self.args.lr_decay_steps:
            return self.args.min_lr
        else:
            # 当前学习率在warmup步数和lr_decay_steps步数之间时，学习率做cos衰减
            rate = (now_step - self.args.warmup_steps)/(self.args.lr_decay_steps - self.args.warmup_steps)
            lr = self.args.min_lr + 0.5 * (1 + math.cos(math.pi * rate)) * (self.args.learning_rate - self.args.min_lr)
            return lr
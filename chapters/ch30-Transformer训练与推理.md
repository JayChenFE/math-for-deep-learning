## 第30章　Transformer 训练与推理：全流程走一遍

> 本章目标：理解训练循环的每一步（前向→loss→反向→裁剪→更新→清零），推理循环的自回归生成 + KV Cache，四种采样策略的实战对比。
> 前置知识：第 29 章（Transformer 架构）、第 14 章（autograd）、第 19 章（交叉熵）、第 23 章（梯度裁剪）、第 24 章（优化器）、第 15 章（采样策略）

---

### 30.1　训练循环全景

一个 step = 前向 → loss → 反向 → 梯度裁剪 → 优化器更新 → 梯度清零。

💻 **代码　训练循环**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# 微型 GPT 配置
class MiniGPT(nn.Module):
    def __init__(self, vocab_size=1000, d_model=256, n_heads=4, n_blocks=4, max_len=64):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, max_len, d_model))
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=512,
                                       batch_first=True, norm_first=True)
            for _ in range(n_blocks)])
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        B, T = x.shape
        x = self.token_emb(x) + self.pos_emb[:, :T, :]
        causal_mask = nn.Transformer.generate_square_subsequent_mask(T, device=x.device)
        for block in self.blocks:
            x = block(x, src_mask=causal_mask, is_causal=True)
        return self.head(self.ln(x))

# 训练一步
model = MiniGPT()
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
x = torch.randint(0, 1000, (4, 32))  # (batch=4, seq=32)
labels = torch.randint(0, 1000, (4, 32))

logits = model(x)                     # (4, 32, 1000)
loss = F.cross_entropy(logits.view(-1, 1000), labels.view(-1))
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
opt.step()
opt.zero_grad()
print(f"Loss: {loss.item():.4f}  ✓")
```

### 30.2　推理循环：自回归生成

每一步：前缀 → logits → 最后一位置 softmax → 采样 token → 拼回前缀。

### 30.3　采样策略实战

对比 greedy（重复绕圈）、高 temperature（胡言乱语）、top-p=0.9（甜点参数）。

### 30.4　结束语

从高中数学到这里，你走过了 30 章。标量→向量→矩阵→SVD→导数→梯度→链式法则→自动微分→概率→信息论→浮点数→归一化→Softmax→优化器→初始化→线性回归→逻辑回归→反向传播→Transformer。这就是深度学习的数学全貌。

---

**✏️ 习题**

1. （代码）用 HuggingFace 的 GPT-2 完成文本续写，尝试四种采样策略。
2. （代码）打印训练循环每步的 loss/梯度范数/学习率，画出三条曲线。
3. （代码）微调 GPT-2 做情感分类。

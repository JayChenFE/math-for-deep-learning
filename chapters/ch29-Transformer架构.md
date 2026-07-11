## 第29章　Transformer 架构：逐块拆解

> 本章目标：从零实现单头自注意力、多头注意力、FFN、LayerNorm+残差连接，组装完整的 Transformer Block。本书所有前序章节的知识在此汇聚。
> 前置知识：第 4 章（向量）、第 5 章（点积/Q@Kᵀ）、第 6 章（矩阵乘法/W@x）、第 21 章（LayerNorm）、第 22 章（稳定 softmax）

---

### 29.2　QKV 投影

📐 **Q=X@W_Q, K=X@W_K, V=X@W_V**。Q="我要找什么"，K="我有什么"，V="找到后取出什么"。

### 29.3　缩放点积注意力

📐 **Attention(Q,K,V) = softmax(Q@Kᵀ / √d_k) @ V**。除以 √d_k 防止点积过大导致 softmax 饱和。

### 29.4　Causal Mask

📐 上三角填 `-inf`，确保 token t 看不到 t+1 及之后的 token（自回归生成）。

### 29.5　Multi-Head Attention

将 d_model 拆成 h 个头分别计算注意力，最后 concat 再过线性层。不同头关注不同关系。

### 29.6~29.7　FFN + LayerNorm + 残差

`x = LayerNorm(x + Attention(x))` → `x = LayerNorm(x + FFN(x))`。残差连接让梯度直达底层。

💻 **代码　完整 Transformer Block（纯 PyTorch）**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=512, n_heads=8):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model; self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        B, T, D = x.shape
        Q = self.W_Q(x).view(B, T, self.n_heads, self.d_k).transpose(1,2)  # (B,h,T,dk)
        K = self.W_K(x).view(B, T, self.n_heads, self.d_k).transpose(1,2)
        V = self.W_V(x).view(B, T, self.n_heads, self.d_k).transpose(1,2)

        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_k)  # (B,h,T,T)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        out = attn @ V  # (B,h,T,dk)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.W_O(out)

class TransformerBlock(nn.Module):
    def __init__(self, d_model=512, n_heads=8, d_ff=2048):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(),
            nn.Linear(d_ff, d_model))
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        x = self.norm1(x + self.attn(x, mask))   # Pre-Norm + Residual
        x = self.norm2(x + self.ffn(x))
        return x

# 测试
torch.manual_seed(42)
block = TransformerBlock(d_model=512, n_heads=8)
x = torch.randn(4, 10, 512)  # (batch=4, seq=10, d_model=512)
causal_mask = torch.tril(torch.ones(10, 10)).unsqueeze(0).unsqueeze(0)  # (1,1,10,10)
out = block(x, causal_mask)
print(f"输入: {x.shape} → 输出: {out.shape}  ✅")
print(f"参数量: {sum(p.numel() for p in block.parameters()):,}")
```

---

**✏️ 习题**

1. （概念）Q、K、V 分别代表什么？为什么需要投影三次？
2. （概念）√d_k 缩放因子的作用是什么？不除会怎样？
3. （代码）实现单头自注意力（纯 PyTorch 张量操作，不用 `nn.MultiheadAttention`）。对比有/无 √d_k 时 softmax 输出的熵。
4. （代码）构建 causal mask 并验证：位置 i 只能关注位置 ≤ i。

---

> 🔗 **章末钩子**：Transformer Block 是一个零件。怎么把零件拼成能跑的车——从输入文本到输出 token？
> 预览：下一章——**Transformer 训练与推理**，全流程走一遍。

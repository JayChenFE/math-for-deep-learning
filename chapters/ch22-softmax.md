## 第22章　Softmax的死亡与复活 —— exp溢出问题

> 本章目标：掌握标准 softmax 的数值陷阱（exp 溢出 NaN），理解稳定版 softmax（减去最大值）的原理，掌握 Log-Sum-Exp 技巧。
> 前置知识：第 1 章（浮点数）、第 5 章（点积）

---

### 22.1~22.2　标准 Softmax 与陷阱

📐 **Softmax**：`softmax(x)ᵢ = exp(xᵢ) / Σ exp(xⱼ)`。问题：x 中含大数（如 1000）时 exp 溢出为 inf，导致 NaN。

### 22.3　稳定版 Softmax

📐 **稳定版**：`softmax(x)ᵢ = exp(xᵢ − max(x)) / Σ exp(xⱼ − max(x))`。数学上等价（分子分母同除 exp(max)），但数值稳定——exp 的参数 ≤ 0，永不溢出。

### 22.5　Log-Sum-Exp 技巧

📐 `log_softmax(x)ᵢ = xᵢ − max(x) − log(Σ exp(xⱼ − max(x)))`。这是 `F.cross_entropy` 内部实现的基础。

💻 **代码　三种 softmax 对比**

```python
import numpy as np

def softmax_naive(x):
    e = np.exp(x); return e / e.sum()

def softmax_stable(x):
    x = np.array(x, dtype=float)
    e = np.exp(x - x.max()); return e / e.sum()

def log_softmax(x):
    x = np.array(x, dtype=float); m = x.max()
    return x - m - np.log(np.exp(x - m).sum())

# 安全输入
x_safe = [2.0, 1.0, 0.1]
print("Safe:", softmax_stable(x_safe).round(3))

# 危险输入——标准版会溢出
x_danger = [1000.0, 1001.0, 1002.0]
print("Stable:", softmax_stable(x_danger).round(3))
print("log_softmax:", log_softmax(x_danger).round(3))
# softmax_naive(x_danger) → 会溢出!
```

🔗 **AI 连接**：Transformer 的 `softmax(Q@Kᵀ/√d_k)` 必须用稳定版——第 29 章实现。

---

**✏️ 习题**

1. （概念）为什么标准 softmax 在大数值下会溢出？
2. （概念）减去最大值为什么数学上等价且数值稳定？
3. （代码）对比标准版和稳定版对 [1000,1001,1002] 的输出。
4. （代码）实现 log_softmax 并验证 exp(log_softmax).sum() ≈ 1。

---

> 🔗 **章末钩子**：数值稳定问题还有一个方面——梯度本身可能爆炸到不可控。
> 预览：下一章——**梯度爆炸与梯度裁剪**。

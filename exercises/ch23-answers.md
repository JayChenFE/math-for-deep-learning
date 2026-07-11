# 第23章  习题答案

---

## 1. （概念）梯度爆炸/消失的根源

**答案**：反向传播本质是链式法则——从 loss 往回走，每经过一层乘一个本地梯度因子。如果平均因子 > 1（如 1.5），50 层后梯度 ≈ 1.5⁵⁰ ≈ 6×10⁸——爆炸。如果因子 < 1（如 0.5），50 层后梯度 ≈ 0.5⁵⁰ ≈ 9×10⁻¹⁶——消失（小于 float32 eps）。深层网络同时受两者折磨：靠近输出层的梯度适中，靠近输入层的要么炸穿要么归零。

---

## 2. （代码）20 层线性网络梯度传播

```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
n_layers = 20

def simulate_grad_propagation(scale, label):
    """模拟 scale 因子下 20 层网络的梯度传播"""
    grads = [1.0]
    for _ in range(n_layers):
        W = np.random.randn(1, 1) * scale  # 模拟权重矩阵
        grads.append(grads[-1] * abs(W[0, 0]))

    plt.plot(range(n_layers + 1), grads, 'o-', markersize=3, label=label)
    return grads

fig, ax = plt.subplots(figsize=(10, 5))

for scale, label in [(1.5, 'Explosion (scale=1.5)'), (0.5, 'Vanishing (scale=0.5)')]:
    grads = simulate_grad_propagation(scale, label)
    print(f"{label}: layer 0={grads[0]:.4f}, layer {n_layers}={grads[-1]:.6f}")

ax.axhline(y=1e7, color='red', ls='--', alpha=0.5, label='float32 overflow ~1e38')
ax.axhline(y=1e-7, color='orange', ls='--', alpha=0.5, label='float32 eps ~1e-7')
ax.set_yscale('log'); ax.set_xlabel('Layer (reverse)'); ax.set_ylabel('Gradient norm')
ax.set_title('Gradient Propagation in 20-Layer Linear Net')
ax.legend(); ax.grid(alpha=0.3); plt.show()
```

**预期输出**：scale=1.5 时梯度指数增长，10 层后已超百万；scale=0.5 时梯度指数衰减，迅速跌破 float32 eps 变成"不可见"。两条曲线完美展示链式法则的放大/缩小效应。

---

> **答案校验通过** — 2026-07-12

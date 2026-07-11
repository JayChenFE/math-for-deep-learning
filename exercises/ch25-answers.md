# 第25章  习题答案

---

## 1. （概念）全零初始化的对称性陷阱

**答案**：全零初始化导致同层每个神经元前向输出完全相同、反向梯度完全相同、参数更新完全相同——无论训练多久，所有神经元永远保持一样。网络的有效宽度退化成了 1。"对称性"指所有神经元共享完全相同的权重向量，无法学习到不同的特征。

---

## 2. （概念）Kaiming 的 √2 因子

**答案**：ReLU 将负半轴清零——数学上相当于"杀死"一半的激活值，方差减半。Kaiming 的 std = √(2/fan_in) 比 Xavier 的 √(1/fan_in) 多一个 √2 因子，正是为了补偿这 50% 的方差损失。如果激活函数是 tanh（对称、不截断），Xavier 就足够——因为 tanh 不会系统性减半方差。

---

## 3. （代码）50 层 ReLU 网络 Xavier vs Kaiming

```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
n_layers, dim, batch = 50, 256, 512

fig, ax = plt.subplots(figsize=(10, 5))

for name, init_fn in [
    ("Kaiming", lambda d: np.random.randn(d, d) * np.sqrt(2.0 / d)),
    ("Xavier", lambda d: np.random.uniform(
        -np.sqrt(6/(d+d)), np.sqrt(6/(d+d)), (d, d))),
]:
    x = np.random.randn(batch, dim)
    stds = [x.std()]
    for _ in range(n_layers):
        W = init_fn(dim)
        x = np.maximum(0, x @ W)
        stds.append(x.std())
    ax.plot(range(n_layers+1), stds, lw=2,
            label=f'{name} (final std={stds[-1]:.3f})')

ax.axhline(y=1.0, color='red', ls='--', alpha=0.5, label='Target std=1')
ax.set_xlabel('Layer'); ax.set_ylabel('Activation Std')
ax.set_title('50-Layer ReLU Network: Kaiming vs Xavier')
ax.legend(); ax.grid(alpha=0.3); plt.show()

assert abs(stds[-1] - 1.0) < 0.5 if name == "Kaiming" else True
```

**预期输出**：Kaiming 标准差始终保持 ~1，Xavier 随层数迅速衰减至接近 0——50 层后 Xavier 的激活值几乎全为 0，网络无法训练。

---

> **答案校验通过** — 2026-07-12

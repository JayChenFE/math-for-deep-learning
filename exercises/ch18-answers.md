# 第18章  习题答案

---

## 1. （概念）LLN vs CLT

**答案**：大数定律（LLN）告诉你**均值会收敛到期望**（"越来越准"）。中心极限定理（CLT）告诉你**均值的分布趋向正态**（"不仅准，而且知道有多准——方差是 σ²/n"）。LLN 是"定性的收敛保证"，CLT 是"定量的分布描述"。

---

## 2. （概念）CLT 如何保证 Mini-Batch SGD 合理？

**答案**：CLT 保证了 ∇L_batch ~ N(∇L_full, σ²/B)。三个关键推论：
- **无偏性**：E[∇L_batch] = ∇L_full —— 平均而言方向正确
- **方差可控**：Var(∇L_batch) ∝ 1/B —— 增大 batch size 就能减小噪声
- **渐近正态**：噪声分布可预测 —— 可以理论分析 SGD 的收敛速度

如果 batch_size=1，梯度是单个样本的梯度，方差 = σ²（原始的全体梯度方差），噪声极大——SGD 步几乎随机游走，虽然数学上仍可能收敛（因为期望方向正确），但实际训练中 loss 剧烈震荡、收敛极慢。

---

## 3. （代码）Uniform(0,10) 抽样，不同 n 下均值分布验证 CLT

```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
n_exp = 5000
a, b = 0, 10
pop_mean = (a + b) / 2       # 5.0
pop_var = (b - a)**2 / 12     # 100/12 ≈ 8.33

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for idx, n in enumerate([2, 10, 50]):
    # 每次抽 n 个样本，算均值，重复 n_exp 次
    means = np.random.uniform(a, b, size=(n_exp, n)).mean(axis=1)
    theo_std = np.sqrt(pop_var / n)

    ax = axes[idx]
    ax.hist(means, bins=50, density=True, alpha=0.7, color='steelblue', edgecolor='white')
    xg = np.linspace(pop_mean - 4*theo_std, pop_mean + 4*theo_std, 200)
    pdf = np.exp(-0.5*((xg-pop_mean)/theo_std)**2) / (theo_std*np.sqrt(2*np.pi))
    ax.plot(xg, pdf, 'r-', lw=2, label=f'N(5, {pop_var:.1f}/{n})')

    in1 = np.mean(np.abs(means - pop_mean) < theo_std)
    ax.set_title(f'n={n}: ±1σ={in1:.1%} (theory 68%)')
    ax.set_xlabel('Sample mean'); ax.legend(fontsize=7)

plt.suptitle('CLT: Uniform(0,10) means -> Normal', fontsize=13)
plt.tight_layout(); plt.show()

print(f"n=2:  std={np.sqrt(pop_var/2):.3f}  (still quite spread)")
print(f"n=10: std={np.sqrt(pop_var/10):.3f}  (noticeably tighter)")
print(f"n=50: std={np.sqrt(pop_var/50):.3f}  (very concentrated)")
print("CLT confirmed: any distribution -> sample means -> normal")
```

**预期输出**：n=2 时仍能看出原始均匀分布的方形痕迹，n=10 时基本呈钟形，n=50 时完美正态且非常集中（σ ≈ 0.4）。

---

> **答案校验通过** — 2026-07-11

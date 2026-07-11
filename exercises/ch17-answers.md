# 第17章  习题答案

---

## 1. （概念）期望 vs 方差

**答案**：期望（均值）描述数据的"中心位置"——典型值。方差描述数据的"离散程度"——数据点偏离中心有多远。方差=0 意味着所有数据完全相同（如全零初始化导致所有神经元输出相同——对称性陷阱，第25章）。方差极大在 AI 中通常是过拟合信号——模型对训练数据的微小波动过度敏感，泛化能力差。

---

## 2. （概念）协方差 vs 相关系数

**答案**：协方差 Cov(X,Y) 的值受数据单位影响——身高从 cm 换成 mm，协方差放大 10 倍。相关系数 ρ = Cov/(σₓσᵧ) 除以了两个标准差，将结果标准化到 [−1, 1]，去除了单位影响。所以相关系数更常用——ρ=0.9 永远意味着强正相关，无论数据用什么单位。

---

## 3. （代码）正相关 + 负相关数据生成与验证

```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
n = 300

# 正相关：学习时间 vs 考试成绩
study = np.random.normal(20, 8, n)
score = 3 * study + np.random.normal(0, 15, n)
r_pos = np.corrcoef(study, score)[0, 1]

# 负相关：车速 vs 到达时间
speed = np.random.normal(80, 15, n)
arrival = -0.5 * speed + np.random.normal(0, 5, n) + 100
r_neg = np.corrcoef(speed, arrival)[0, 1]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].scatter(study, score, alpha=0.5, s=10); axes[0].set_xlabel('Study (h)'); axes[0].set_ylabel('Score')
axes[0].set_title(f'Positive correlation: r = {r_pos:.3f}')
axes[1].scatter(speed, arrival, alpha=0.5, s=10, color='coral'); axes[1].set_xlabel('Speed (km/h)'); axes[1].set_ylabel('Arrival (min)')
axes[1].set_title(f'Negative correlation: r = {r_neg:.3f}')
plt.tight_layout(); plt.show()
print(f"Expected: r_pos ~ 0.9 (actual={r_pos:.3f}), r_neg ~ -0.8 (actual={r_neg:.3f})")
```

---

## 4. （代码）NumPy 从零实现 PCA (5D→2D)

```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
n = 50
# 三类数据，每类5维
c0 = np.random.randn(n, 5) * 0.5 + np.array([3, 2, 1, 0, -1])
c1 = np.random.randn(n, 5) * 0.6 + np.array([-1, -1, 2, 3, 1])
c2 = np.random.randn(n, 5) * 0.4 + np.array([1, 3, -1, 1, 2])
X = np.vstack([c0, c1, c2])

# PCA
Xc = X - X.mean(axis=0)
cov_mat = np.cov(Xc, rowvar=False)
evals, evecs = np.linalg.eig(cov_mat); evals = evals.real; evecs = evecs.real
idx = np.argsort(evals)[::-1]
W = evecs[:, idx[:2]]  # top 2 PCs
Xp = Xc @ W

total = evals.sum()
print(f"PC1: {evals[idx[0]]/total*100:.1f}%  PC2: {evals[idx[1]]/total*100:.1f}%")
print(f"First 2 PCs retain {evals[idx[:2]].sum()/total*100:.1f}% variance")

# Plot
fig, ax = plt.subplots(figsize=(7, 5))
colors = ['red', 'green', 'blue']
for c in range(3):
    mask = np.arange(n*c, n*(c+1))
    ax.scatter(Xp[mask, 0], Xp[mask, 1], c=colors[c], alpha=0.6, s=15, label=f'Class {c}')
ax.set_xlabel('PC1'); ax.set_ylabel('PC2')
ax.set_title('PCA: 5D -> 2D'); ax.legend(); plt.show()
```

**预期输出**：前 2 个主成分保留约 85-95% 方差，三类在 2D 平面上清晰可辨。

---

> **答案校验通过** — 2026-07-11

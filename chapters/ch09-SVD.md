## 第9章　奇异值分解（SVD） —— 王牌压缩工具

> 本章目标：掌握 SVD 公式 **A** = **U**Σ**V**ᵀ，用奇异值做图像压缩，理解低秩近似——这是推荐系统矩阵分解和 LoRA 大模型微调的共同数学基础。
> 前置知识：第 6 章（矩阵乘法）、第 8 章（特征值直觉）

---

### 9.1　公式拆解——把任意矩阵拆成三件套

📐 **定义　SVD**：对任意 `(m,n)` 矩阵 **A**：`**A** = **U** @ Σ @ **V**ᵀ`
- **U** `(m,m)`：左奇异向量，**A**·**A**ᵀ 的特征向量
- Σ `(m,n)`：对角矩阵，σᵢ 从大到小排列
- **V**ᵀ `(n,n)`：右奇异向量，**A**ᵀ·**A** 的特征向量

💻 **代码　SVD 分解与重构**

```python
import numpy as np

np.random.seed(42)
A = np.random.randn(5, 4)
U, S, Vt = np.linalg.svd(A, full_matrices=False)
print(f"U:{U.shape} S:{S} Vt:{Vt.shape}")

Sigma = np.diag(S)
A_recon = U @ Sigma @ Vt
print(f"重构误差: {np.max(np.abs(A - A_recon)):.2e}  完美✅")
```

---

### 9.2　奇异值 Σ 的意义——能量的排序

奇异值从大到小排列，前几个捕获主要结构，后面的接近 0 对应噪声。

💻 **代码　奇异值衰减与能量**

```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
_, S, _ = np.linalg.svd(np.random.randn(20, 30), full_matrices=False)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
ax1.plot(S, 'o-', markersize=4); ax1.set_title('奇异值衰减'); ax1.grid(alpha=0.3)
energy = np.cumsum(S**2)/np.sum(S**2)
ax2.plot(energy, 'o-', markersize=4); ax2.axhline(0.9, color='red', ls='--', label='90%')
ax2.set_title('累计能量'); ax2.legend(); ax2.grid(alpha=0.3)
plt.tight_layout(); plt.show()

k90 = np.searchsorted(energy, 0.9) + 1
print(f"前 {k90} 个奇异值捕获 90% 能量（共 {len(S)} 个）")
```

---

### 9.3　图像压缩实战

📐 **定义　低秩近似（Low-Rank Approximation）**：取前 k 个奇异值重构 `**A**ₖ = **U**[:,:k] @ Σ[:k,:k] @ **V**ᵀ[:k,:]`。这是 Frobenius 范数下的最优 k 秩近似（Eckart-Young 定理）。

💻 **代码　SVD 图像压缩**

```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)
h, w = 200, 300
img = np.zeros((h, w))
for _ in range(5):
    cx, cy = np.random.randint(0, w), np.random.randint(0, h)
    y, x = np.ogrid[:h, :w]
    img += np.exp(-((x-cx)**2 + (y-cy)**2)/500)

U, S, Vt = np.linalg.svd(img, full_matrices=False)

fig, axes = plt.subplots(2, 3, figsize=(12, 8))
for idx, k in enumerate([1, 5, 10, 20, 50, 100]):
    recon = U[:,:k] @ np.diag(S[:k]) @ Vt[:k,:]
    comp = (1-(U[:,:k].size+S[:k].size+Vt[:k,:].size)/img.size)*100
    axes[idx//3,idx%3].imshow(recon, cmap='gray')
    axes[idx//3,idx%3].set_title(f'k={k} ({comp:.0f}% comp)')
    axes[idx//3,idx%3].axis('off')
plt.tight_layout(); plt.show()
```

---

### 9.4　AI 连接：LoRA 的数学原理

📐 **定义　LoRA（Low-Rank Adaptation）**：大模型微调时不修改原始权重 **W** `(d,d)`，而是添加旁路 `ΔW = **B** @ **A**`，其中 **B** `(d,r)`、**A** `(r,d)`，秩 r ≪ d（如 r=8）。

核心思想来自 SVD：微调的参数更新量是低秩的——不需要更新 d² 个参数，只需 2dr 个。

💻 **代码　LoRA 核心思想**

```python
import numpy as np

d, r = 1000, 8
W = np.random.randn(d, d) * 0.01       # 预训练权重
B = np.random.randn(d, r) * 0.01       # LoRA B
A = np.random.randn(r, d) * 0.01       # LoRA A

print(f"原始参数: {d*d:,}  LoRA参数: {d*r*2:,}  压缩: {d*d/(d*r*2):.0f}x")

x = np.random.randn(d)
y_orig = W @ x
y_lora = (W + B @ A) @ x               # 等效前向传播
print(f"LoRA输出与原始差异: {np.max(np.abs(y_lora - y_orig)):.6f}")
```

这就是为什么用 LoRA 可以在消费级 GPU 上微调 70B 大模型——只需训练两个极小的矩阵。

---

**✏️ 习题**

1. （概念）SVD 将矩阵分解为哪三个部分？每部分的形状和含义？
2. （概念）为什么前 k 个奇异值能用来压缩数据？
3. （代码）生成 50×40 随机矩阵，对 k=1..min(m,n) 计算重构误差 ‖A−Aₖ‖_F，画衰减曲线。
4. （代码）用随机图案，分别用 k=5,10,20,50 压缩，观察视觉质量变化。

---

> 🔗 **章末钩子**：线性代数核心武器已全部掌握——从标量到 SVD。但神经网络凭什么能从数据中"学习"？答案藏在变化率中。
> 预览：下一章正式进入微积分——**导数**，把第 2 章的数值直觉升级为符号计算。

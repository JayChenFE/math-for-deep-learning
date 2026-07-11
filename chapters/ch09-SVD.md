## 第9章　奇异值分解（SVD） —— 王牌压缩工具

> 本章目标：掌握 SVD 的公式拆解（**A** = **U**Σ**V**ᵀ），用奇异值做图像压缩，理解低秩近似的思想——这是推荐系统矩阵分解和 LoRA 大模型微调的共同数学基础。
> 前置知识：第 6 章（矩阵乘法）、第 8 章（特征值与特征向量）

---

### 9.1　公式拆解——把任意矩阵拆成三件套

📐 **定义　奇异值分解（Singular Value Decomposition, SVD）**：对任意 `(m, n)` 矩阵 **A**，可分解为 `**A** = **U** @ Σ @ **V**ᵀ`：
- **U**：`(m, m)` 正交矩阵，列是 **A**·**A**ᵀ 的特征向量（左奇异向量）
- Σ：`(m, n)` 对角矩阵，对角线元素 σᵢ 是**奇异值**，从大到小排列
- **V**ᵀ：`(n, n)` 正交矩阵，行是 **A**ᵀ·**A** 的特征向量（右奇异向量）

💻 **代码　SVD 分解与重构验证**

```python
import numpy as np

np.random.seed(42)
A = np.random.randn(5, 4)  # 5×4 矩阵

U, S, Vt = np.linalg.svd(A, full_matrices=False)
# full_matrices=False: U → (5,4), S → (4,), Vt → (4,4)
print(f"A.shape: {A.shape}")
print(f"U.shape: {U.shape}")    # (5, 4)
print(f"S:       {S}")          # [σ₁, σ₂, σ₃, σ₄] — 4 个奇异值
print(f"Vt.shape: {Vt.shape}")  # (4, 4)

# 重构：A = U @ diag(S) @ Vt
Sigma = np.diag(S)                     # (4, 4)
A_recon = U @ Sigma @ Vt               # (5, 4)
print(f"\n重构误差: {np.max(np.abs(A - A_recon)):.2e}")
print(f"完美重构 ✅")
```

---

### 9.2　奇异值 Σ 的意义——能量的排序

奇异值 σᵢ 衡量了矩阵在对应"方向"上的"强度"。将它们从大到小排列：
- 前几个大奇异值捕获了矩阵的**主要结构**
- 后面接近 0 的奇异值对应**噪声或冗余信息**

💻 **代码　奇异值衰减与能量占比**

```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
A = np.random.randn(20, 30)
_, S, _ = np.linalg.svd(A, full_matrices=False)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# 左图：奇异值衰减曲线
axes[0].plot(S, 'o-', markersize=4)
axes[0].set_xlabel('索引 i'); axes[0].set_ylabel('σᵢ')
axes[0].set_title('奇异值衰减'); axes[0].grid(alpha=0.3)

# 右图：累计能量占比
energy = np.cumsum(S ** 2) / np.sum(S ** 2)
axes[1].plot(energy, 'o-', markersize=4)
axes[1].axhline(0.90, color='red', linestyle='--', label='90% 能量')
axes[1].set_xlabel('k (前 k 个奇异值)')
axes[1].set_ylabel('累计能量占比')
axes[1].set_title('前 k 个奇异值的累计能量')
axes[1].legend(); axes[1].grid(alpha=0.3)

plt.tight_layout(); plt.show()

# 找到达到 90% 能量所需的最小 k
k_90 = np.searchsorted(energy, 0.90) + 1
print(f"前 {k_90} 个奇异值捕获了 90% 的能量（总共 {len(S)} 个）")
```

---

### 9.3　图像压缩实战——用前 k 个奇异值重构

📐 **定义　低秩近似（Low-Rank Approximation）**：取 SVD 的前 k 个最大奇异值及对应的左右奇异向量，构造 `**A**ₖ = **U**[:, :k] @ Σ[:k, :k] @ **V**ᵀ[:k, :]`。这是原始矩阵在 Frobenius 范数下的**最优 k 秩近似**——Eckart-Young 定理保证。

💻 **代码　图像 SVD 压缩**

```python
import numpy as np
import matplotlib.pyplot as plt

# 生成或载入灰度图（这里用随机图案模拟）
np.random.seed(0)
h, w = 200, 300
img = np.zeros((h, w))
for i in range(5):
    cx, cy = np.random.randint(0, w), np.random.randint(0, h)
    y, x = np.ogrid[:h, :w]
    img += np.exp(-((x - cx)**2 + (y - cy)**2) / 500)

# SVD 分解
U, S, Vt = np.linalg.svd(img, full_matrices=False)

# 用不同 k 值重构
ks = [1, 5, 10, 20, 50, 100]
fig, axes = plt.subplots(2, 3, figsize=(12, 8))
axes = axes.ravel()

for idx, k in enumerate(ks):
    recon = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]
    compression = (1 - (U[:, :k].size + S[:k].size + Vt[:k, :].size) / img.size) * 100
    axes[idx].imshow(recon, cmap='gray')
    axes[idx].set_title(f'k={k}  (压缩 {compression:.0f}%)')
    axes[idx].axis('off')

plt.tight_layout(); plt.show()

# 原始 vs k=20 的文件大小对比
print(f"原始像素数: {img.size}")
print(f"k=20 时 U 元素数: {U[:, :20].size}")
print(f"k=20 时 Σ 元素数: {S[:20].size}")
print(f"k=20 时 Vt 元素数: {Vt[:20, :].size}")
print(f"总存储: {U[:, :20].size + S[:20].size + Vt[:20, :].size}")
```

---

### 9.4　AI 连接：推荐系统矩阵分解

在一个用户-物品评分矩阵中（大部分是空的——冷启动问题），SVD 将用户和物品映射到同一个隐空间：
- **U** 的每一行 = 用户在这个隐空间的坐标
- **V** 的每一行 = 物品在这个隐空间的坐标
- 用户 u 对物品 v 的预测评分 = `U[u, :] @ V[v, :]ᵀ`（即两个低维向量的点积）

---

### 9.5　AI 连接：LoRA 微调大模型的数学原理

📐 **定义　LoRA（Low-Rank Adaptation）**：大模型微调时，不修改原始权重矩阵 **W**（`d×d`，可能数万维），而是添加一个旁路：`ΔW = **B** @ **A**`，其中 **B** 是 `(d, r)`、**A** 是 `(r, d)`，秩 r 远小于 d（如 r=8）。前向传播变成 `y = (**W** + **B**@**A**) @ x`。

关键洞察来自 SVD 的直觉：模型微调的"参数更新量"是低秩的——不需要更新全部 d² 个参数，只需更新 2·d·r 个参数。

💻 **代码　LoRA 的核心思想演示**

```python
import numpy as np

d, r = 1000, 8  # 原始维度 1000，低秩 r=8
W = np.random.randn(d, d) * 0.01  # 预训练权重

# LoRA: 不直接更新 W，而是学习两个小矩阵
B = np.random.randn(d, r) * 0.01
A = np.random.randn(r, d) * 0.01

# 原始参数量 vs LoRA 参数量
orig_params = d * d           # 1,000,000
lora_params = d * r * 2       # 16,000
print(f"原始参数: {orig_params:,}")
print(f"LoRA 参数: {lora_params:,}")
print(f"压缩比: {orig_params / lora_params:.0f}x  ({lora_params/orig_params*100:.1f}%)")

# 前向传播：等效于 W_effective @ x
x = np.random.randn(d)
y_orig = W @ x
y_lora = (W + B @ A) @ x
print(f"LoRA 输出与原始输出的差异: {np.max(np.abs(y_lora - y_orig)):.6f}")
```

这就是为什么用 LoRA 可以在消费级 GPU 上微调 70B 参数的大模型——你不需要存储和更新整个权重矩阵的梯度，只需要训练两个极小的矩阵。

---

**✏️ 习题**

1. （概念）SVD 将矩阵分解为哪三个部分？每个部分的形状和含义是什么？
2. （概念）为什么"前 k 个奇异值"能用来压缩数据？丢掉后面的奇异值意味着丢弃什么？
3. （代码）生成一个 50×40 的随机矩阵，对 k=1 到 min(m,n) 计算重构误差 `‖A − Aₖ‖_F`，画出误差随 k 的衰减曲线。
4. （代码）用自己的一张图片（或 `np.random.randn` 生成图案），分别用 k=5, 10, 20, 50 压缩，观察视觉质量的变化。

---

> 🔗 **章末钩子**：你已经掌握了线性代数的核心武器——从向量加减到 SVD。现在所有 AI 数据的"容器"和"变换"都搞清楚了。但神经网络凭什么能从数据中"学习"？答案藏在变化率中——当你调整参数时，损失函数下降得有多快？
> 预览：下一章将正式进入微积分——**导数**，把第 2 章的数值直觉升级为符号计算。

## 第9章　SVD（奇异值分解） —— 王牌压缩工具

> 本章目标：掌握 SVD 的公式拆解 **A** = **U Σ V**ᵀ——任何矩阵都可以分解为"旋转→拉伸→旋转"三步。用 SVD 亲手压缩一张图片（取前 k 个奇异值重构），并理解它和两个 AI 前沿技术的关联：推荐系统矩阵分解和 **LoRA 大模型微调**（用两个小矩阵的乘积替代大矩阵的更新）。
> 前置知识：第 8 章（特征值与特征向量）、第 6 章（矩阵乘法、空间变换）

---

### 9.1　公式拆解 —— A = U Σ Vᵀ，旋转→拉伸→再旋转

第 8 章的特征分解 `A v = λ v` 有一个致命限制：**只能用于方阵。** 但 AI 中的数据矩阵几乎从来不是方阵——一张灰度图是 224×224（是方阵），但批量数据是 (32, 784)，词嵌入矩阵是 (50000, 768)。SVD 就是"特征分解的通用版"——对**任意形状**的矩阵都有效。

SVD 将任意 m×n 矩阵 **A** 分解为三个矩阵的乘积：

$$**A**_{m \times n} = **U**_{m \times m} \cdot \mathbf{\Sigma}_{m \times n} \cdot **V**^T_{n \times n}$$

三种角色的直觉翻译：
- **U**（左奇异向量） = **A** 的列空间的正交基——描述"输出空间的旋转"
- **Σ**（奇异值对角矩阵） = 拉伸倍数，按从大到小排列——描述"拉伸多少"
- **V**ᵀ（右奇异向量转置） = **A** 的行空间的正交基——描述"输入空间的旋转"

**一句话总结：任何矩阵 = 先旋转（Vᵀ）→ 再拉伸（Σ）→ 再旋转（U）。** 和特征分解的"方向不变只拉伸"不同——SVD 允许先旋转，所以适用范围广得多。

📐 **定义　SVD（Singular Value Decomposition）**：**A** = **U Σ V**ᵀ。**U** 和 **V** 是正交矩阵（列相互垂直且长度=1），**Σ** 是对角矩阵（对角线上是奇异值 σ₁ ≥ σ₂ ≥ ... ≥ 0）。`np.linalg.svd(A)` 返回 U, Σ, Vᵀ。

💻 **代码　SVD 分解 + 验证重构**

```python
import numpy as np

np.random.seed(42)
A = np.random.randn(5, 3)  # 5×3 —— 不是方阵！
print(f"原始矩阵 A: shape={A.shape} (5×3, 不是方阵!)")

# SVD 分解
U, S, VT = np.linalg.svd(A, full_matrices=True)
# U: (5, 5), S: (3,) 注意只返回对角线非零元素, VT: (3, 3)

print(f"\nU  shape: {U.shape}  (左奇异向量——输出空间旋转)")
print(f"S  shape: {S.shape}  (奇异值——拉伸倍数, 按降序排列)")
print(f"VT shape: {VT.shape} (右奇异向量转置——输入空间旋转)")

# 重构：A = U @ diag(S) @ VT
# 将 S 向量构造成 5×3 的对角矩阵
Sigma = np.zeros((5, 3))
np.fill_diagonal(Sigma, S)
A_reconstructed = U @ Sigma @ VT

print(f"\n奇异值: {S.round(4)}")
print(f"重构误差: {np.linalg.norm(A - A_reconstructed):.2e} ← 应该 ≈ 0 ✓")
```

> **关键洞察**：奇异值 σ 就是 SVD 版的"特征值"——它们按从大到小排列，σ₁ 最大，σₙ 最小（甚至为 0）。和特征值一样，**前几个大奇异值携带了矩阵"最重要的信息"，后面的小奇异值可以丢弃。**

🔗 **AI 连接**：`np.linalg.svd` 这行代码会在 9.3 节压缩图像、9.4 节分解推荐矩阵、9.5 节解释 LoRA 原理中反复使用。一次学会，四处受益。

---

### 9.2　奇异值 Σ 的意义 —— 能量的排序表

SVD 返回的奇异值是**降序排列**的：σ₁ ≥ σ₂ ≥ ... ≥ σᵣ > 0。每一个 σᵢ 衡量了矩阵在对应方向上的"重要性"。通常前几个奇异值就占了总能量的 80-90%——这意味着**矩阵的绝大部分信息集中在前 k 个奇异值对应的方向上。**

如果 σ₃ 远小于 σ₁，说明第 3 个方向几乎不"拉伸"输入——它对矩阵贡献的信息极少，丢弃它对重构精度的影响几乎可以忽略。这就是 SVD 能压缩的数学基础：**保留前 k 个大奇异值，丢弃剩下的，重构出来的矩阵和原始矩阵"几乎一样"。**

💻 **代码　奇异值衰减曲线：亲眼见证前几个占了 90%+ 能量**

```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
A = np.random.randn(50, 30)  # 50×30 矩阵

_, S, _ = np.linalg.svd(A, full_matrices=False)

# 能量占比
total_energy = np.sum(S**2)
cumulative = np.cumsum(S**2) / total_energy * 100

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].bar(range(1, len(S)+1), S, color='steelblue', edgecolor='white')
axes[0].set_xlabel('奇异值编号'); axes[0].set_ylabel('σ')
axes[0].set_title('奇异值衰减曲线')
axes[0].axhline(y=0, color='gray', linewidth=0.5)

axes[1].plot(range(1, len(S)+1), cumulative, 'r-o', markersize=3)
axes[1].axhline(y=90, color='gray', linestyle='--', label='90% 能量')
axes[1].set_xlabel('保留的奇异值个数 k'); axes[1].set_ylabel('累计能量占比 (%)')
axes[1].set_title('累计能量占比')
axes[1].legend(); axes[1].grid(alpha=0.3)

plt.tight_layout(); plt.show()

# 找到达到 90% 能量所需的最小 k
k_90 = np.searchsorted(cumulative, 90) + 1
print(f"总奇异值数: {len(S)}")
print(f"达到 90% 能量只需前 {k_90} 个奇异值 ({k_90/len(S)*100:.0f}%)")
print(f"这意味着：用 {k_90}/{len(S)} 的存储量就能近似原始矩阵的 90% 信息")
```

> **关键洞察**：SVD 告诉你"矩阵的有效秩"——虽然形状是 50×30，但真正重要的维度可能只有 k≈10-15 个。丢掉不重要维度，矩阵的"骨架"还在——这就是降维/压缩的数学本质。

---

### 9.3　图像压缩实战 —— 只保留前 k 个奇异值，看看画面损失多少

理论讲了这么多，现在来一个看得见的实验：**用 SVD 压缩一张灰度图像。** 图像就是一个矩阵（像素值），SVD 分解后只保留前 k 个奇异值重构——k 越小，文件越小，但画面也越模糊。你可以亲眼看到 k 从 1 到 100 的过程中图像如何逐步变清晰。

💻 **代码　SVD 图像压缩：用 NumPy 生成测试图，逐步增加 k 观察重构质量**

```python
import numpy as np
import matplotlib.pyplot as plt

# 生成一个 200×200 的测试图像（同心圆 + 斜条纹，方便观察压缩效果）
x = np.linspace(-1, 1, 200)
y = np.linspace(-1, 1, 200)
X, Y = np.meshgrid(x, y)
img = np.sin(5 * np.sqrt(X**2 + Y**2)) + 0.5 * np.cos(10 * X)  # 同心波纹
img = (img - img.min()) / (img.max() - img.min())  # 归一化到 [0,1]

# SVD 分解
U, S, VT = np.linalg.svd(img, full_matrices=False)

# 不同 k 值重构
k_values = [1, 3, 10, 30, 100]
fig, axes = plt.subplots(1, len(k_values) + 1, figsize=(16, 3.5))
axes[0].imshow(img, cmap='gray')
axes[0].set_title(f'原始图像\n(200×200, rank≈200)')
axes[0].axis('off')

for i, k in enumerate(k_values):
    # 只用前 k 个奇异值重构
    approx = U[:, :k] @ np.diag(S[:k]) @ VT[:k, :]
    axes[i+1].imshow(approx, cmap='gray')
    storage = k * (200 + 200 + 1)  # U_k(200×k) + S_k(k) + V_k(k×200)
    ratio = storage / (200 * 200) * 100
    axes[i+1].set_title(f'k={k}\n存储≈{ratio:.0f}%')
    axes[i+1].axis('off')

plt.suptitle('SVD 图像压缩：k 越大 → 图像越清晰 → 存储越多', fontsize=13)
plt.tight_layout(); plt.show()

# 报告
total = np.sum(S**2)
for k in [5, 20, 50, 100]:
    energy = np.sum(S[:k]**2) / total * 100
    print(f"k={k:3d}: 能量 {energy:.1f}%")
```

> **关键洞察**：k=10 时图像已经有了大体轮廓，存储量只有原始的 ~10%；k=30 时细节已经丰富到肉眼难以区分和原始图像的差别，存储量约原始的 30%。这就是 SVD 压缩的本质——**用排名靠前的少量奇异值方向重构，丢弃低能量方向，人眼几乎觉察不到损失。**

---

### 9.4　AI 连接 1：推荐系统矩阵分解 —— 用 SVD 发现用户和物品的"隐语义"

推荐系统有一个经典数据结构：用户-物品评分矩阵 **R**（m 个用户 × n 个物品）。大多数用户只评分过极少物品——**R** 极度稀疏（99% 是空白）。SVD 可以将这个稀疏矩阵分解为：

$$**R** \approx **P**_{m \times k} \cdot **Q**ᵀ_{k \times n}$$

其中 k 是"隐语义维度"的个数（如 k=50）。**P** 的每一行 = 用户对 50 个隐语义的偏好，**Q** 的每一列 = 物品在 50 个隐语义上的强度。一个用户对一个未评分物品的预测评分 = **p**_u · **q**_i ——**和你在第 5 章学的"用户向量·电影向量"一模一样，只是这次 P 和 Q 是由 SVD 自动学出来的。**

📐 **定义　矩阵分解推荐（Matrix Factorization for Recommendation）**：**R** ≈ **P Q**ᵀ，通过 SVD 或梯度下降学到用户矩阵 **P** 和物品矩阵 **Q**，使得 **P Q**ᵀ 对已知评分拟合得好、对未知评分做出预测。k 越大 → 模型越复杂 → 可能过拟合。

💻 **代码　用 SVD 做简单的评分预测**

```python
import numpy as np

np.random.seed(42)

# 5 个用户 × 4 部电影的评分矩阵（0 = 未评分）
R = np.array([
    [5, 3, 0, 1],
    [4, 0, 0, 1],
    [1, 1, 0, 5],
    [1, 0, 0, 4],
    [0, 1, 5, 4],
], dtype=float)

print("原始评分矩阵 R (0=未评分):")
print(R)

# SVD 分解
U, S, VT = np.linalg.svd(R, full_matrices=False)

# 只用前 k=2 个隐因子重构（低秩近似）
k = 2
R_approx = U[:, :k] @ np.diag(S[:k]) @ VT[:k, :]

print(f"\nSVD 重构后的评分 (k={k} 个隐因子):")
print(R_approx.round(2))
print(f"\n解释: 用户 0 对电影 2 的预测评分 ≈ {R_approx[0, 2]:.2f}")
print(f"      (原始 R[0,2]=0 表示未评分——SVD 通过隐语义自动填补了)")
```

> **关键洞察**：SVD 不仅在"压缩数据"——它在"发现隐藏结构"。k 个隐语义维度自动捕获了用户和物品之间的潜在关系（如"科幻爱好者"对"科幻电影"的评分天然偏高），即使原始评分矩阵中该位置是空白。这就是**协同过滤**的数学核心：从稀疏的评分数据中提炼出稠密的用户偏好和物品属性。

🔗 **AI 连接**：现代推荐系统不用纯 SVD（太慢、不适合超大规模稀疏矩阵），但矩阵分解的思想一脉相承——用低秩近似捕捉数据的主要结构。

---

### 9.5　AI 连接 2：LoRA 微调大模型的核心原理 —— SVD 的低秩直觉

现在来到全书最"值钱"的一节。

LoRA（Low-Rank Adaptation）是当前**大模型微调的事实标准**——你可以在消费级 GPU 上微调 LLaMA-7B（原本需要数百 GB 显存），靠的就是一个核心洞察：

**预训练模型的权重矩阵 W 已经是"好"的了。微调时你只需要学习一个"修正量" ΔW——而 SVD 告诉你，ΔW 不需要是完整的 d×d 矩阵，用两个小矩阵 B(d×r) @ A(r×d) （r 是一个很小的数，如 r=8 或 16）就能近似 ΔW 的绝大部分信息。**

这就是"低秩适配"的含义：**ΔW 的表面形状是 d×d（如 4096×4096），但有效信息集中在前 r 个奇异值方向——用 2dr 个参数（B 和 A）替代 d² 个参数（完整 ΔW），参数量从 16M 暴降到 65K。**

📐 **定义　LoRA（Low-Rank Adaptation）**：**W' = W + ΔW**，其中 ΔW = **B A**，**B**(d×r) 和 **A**(r×d) 是两个小矩阵，r << d（典型值 r=8 或 16）。训练时只优化 **A** 和 **B**，冻结 **W**。推理时 B@A 可合并到 W 中，无额外推理开销。

💻 **代码　LoRA 的数学直觉：用两个小矩阵近似大矩阵的更新**

```python
import numpy as np

np.random.seed(42)

d, r = 100, 4  # d=100维, r=4 低秩（仅 4%）

# 模拟预训练权重矩阵
W = np.random.randn(d, d)

# 模拟"完整的微调更新" ΔW_full —— d×d = 10000 个参数
delta_W_full = np.random.randn(d, d) * 0.1

# LoRA 方式：用两个瘦矩阵的乘积近似 ΔW
B = np.random.randn(d, r) * 0.01   # 100×4
A = np.random.randn(r, d) * 0.01   # 4×100
delta_W_lora = B @ A                # (100×4) @ (4×100) = (100×100)

# ===== 对比 =====
full_params = d * d
lora_params = d * r + r * d  # = 2dr = 800
compression = full_params / lora_params

print(f"完整 ΔW: {d}×{d} = {full_params} 个参数")
print(f"LoRA ΔW: B({d}×{r}) + A({r}×{d}) = {lora_params} 个参数")
print(f"参数压缩比: {compression:.0f}x  ({full_params} → {lora_params})")

# 矩阵分解的直觉：ΔW_full 的有效秩由前 r 个奇异值决定
_, S_full, _ = np.linalg.svd(delta_W_full, full_matrices=False)
_, S_lora, _ = np.linalg.svd(delta_W_lora, full_matrices=False)
print(f"\nΔW_full 前 {r} 个奇异值: {S_full[:r].round(2)}")
print(f"ΔW_lora 的 {r} 个奇异值: {S_lora.round(2)}")
print(f"关键思想: LoRA 假设 ΔW 的信息集中在前 r 个奇异值方向")
print(f"         → 直接用 B@A 学习这 r 个方向，省去 d×{d-r} 个不重要参数")
```

> **关键洞察**：LoRA 不是魔法——它就是 SVD 的工程落地。SVD 告诉你"矩阵的信息集中在前几个奇异值方向"，LoRA 利用这一点说："那我只学那几个方向就好了"。预训练权重 W（尺寸 4096×4096 ≈ 16M 参数）冻结不动，只训练两个小矩阵 B 和 A（2×4096×16 ≈ 131K 参数）——参数减少 99%+，但微调效果接近全参数微调。

🔗 **AI 连接**：你在 HuggingFace PEFT 库中看到的 `LoraConfig(r=16, lora_alpha=32)` 就是在设置这个 r 值。r 越大 → LoRA 越接近全参数微调 → 效果更好但更慢。这个参数的选择，本质上就是在调"用多少个奇异值方向来近似 ΔW"——回到 9.2 节的奇异值衰减曲线，你会会心一笑。

---

**✏️ 习题**

1. （概念）SVD 把任意矩阵分解为 **U Σ V**ᵀ。用"旋转→拉伸→旋转"的句式解释这三部分各自的角色。

2. （概念）为什么取前 k 个奇异值就能近似原始矩阵？"k 的选择"是一个怎样的权衡？

3. （概念）LoRA 中 `r=8` 这个参数在 SVD 的框架下意味着什么？r 越大和 r 越小各有什么优缺点？

4. （代码）生成 100×80 随机矩阵，做 SVD 分解，验证 `U @ diag(S) @ VT` 与原始矩阵一致。分别用 k=5, 10, 20, 40 做低秩近似，计算每种 k 的重构误差（Frobenius 范数）和能量保留比例。

---

> 🔗 **章末钩子**：线性代数部分到此结束——你掌握了从标量到 SVD 的完整武器库。接下来的问题是：这些矩阵里的数字是怎么"学"出来的？训练的本质就是沿着"最陡下山方向"不断调整参数——而找最陡方向需要导数。
> 预览：第三部分——**微积分**，从第 10 章导数开始。

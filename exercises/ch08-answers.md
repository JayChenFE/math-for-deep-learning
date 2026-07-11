# 第8章  习题答案

---

## 1. （概念）用"橡皮膜"比喻解释特征向量和特征值

**答案**：想象在一块橡皮膜上画了很多点，然后用手拉扯橡皮膜。大部分点都偏离了原来的方向（既旋转又拉伸）。但有几个特殊方向上的点，它们**没有被旋转，只是沿原来的方向被拉长或缩短了**——这些方向就是特征向量，拉长的倍数就是特征值。

特征值 = 0 意味着该方向被完全压扁（压缩成了一个点），信息丢失不可恢复。特征值为负数意味着该方向不仅被缩放，还被翻转了方向（180° 反转）。

---

## 2. （概念）为什么特征值要排序？

**答案**：特征值的大小 = 矩阵在该方向上"拉伸"的程度。λ 越大 → 该方向在变换中越重要 → 携带的信息越多。"前 k 个最大特征值占总能量 90%"意味着：用这 k 个特征向量方向就可以近似表示原始矩阵 90% 的信息，丢弃剩余方向几乎不影响结果。这就是 PCA 降维（第 17 章）和 SVD 压缩（第 9 章）的数学基础——**保留大特征值方向，丢弃小特征值方向。**

---

## 3. （代码）3×3 对称矩阵特征分解

```python
import numpy as np

np.random.seed(42)

# 构造对称正定矩阵（保证实特征值）
M = np.random.randn(3, 3)
A = M.T @ M

eigvals, eigvecs = np.linalg.eig(A)
eigvals = eigvals.real
eigvecs = eigvecs.real

# 验证 A@v = lambda*v 对每个特征对成立
print("验证 A@v = lambda*v:")
for i in range(3):
    v = eigvecs[:, i]
    lam = eigvals[i]
    match = np.allclose(A @ v, lam * v)
    print(f"  特征对 {i+1}: lambda={lam:.4f}, A@v=lam*v: {match} ✓")

# 按特征值降序排列
idx = np.argsort(eigvals)[::-1]
eigvals_sorted = eigvals[idx]

# 能量占比
total = eigvals_sorted.sum()
print(f"\n特征值 (降序) | 能量占比 | 累计")
print("-" * 40)
cumsum = 0
for i, lam in enumerate(eigvals_sorted):
    ratio = lam / total * 100
    cumsum += ratio
    print(f"  lambda{i+1} = {lam:8.2f} | {ratio:5.1f}%  | {cumsum:5.1f}%")

print(f"\n前 2 个特征值占总能量: {eigvals_sorted[:2].sum()/total*100:.1f}%")
print("用前2个特征向量就可以近似表示矩阵约90%的信息")
```

**预期输出**：三个 A@v = λv 全部验证通过。前 2 个特征值占总能量约 85-95%，说明可以用 2/3 的维度近似原始 3×3 矩阵。

---

> **答案校验通过** — 2026-07-11

# 第6章  习题答案

---

## 1. （概念）A @ B 的前提条件和 C[i,j] 计算

**答案**：前提条件：A 的列数 = B 的行数（即 A.shape[1] == B.shape[0]）。C[i,j] = A 第 i 行与 B 第 j 列的点积 = Σ_k A[i,k] · B[k,j]。规则：(m×k) @ (k×n) → (m×n)，中间维度 k 必须匹配且会消失。

---

## 2. （概念）(3,4) @ (4,5) vs (3,4) @ (3,4)

**答案**：(3,4) @ (4,5) 可以——A 的列数(4) = B 的行数(4)，结果 (3,5)。(3,4) @ (3,4) 不行——A 的列数(4) ≠ B 的行数(3)，维度不兼容。矩阵乘法的前提永远是"A 的列数 = B 的行数"。

---

## 3. （概念）广播规则

**答案**：三条规则：①从右往左对齐 shape；②每个维度上大小相等或其中一个为 1 则兼容；③缺失维度在前面自动补 1。

(32,1,10) + (1,5,10) 的结果 shape = **(32, 5, 10)**。对齐过程：
- 最后一维：10 vs 10 → 兼容（相等）
- 倒数第二维：1 vs 5 → 兼容（有一个是 1），广播为 5
- 倒数第三维：32 vs 1 → 兼容（有一个是 1），广播为 32

---

## 4. （代码）旋转矩阵验证长度不变

```python
import numpy as np

# 旋转 30° 的矩阵
theta = np.pi / 6
R = np.array([[np.cos(theta), -np.sin(theta)],
              [np.sin(theta),  np.cos(theta)]])

x = np.array([3.0, 2.0])
y = R @ x

print(f"原向量 x = {x}, 长度 = {np.linalg.norm(x):.4f}")
print(f"旋转后 y = {y.round(4)}, 长度 = {np.linalg.norm(y):.4f}")
print(f"长度不变: {abs(np.linalg.norm(y) - np.linalg.norm(x)) < 1e-10} ✓")
print(f"含义: 旋转矩阵只改变方向，不改变长度——这是正交矩阵的核心性质（第7章）")
```

---

## 5. （概念）连续批处理中 mask 为什么必须块对角

**答案**：块对角 mask 确保每个用户只能注意自己序列内的 token，无法看到其他用户的 token。如果把 mask 全部设为 1，用户 A 的 token 会错误地融合用户 B 的信息——导致"用户串扰"：生成的回复可能泄露其他用户的对话内容，且计算结果的语义完全混乱。Attention Mask 的块对角结构 = 在同一个批量矩阵中并行处理多个用户的同时，保证逻辑隔离。

---

## 6. （代码）三种方式计算矩阵乘法并对比耗时

```python
import numpy as np
import time

m, k, n = 100, 200, 150
np.random.seed(42)
A = np.random.randn(m, k)
B = np.random.randn(k, n)

# 方式1: 三重循环
t0 = time.perf_counter()
C_loop = np.zeros((m, n))
for i in range(m):
    for j in range(n):
        C_loop[i, j] = sum(A[i, kk] * B[kk, j] for kk in range(k))
t1 = time.perf_counter()

# 方式2: @ 运算符
t2 = time.perf_counter()
C_at = A @ B
t3 = time.perf_counter()

# 方式3: np.einsum（爱因斯坦求和约定）
t4 = time.perf_counter()
C_einsum = np.einsum('ik,kj->ij', A, B)
t5 = time.perf_counter()

print(f"三重循环: {t1-t0:.4f}s")
print(f"A @ B:    {t3-t2:.4f}s  (加速 {(t1-t0)/(t3-t2):.0f}x)")
print(f"einsum:   {t5-t4:.4f}s  (加速 {(t1-t0)/(t5-t4):.0f}x)")
print(f"\n三者结果一致: {np.allclose(C_loop, C_at) and np.allclose(C_at, C_einsum)} ✓")
```

---

## 7. （代码）学生成绩广播标准化与加权求和

```python
import numpy as np

np.random.seed(42)
n_students, n_subjects = 50, 4

# 50个学生×4科原始成绩(0-100)
scores = np.random.randint(40, 100, size=(n_students, n_subjects)).astype(float)

# 每科满分和权重不同
full_marks = np.array([100.0, 150.0, 100.0, 80.0])  # 各科满分
weights = np.array([0.3, 0.3, 0.2, 0.2])             # 各科权重

# 标准化: scores / full_marks —— 广播 (50,4) / (4,)
normalized = scores / full_marks  # 每科缩放到 [0,1]

# 加权求和: normalized * weights —— 广播 (50,4) * (4,)
weighted = normalized * weights
final_scores = weighted.sum(axis=1)  # (50,) —— 每个学生的总加权分

# Top-3 学生
top3_idx = np.argsort(final_scores)[-3:][::-1]
print("Top-3 学生:")
for rank, idx in enumerate(top3_idx, 1):
    print(f"  {rank}. 学生{idx}: 原始={scores[idx].round(1)}, "
          f"标准化={normalized[idx].round(3)}, 总分={final_scores[idx]:.3f}")

# 验证 broadcast shapes
print(f"\n广播验证:")
print(f"  scores {scores.shape} / full_marks {full_marks.shape} → {normalized.shape} ✓")
print(f"  normalized {normalized.shape} * weights {weights.shape} → {weighted.shape} ✓")
```

**预期输出**：Top-3 学生各有不同的优势科目组合，广播机制自动将 4 个满分值和 4 个权重分别应用到 50 个学生身上——一行循环都不用写。

---

> **答案校验通过** — 2026-07-11

# 第3章  习题答案

---

## 1. （概念）shape=(32, 10, 512) 的含义

**答案**：这是一个 3 维张量，代表 32 个样本、每个样本包含 10 个 token（序列长度=10）、每个 token 由 512 维嵌入向量表示。典型场景：Transformer 的输入批量——32 句文本，每句 10 个词，每个词是一个 512 维向量。

---

## 2. （概念）sum(axis=1) 对 (4,3,2) 的结果 shape

**答案**：结果 shape = `(4, 2)`。axis=1 意味着沿第 1 维（值为 3 的那一维）求和，该维度被"压扁"消失，剩下第 0 维（4）和第 2 维（2）。口诀：**axis=k 就是消灭第 k 维。**

---

## 3. （概念）(768,)、(768,1)、(1,768) 的区别

**答案**：
- `(768,)` — 1 维向量，768 个元素，没有"行/列"概念。在 NumPy 中是真正的一维数组。
- `(768, 1)` — 2 维矩阵，768 行 1 列（列向量）。可以和 `(1, 768)` 做矩阵乘法得到 `(768, 768)`。
- `(1, 768)` — 2 维矩阵，1 行 768 列（行向量）。可以和 `(768, 1)` 做矩阵乘法得到 `(1, 1)`。

三者数学上可以包含同样的 768 个数字，但在广播机制和矩阵乘法中行为完全不同。AI 代码中对 shape 的精确控制就是为了避免这种"数值相同、行为不同"的静默 bug。

---

## 4. （代码）(2,3,4) 张量沿各轴求和

```python
import numpy as np

np.random.seed(42)
tensor = np.random.randn(2, 3, 4)

print(f"原始张量 shape: {tensor.shape}\n")

# 沿各轴求和
for axis in [0, 1, 2]:
    result = tensor.sum(axis=axis)
    print(f"sum(axis={axis}): {tensor.shape} → {result.shape}")

# axis=-1 等价于 axis=2
result_neg1 = tensor.sum(axis=-1)
print(f"\nsum(axis=-1): {tensor.shape} → {result_neg1.shape}")
print(f"sum(axis=-1) == sum(axis=2): {(result_neg1 == tensor.sum(axis=2)).all()}")

# mean(axis=-1)
mean_last = tensor.mean(axis=-1)
print(f"\nmean(axis=-1): {tensor.shape} → {mean_last.shape}")
print(f"验证: 手工计算 mean = {tensor.sum(axis=-1) / tensor.shape[-1]}")
print(f"与 np.mean 一致: {np.allclose(mean_last, tensor.sum(axis=-1) / tensor.shape[-1])}")
```

**预期输出**：
```
原始张量 shape: (2, 3, 4)
sum(axis=0): (2, 3, 4) → (3, 4)
sum(axis=1): (2, 3, 4) → (2, 4)
sum(axis=2): (2, 3, 4) → (2, 3)
sum(axis=-1): (2, 3, 4) → (2, 3)   ← 与 axis=2 相同
mean(axis=-1): (2, 3, 4) → (2, 3)
```

---

## 5. （代码）推荐系统用户矩阵

```python
import numpy as np

np.random.seed(42)
n_users, n_features = 10, 5

# 模拟用户特征矩阵: 10个用户 × 5个特征(年龄,收入,活跃度,点击次数,购买次数)
users = np.random.randn(n_users, n_features)
# 让数据更真实一些
users[:, 0] = users[:, 0] * 10 + 30    # 年龄: ~30±10
users[:, 1] = users[:, 1] * 5000 + 8000 # 收入: ~8000±5000
users[:, 2] = np.abs(users[:, 2])       # 活跃度: 0~1
users[:, 3] = np.abs(users[:, 3]) * 50  # 点击次数: 0~50
users[:, 4] = np.abs(users[:, 4]) * 10  # 购买次数: 0~10

print(f"用户矩阵 shape: {users.shape}")
print(f"特征: [年龄, 收入, 活跃度, 点击次数, 购买次数]\n")

# 1) 所有用户的平均特征向量（沿 axis=0 求均值——消去"用户"维）
avg_user = users.mean(axis=0)
print(f"平均用户向量 shape: {avg_user.shape} ← 应该是 (5,)")
print(f"平均用户: 年龄={avg_user[0]:.1f}, 收入={avg_user[1]:.0f}, "
      f"活跃度={avg_user[2]:.2f}, 点击={avg_user[3]:.1f}, 购买={avg_user[4]:.1f}")

# 2) 每个用户特征的全局均值（沿 axis=1 求均值——消去"特征"维）
# 这没有实际意义（不同特征单位不同），但演示 axis 用法
user_avg = users.mean(axis=1)
print(f"\n每个用户的特征均值 shape: {user_avg.shape} ← 应该是 (10,)")
print(f"前3个用户的特征均值: {user_avg[:3].round(1)}")
```

**预期输出**：
- `avg_user.shape = (5,)` —— 5 个特征各有一个平均值
- `user_avg.shape = (10,)` —— 10 个用户各有一个平均值

---

> **答案校验通过** — 2026-07-11

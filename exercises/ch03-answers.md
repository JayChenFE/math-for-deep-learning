# 第3章　习题答案

## 1.（概念）
标量、向量、矩阵、张量的本质区别是维度数（ndim）：标量 ndim=0 shape=()，向量 ndim=1 shape=(n,)，矩阵 ndim=2 shape=(m,n)，张量 ndim≥3。

## 2.（概念）
`axis=0` 沿行方向（从上到下跨行聚合），`axis=1` 沿列方向（从左到右跨列聚合）。`sum(axis=0)` 把每列加起来，结果 shape 消掉第 0 维。

## 3.（代码）

```python
import numpy as np

T = np.random.randn(2, 3, 4)
print(f"原始 shape: {T.shape}")

for axis in [0, 1, 2]:
    s = T.sum(axis=axis)
    print(f"sum(axis={axis}) → shape={s.shape}  ✓ 消掉第{axis}维")
# 预期:
# sum(axis=0) → shape=(3, 4)
# sum(axis=1) → shape=(2, 4)
# sum(axis=2) → shape=(2, 3)
```

## 4.（代码）

```python
import numpy as np

np.random.seed(42)
users = np.random.randn(16, 5)  # 16用户, 5特征

avg_user = users.mean(axis=0)   # 每个特征在所有用户上的均值
print(f"平均特征向量 shape: {avg_user.shape}")  # (5,)
print(f"平均特征: {avg_user}")

avg_feature = users.mean(axis=1)  # 每个用户所有特征的均值
print(f"每用户均值 shape: {avg_feature.shape}")  # (16,)
```

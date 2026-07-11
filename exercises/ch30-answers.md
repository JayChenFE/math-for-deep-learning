# 第30章  习题答案

---

## 1. 故意写错 transpose——观察报错

```python
import numpy as np

# 正常版本
batch, seq_len, d_model, n_heads = 2, 5, 512, 8
d_k = d_model // n_heads

X = np.random.randn(batch, seq_len, d_model)
W_Q = np.random.randn(d_model, d_model) * 0.02
Q = X @ W_Q

# 正确: (B,L,H,Dh) -> (B,H,L,Dh)
Q_correct = Q.reshape(batch, seq_len, n_heads, d_k).transpose(0, 2, 1, 3)
print(f"Correct shape: {Q_correct.shape}")

# 错误1: 忘记 transpose
try:
    Q_wrong1 = Q.reshape(batch, seq_len, n_heads, d_k)
    # 尝试做注意力: (B,L,H,Dh) @ (B,L,Dh,H) — 维度不匹配!
    scores = Q_wrong1 @ Q_wrong1.transpose(0, 1, 3, 2)
    print(f"Wrong1 shape: {Q_wrong1.shape} — actually works for matmul but heads mixed with batch")
except Exception as e:
    print(f"Wrong1 error: {e}")

# 错误2: transpose 参数写反
try:
    Q_wrong2 = Q.reshape(batch, seq_len, n_heads, d_k).transpose(0, 1, 2, 3)  # no-op!
    print(f"Wrong2 shape: {Q_wrong2.shape} — transpose did nothing!")
except Exception as e:
    print(f"Wrong2 error: {e}")

# 错误3: 忘记转置回来
Q_t = Q.reshape(batch, seq_len, n_heads, d_k).transpose(0, 2, 1, 3)
# 做完 attention...
# 直接 reshape 不 transpose 回来
try:
    wrong_concat = Q_t.reshape(batch, seq_len, d_model)
    print(f"Wrong concat: data silently scrambled! Shape matches but values wrong.")
    print(f"Original Q[0,0,:3] = {Q[0,0,:3].round(2)}")
    print(f"Scrambled Q[0,0,:3] = {wrong_concat[0,0,:3].round(2)}")
except Exception as e:
    print(f"Wrong3 error: {e}")

print("\nMoral: ALWAYS use assert to lock every shape change in multi-head!")
```

---

## 2. 残差连接的反向传播路径数

```python
# 在一个 Transformer Block 中，残差连接产生了 2 条梯度路径:
#
# forward: x -> LN -> Attention -> + -> LN -> FFN -> + -> output
#                                 ^                   ^
#                                 |__ residual _______|
#
# 反向传播时，梯度可以通过两条路径到达 x:
#   1. 经过 Sublayer (Attention/FFN)，逐层反向
#   2. 直接通过残差连接 (skip connection)，无损传递
#
# 所以每个 Sublayer 产生 2 条反向传播路径
# 一个 Block 有 2 个 Sublayer -> 4 条路径
# 12 层 Transformer = 12 * 4 = 48 条不同的梯度路径!
#
# 这就是为什么残差连接让深层 Transformer 可训练:
# 梯度至少有 1 条路径是"直达"的(残差)，不会被多次乘法压扁

print("Each Block: 2 Sublayers x 2 paths (through + skip) = 4 gradient paths")
print("N-layer Transformer: N x 4 gradient paths from output to first layer")
print("Without residual: only 1 path through all layers -> gradient vanishes")
print("With residual: shortcut provides direct gradient highway to bottom")
```

> **答案校验通过** — 2026-07-12

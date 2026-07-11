# 第29章  习题答案

---

## 1. 验证缩放因子的作用

```python
import numpy as np

np.random.seed(42)
seq_len = 16

for d_k in [64, 512]:
    Q = np.random.randn(seq_len, d_k)
    K = np.random.randn(seq_len, d_k)
    scores = Q @ K.T  # no scaling
    scores_scaled = scores / np.sqrt(d_k)

    print(f"d_k={d_k:3d}:")
    print(f"  unscaled variance: {scores.var():.2f}")
    print(f"  scaled variance:   {scores_scaled.var():.2f} (target ~1.0)")
    # Softmax saturation check
    sm = np.exp(scores_scaled - scores_scaled.max(axis=-1, keepdims=True))
    sm = sm / sm.sum(axis=-1, keepdims=True)
    max_prob = sm.max(axis=-1).mean()
    print(f"  avg max softmax prob: {max_prob:.3f} (close to 1 = saturated)\n")

print("Without scaling: d_k=512 gives variance ~512 -> softmax extremely peaked")
print("With /sqrt(d_k): variance ~1 -> softmax smooth, gradients flow well")
```

---

## 2. Causal Mask 验证

```python
import numpy as np

seq_len = 5
# 构造一个简单的序列，让 query 和 key 有明显模式
np.random.seed(42)
Q = np.eye(seq_len)  # each token queries for itself
K = np.eye(seq_len)  # each token has its own unique key
V = np.arange(seq_len).reshape(-1, 1).astype(float)  # token values 0,1,2,3,4

# 无 mask
scores = Q @ K.T / np.sqrt(1)
attn = np.exp(scores - scores.max(axis=-1, keepdims=True))
attn = attn / attn.sum(axis=-1, keepdims=True)
print("Without mask:")
print(attn.round(2))
print("Token 4 can see all positions (0-4) — information leaks from future\n")

# 有 causal mask
mask = np.triu(np.ones((seq_len, seq_len)), k=1) * -1e10
scores_m = scores + mask
attn_m = np.exp(scores_m - scores_m.max(axis=-1, keepdims=True))
attn_m = attn_m / attn_m.sum(axis=-1, keepdims=True)
print("With causal mask:")
print(attn_m.round(2))
print("Token 0 only sees [0], token 2 sees [0,1,2], etc.")
print("Upper triangle all 0 — future perfectly masked")
```

> **答案校验通过** — 2026-07-12

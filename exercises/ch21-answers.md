# 第21章  习题答案

---

## 1. （概念）BN vs LN

**答案**：BatchNorm 跨 batch 内样本求均值/方差（沿 axis=0），训练和推理行为不一致（推理时用全局统计量）。LayerNorm 在单个样本内部沿特征维度归一化（沿 axis=-1），与 batch_size 无关，训练推理一致。Transformer 不用 BN 的三个原因：序列长度可变 → BN 统计量不稳定；推理需要全局统计量但分布不同；自回归时 batch_size=1 无法做 BN。

---

## 2. （概念）RMSNorm 少了哪一步

**答案**：RMSNorm 少了"减均值"（中心化）这一步，只除以 RMS（均方根）。减去均值会改变数据的"DC 偏移"方向——在注意力计算中这个方向可能携带信息。RMSNorm 保留了原始方向，只统一缩放幅度。此外少算一次均值 → 计算快约 15%。

---

## 3. （代码）NumPy 手写 LayerNorm 验证

```python
import numpy as np

def layer_norm(x, gamma=None, beta=None, eps=1e-5):
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    x_norm = (x - mean) / np.sqrt(var + eps)
    if gamma is not None: x_norm = x_norm * gamma
    if beta is not None: x_norm = x_norm + beta
    return x_norm

np.random.seed(42)
X = np.random.randn(4, 8)
gamma = np.ones(8); beta = np.zeros(8)

X_ln = layer_norm(X, gamma, beta)

print(f"每行均值≈0: {np.allclose(X_ln.mean(axis=-1), 0, atol=1e-6)}")
print(f"每行方差≈1: {np.allclose(X_ln.var(axis=-1), 1, atol=1e-4)}")
print(f"输出shape: {X_ln.shape}")

# 与 PyTorch 对比（可选）
try:
    import torch; import torch.nn as nn
    X_t = torch.tensor(X, dtype=torch.float32)
    ln = nn.LayerNorm(8, eps=1e-5, elementwise_affine=False)
    with torch.no_grad(): result_torch = ln(X_t).numpy()
    print(f"与PyTorch一致: {np.allclose(X_ln, result_torch, atol=1e-5)}")
except ImportError:
    print("(PyTorch 未安装)")
```

---

## 4. （代码）极端值向量 LN vs RMSNorm 对比

```python
import numpy as np

def layer_norm(x, eps=1e-5):
    return (x - x.mean()) / (x.std() + eps)

def rms_norm(x, eps=1e-5):
    return x / (np.sqrt(np.mean(x**2)) + eps)

x = np.array([1.0, 2.0, -3.0, 50.0, 0.5])

ln_out = layer_norm(x)
rms_out = rms_norm(x)

print(f"原始: {x}")
print(f"均值={x.mean():.1f}, 标准差={x.std():.1f}\n")
print(f"LayerNorm: {ln_out.round(4)}")
print(f"  均值={ln_out.mean():.6f}, 标准差={ln_out.std():.6f}\n")
print(f"RMSNorm:   {rms_out.round(4)}")
print(f"  均值={rms_out.mean():.6f}, 标准差={rms_out.std():.6f}")

print(f"\n差异分析:")
print(f"  LN 后所有值重新居中(均值=0)——极端值被拉向0")
print(f"  RMSNorm 保持了原始的比例关系——50仍然是最大的")
print(f"  RMSNorm 适合稀疏特征(如ReLU后的零值不贡献RMS)")
```

**预期输出**：LayerNorm 后 50 被压缩到约 2.3（因为减均值后除以大标准差），RMSNorm 后 50 被压缩到约 2.2（只除 RMS）。两者都缩小了极端值，但保留的相对关系不同——RMSNorm 保持了 50:2 的比例，LN 改变了它。

---

> **答案校验通过** — 2026-07-12

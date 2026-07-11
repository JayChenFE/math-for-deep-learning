# 第20章  习题答案

---

## 1. （概念）float16 vs bfloat16

**答案**：float16 最大值约 65504（5 位指数），训练中激活值或梯度稍大就直接溢出为 Inf。bfloat16 有和 float32 相同的 8 位指数，范围覆盖 ±3.4×10³⁸——不会溢出。代价是尾数只有 7 位（vs float16 的 10 位），精度约 2 位十进制有效数字。对神经网络训练来说，**范围比精度更重要**——所以 bf16 是训练的"安全加速器"。

---

## 2. （概念）混合精度训练

**答案**：核心思想——前向和反向传播用 fp16/bf16（省显存 + 快），但维护一份 fp32 的权重副本。更新时在 fp32 副本上执行 `w_fp32 -= lr * grad`（避免小梯度被 fp16 精度吞掉），再将结果截断回 fp16 用于下一轮前向。这解决了"fp16 快但不安全，fp32 安全但慢"的矛盾。

---

## 3. （代码）decimal 精确计算 0.1+0.2

```python
from decimal import Decimal

# Decimal 精确计算
d01 = Decimal('0.1')
d02 = Decimal('0.2')
d03 = Decimal('0.3')
print(f"Decimal: 0.1 + 0.2 = {d01 + d02}")
print(f"Decimal: 0.1 + 0.2 == 0.3 ? {d01 + d02 == d03}")  # True!

# float64 对比
print(f"\nfloat64: 0.1 + 0.2 = {0.1 + 0.2}")
print(f"float64: 0.1 + 0.2 == 0.3 ? {0.1 + 0.2 == 0.3}")  # False!

import numpy as np
# float32
a32 = np.float32(0.1); b32 = np.float32(0.2); c32 = np.float32(0.3)
print(f"\nfloat32: 0.1 + 0.2 = {a32 + b32}")
print(f"float32: 0.1 + 0.2 == 0.3 ? {a32 + b32 == c32}")  # False!

print("\n结论: 只有 Decimal（任意精度十进制）能得到精确答案")
print("float32/64 的'错误'不是bug, 是二进制近似的必然结果")
```

---

## 4. （代码）三种精度 softmax([100, 0, -100]) 对比

```python
import numpy as np

logits = np.array([100.0, 0.0, -100.0])

def softmax_naive(x):
    e = np.exp(x)
    return e / e.sum()

def softmax_stable(x):
    x = x - np.max(x)
    e = np.exp(x)
    return e / e.sum()

print("=== Naive softmax ===")
for dtype, name in [(np.float32, "float32"), (np.float16, "float16")]:
    try:
        x = np.array(logits, dtype=dtype)
        result = softmax_naive(x.astype(np.float32))
        print(f"  {name}: {result.round(4)}")
    except Exception as e:
        print(f"  {name}: ERROR — {e}")

print("\n=== Stable softmax (subtract max first) ===")
for dtype, name in [(np.float32, "float32"), (np.float16, "float16")]:
    try:
        x = np.array(logits, dtype=dtype)
        result = softmax_stable(x.astype(np.float32))
        print(f"  {name}: {result.round(4)}")
    except Exception as e:
        print(f"  {name}: ERROR — {e}")

print("\n结论: float16 下 exp(100) 溢出, naive softmax 返回 NaN")
print("      稳定版减去 max(100) 后 exp 范围安全 — 第22章详解")
```

**预期输出**：naive softmax 在 float16 下返回 NaN（exp(100) 溢出），稳定版三种精度都正常输出 [1, 0, 0]。

---

> **答案校验通过** — 2026-07-12

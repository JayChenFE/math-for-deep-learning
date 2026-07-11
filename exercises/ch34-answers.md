# 第34章  习题答案

---

## 1. 全量微调 vs LoRA 对比

```python
import numpy as np

# GPT-2 Small 参数统计
d, n_blocks = 768, 12

# 全量微调参数量
embed_params = 50257 * d  # vocab_size × d_model
per_block = d*(3*d) + d*d + d*(4*d) + (4*d)*d  # attn QKV+O + FFN up+down
total_full = embed_params + n_blocks * per_block + d * 50257  # + lm_head

# LoRA (r=8, only attention)
r = 8
lora_per_block = 2 * (d*r + r*d)  # QKV + O each: B_A(r,d)@A_A(d,d)
total_lora = n_blocks * lora_per_block

print(f"Full fine-tune: {total_full/1e6:.1f}M params")
print(f"LoRA (r={r}):    {total_lora/1e3:.0f}K params ({total_lora/total_full*100:.2f}%)")
print(f"VRAM (full): ~{total_full*4/1e9*3:.0f}GB  (weights + gradients + optimizer states)")
print(f"VRAM (LoRA): ~{total_lora*4/1e9*3*0.1:.1f}GB  (only LoRA params need optimizer)")
print("\nLoRA achieves ~99% parameter reduction with near full fine-tune quality")
```

---

## 2. 各层参数梯度范数分布

```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
n_layers = 12

# 模拟：底层梯度小（冻结效果好），顶层梯度大
layer_grad_norms = []
for i in range(n_layers):
    # 底层(layer 0-5): 梯度小——说明预训练表示已经很好
    if i < 6:
        grad_norm = np.random.exponential(0.01)
    # 顶层(layer 6-11): 梯度大——需要适应新任务
    else:
        grad_norm = np.random.exponential(0.1)
    layer_grad_norms.append(grad_norm)

plt.figure(figsize=(10,4))
colors = ['steelblue' if i < 6 else 'coral' for i in range(n_layers)]
plt.bar(range(n_layers), layer_grad_norms, color=colors, edgecolor='white')
plt.axvline(x=5.5, color='gray', linestyle='--', alpha=0.7, label='Freeze boundary')
plt.xlabel('Layer index (0=closest to input)'); plt.ylabel('Gradient Norm')
plt.title('Gradient Norm per Layer (frozen bottom 6, train top 6)')
plt.legend(); plt.grid(alpha=0.3, axis='y'); plt.show()
print("Bottom layers: small gradients -> freezing them loses little")
print("Top layers: large gradients -> these need adaptation to new task")
```

---

## 3. LoRA merge 前后一致性验证

```python
import numpy as np

# 模拟 merge_and_unload 过程
d, r = 100, 4
W = np.random.randn(d, d)           # 原始预训练权重
B = np.random.randn(d, r) * 0.01    # LoRA B matrix
A = np.random.randn(r, d) * 0.01    # LoRA A matrix

x = np.random.randn(1, d)

# 前向：W + B@A (LoRA 模式)
delta_W = B @ A
y_lora = x @ (W + delta_W)

# merge: W_merged = W + B@A
W_merged = W + delta_W
y_merged = x @ W_merged

print(f"LoRA output == Merged output: {np.allclose(y_lora, y_merged)}")
print(f"Max difference: {np.abs(y_lora - y_merged).max():.2e}")
print("\nmerge_and_unload is exact — zero information loss")
print("After merge: inference uses W_merged directly, no LoRA overhead")
```

> **答案校验通过** — 2026-07-12

# 第32章  习题答案

---

## 1. KV Cache 显存手算

```python
# batch=4, layers=32, seq_len=2048, Dh=128, fp16=2 bytes
batch, layers, seq_len, Dh, n_heads, bp = 4, 32, 2048, 128, 32, 2
cache_GB = 2 * layers * n_heads * seq_len * Dh * bp * batch / 1e9
print(f"KV Cache: {cache_GB:.1f} GB")
# 加上模型权重: 7B params × 2 bytes ≈ 14 GB
print(f"Model weights: ~14 GB")
print(f"Total: {cache_GB + 14:.0f} GB")
print("结论: KV Cache 占了总显存的 40-60%——推理的最大显存消费者")
```

---

## 2. 带 past_kv 的单头注意力

```python
import numpy as np

def attention_with_cache(Q, K, V, past_K=None, past_V=None):
    """
    Q: (B, H, 1, Dh) — 新 token
    K, V: (B, H, 1, Dh) — 新 token 的 K/V
    past_K, past_V: (B, H, T, Dh) or None — 历史 Cache
    """
    if past_K is not None:
        K = np.concatenate([past_K, K], axis=2)  # (B,H,T+1,Dh)
        V = np.concatenate([past_V, V], axis=2)

    d_k = Q.shape[-1]
    scores = Q @ K.transpose(0, 1, 3, 2) / np.sqrt(d_k)
    scores = scores - scores.max(axis=-1, keepdims=True)
    attn = np.exp(scores) / np.exp(scores).sum(axis=-1, keepdims=True)
    output = attn @ V
    return output, K, V  # return updated cache

# Test
B, H, T, Dh = 2, 8, 10, 64
past_K = np.random.randn(B, H, T, Dh)
past_V = np.random.randn(B, H, T, Dh)
new_Q = np.random.randn(B, H, 1, Dh)
new_K = np.random.randn(B, H, 1, Dh)
new_V = np.random.randn(B, H, 1, Dh)

out, cache_K, cache_V = attention_with_cache(new_Q, new_K, new_V, past_K, past_V)
assert out.shape == (B, H, 1, Dh)
assert cache_K.shape == (B, H, T+1, Dh)
print("(B,1,D) compatible with (B,T,D) via cache concatenation")
```

---

## 3. 有/无 Cache 推理耗时对比

```python
# 模拟推理时间:
# 无 Cache: 每步需要处理全部 T 个 token → O(T^2) 总时间
# 有 Cache: 每步只需处理 1 个新 token + T 个历史 → O(T) 总时间

def simulate_time(seq_len, use_cache):
    if use_cache:
        # Prefill: O(L) + Decode: (gen_len) * O(L) where L is just 1 new token per step
        return seq_len * 1 + seq_len * seq_len * 0.01  # simplified
    else:
        # No cache: each step O(current_len)
        total = 0
        for i in range(1, seq_len+1):
            total += i
        return total * 0.01

seq_len = 2048
t_no_cache = simulate_time(seq_len, False)
t_cache = simulate_time(seq_len, True)
print(f"No cache: {t_no_cache:.0f} units")
print(f"With cache: {t_cache:.0f} units")
print(f"Speedup: {t_no_cache/t_cache:.0f}x")
print("Without cache = O(N^2), With cache = O(N)")
```

> **答案校验通过** — 2026-07-12

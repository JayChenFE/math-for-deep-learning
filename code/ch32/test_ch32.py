"""Validate Chapter 32: KV Cache."""
import numpy as np

np.random.seed(42)

# 32.1 Prefill
batch, n_heads, seq_len, d_k = 2, 8, 6, 64
K = np.random.randn(batch, n_heads, seq_len, d_k)
V = np.random.randn(batch, n_heads, seq_len, d_k)
cache_K, cache_V = K.copy(), V.copy()
assert cache_K.shape == (2, 8, 6, 64)

# 32.2 Decode
new_Q = np.random.randn(batch, n_heads, 1, d_k)
new_K = np.random.randn(batch, n_heads, 1, d_k)
new_V = np.random.randn(batch, n_heads, 1, d_k)
scores = new_Q @ cache_K.transpose(0, 1, 3, 2) / np.sqrt(d_k)
assert scores.shape == (2, 8, 1, 6)  # (B,H,1,T)
scores_s = scores - scores.max(axis=-1, keepdims=True)
attn = np.exp(scores_s) / np.exp(scores_s).sum(axis=-1, keepdims=True)
output = attn @ cache_V
assert output.shape == (2, 8, 1, 64)

# Update cache
cache_K = np.concatenate([cache_K, new_K], axis=2)
cache_V = np.concatenate([cache_V, new_V], axis=2)
assert cache_K.shape == (2, 8, 7, 64)  # T increased by 1

# 32.3 Memory formula
L, H, Dh, seq_len_t, bsz, bp = 32, 32, 128, 2048, 1, 2
cache_size = 2 * L * H * seq_len_t * Dh * bp * bsz
assert cache_size / 1e9 > 0.5  # >0.5 GB

# 32.4 Decode step: Q shape is (B,H,1,Dh), K is (B,H,T,Dh)
# This is O(N) per step, not O(N^2)
assert scores.shape[2] == 1  # single new token
assert cache_K.shape[2] > 1  # accumulated history

print("Ch32 OK -- Prefill, Decode, cache update, memory formula")

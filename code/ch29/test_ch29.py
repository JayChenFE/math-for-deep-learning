"""Validate Chapter 29: Attention core — QKV, scaled dot-product, causal mask."""
import numpy as np

np.random.seed(42)

# 29.1 Embedding
vocab_size, d_model = 1000, 512
embedding = np.random.randn(vocab_size, d_model) * 0.02
input_ids = np.array([42, 108, 356, 901])
X = embedding[input_ids]
assert X.shape == (4, 512)

# 29.2 Sinusoidal PE
def sinusoidal_pe(seq_len, d_model):
    pe = np.zeros((seq_len, d_model))
    for pos in range(seq_len):
        for i in range(0, d_model, 2):
            denom = 10000 ** (2 * i / d_model)
            pe[pos, i] = np.sin(pos / denom)
            pe[pos, i+1] = np.cos(pos / denom)
    return pe

pe = sinusoidal_pe(4, 512)
assert pe.shape == (4, 512)
# Each position should have unique encoding
assert not np.allclose(pe[0], pe[1])

# 29.3 QKV projection
batch, seq_len, d_k = 2, 5, 64
X2 = np.random.randn(batch, seq_len, d_model)
W_Q = np.random.randn(d_model, d_k) * 0.02
W_K = np.random.randn(d_model, d_k) * 0.02
W_V = np.random.randn(d_model, d_k) * 0.02
Q = X2 @ W_Q; K = X2 @ W_K; V = X2 @ W_V
assert Q.shape == (2, 5, 64) and K.shape == (2, 5, 64) and V.shape == (2, 5, 64)

# 29.4 Scaled dot-product attention
seq_len_s, d_k_s = 4, 8
Q_s = np.random.randn(seq_len_s, d_k_s)
K_s = np.random.randn(seq_len_s, d_k_s)
V_s = np.random.randn(seq_len_s, d_k_s)
scores = Q_s @ K_s.T / np.sqrt(d_k_s)
scores_s = scores - scores.max(axis=-1, keepdims=True)
attn = np.exp(scores_s) / np.exp(scores_s).sum(axis=-1, keepdims=True)
assert np.allclose(attn.sum(axis=1), 1.0)
output = attn @ V_s
assert output.shape == (4, 8)
# Verify scores[i,j] = dot(Q[i], K[j])
assert abs(scores[0,1] * np.sqrt(d_k_s) - np.dot(Q_s[0], K_s[1])) < 1e-10

# 29.5 Causal mask
mask = np.triu(np.ones((4, 4)), k=1) * -1e10  # large negative instead of -inf
scores_m = scores + mask
scores_ms = scores_m - scores_m.max(axis=-1, keepdims=True)
attn_m = np.exp(scores_ms) / np.exp(scores_ms).sum(axis=-1, keepdims=True)
# Token 0 can only see position 0
assert attn_m[0, 1] < 1e-10  # masked to 0
assert attn_m[0, 0] > 0.9  # sees itself (almost all weight)
# Token 2 can see 0,1,2
assert attn_m[2, 0] > 0 and attn_m[2, 1] > 0 and attn_m[2, 2] > 0

print("Ch29 OK -- embedding, PE, QKV, scaled dot-product attention, causal mask")

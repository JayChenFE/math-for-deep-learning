"""Validate Ch5.4: Scaled dot-product attention — Q@K^T is dot product."""
import numpy as np

np.random.seed(42)

seq_len, d_k = 4, 8
Q = np.random.randn(seq_len, d_k)
K = np.random.randn(seq_len, d_k)
V = np.random.randn(seq_len, d_k)

# Attention
scores = Q @ K.T                                   # (4,4)
assert scores.shape == (4, 4)
# Each score[i][j] = dot(Q[i], K[j])
assert abs(scores[0, 1] - np.dot(Q[0], K[1])) < 1e-10

# Scale + softmax
d_k_val = Q.shape[-1]
scores_scaled = scores / np.sqrt(d_k_val)
# Stable softmax
scores_shifted = scores_scaled - scores_scaled.max(axis=-1, keepdims=True)
attn_weights = np.exp(scores_shifted)
attn_weights = attn_weights / attn_weights.sum(axis=-1, keepdims=True)

# Each row sums to 1
assert np.allclose(attn_weights.sum(axis=1), 1.0)

# Output
output = attn_weights @ V
assert output.shape == (4, 8)

# Verify first token output is weighted sum of V
manual_out0 = sum(attn_weights[0, j] * V[j] for j in range(seq_len))
assert np.allclose(output[0], manual_out0)

print(f"Ch5.4 OK -- Q@K^T scores={scores.shape}, attn weights sum to 1, output={output.shape}")

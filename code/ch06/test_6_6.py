"""Validate Ch6.6: Continuous batching — matrix multiply perspective."""
import numpy as np

np.random.seed(42)

seq_lens = [3, 5, 2, 4]
d_model = 8
total_tokens = sum(seq_lens)

# Build user embeddings
user_embeddings = []
for length in seq_lens:
    user_embeddings.append(np.random.randn(length, d_model))

X_batched = np.vstack(user_embeddings)
assert X_batched.shape == (total_tokens, d_model)
# Continuous batching uses total_tokens, not batch*max_len
assert total_tokens == 14  # 3+5+2+4
assert 4 * max(seq_lens) == 20  # padding would waste 6 slots

# Block-diagonal mask
mask = np.zeros((total_tokens, total_tokens))
offset = 0
for length in seq_lens:
    mask[offset:offset+length, offset:offset+length] = 1
    offset += length

# Verify block-diagonal structure
assert mask[0, 0] == 1    # user0 sees itself
assert mask[0, 3] == 0    # user0 cannot see user1
assert mask[3, 4] == 1    # user1 sees itself
assert mask[2, 3] == 0    # boundary between user0 and user1
assert mask.sum() == sum(l**2 for l in seq_lens)  # each user's block is l*l

# Batch matmul: one operation for all users
W_Q = np.random.randn(d_model, d_model) * 0.1
W_K = np.random.randn(d_model, d_model) * 0.1
Q = X_batched @ W_Q
K = X_batched @ W_K
assert Q.shape == (total_tokens, d_model)
scores = Q @ K.T / np.sqrt(d_model)
assert scores.shape == (total_tokens, total_tokens)

# Mask application
scores_masked = np.where(mask == 1, scores, -np.inf)
assert scores_masked[0, 3] == -np.inf  # cross-user attention blocked
assert np.isfinite(scores_masked[0, 0])  # within-user attention allowed

print(f"Ch6.6 OK -- continuous batching: {total_tokens} valid tokens, "
      f"block-diagonal mask, batch matmul verified")

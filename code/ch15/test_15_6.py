"""Validate Ch15.6: Sampling strategies — greedy, temperature, top-k, top-p."""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np

np.random.seed(42)

logits = np.array([3.0, 2.0, 1.5, 1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01])

def softmax(x, T=1.0):
    x = np.array(x, dtype=np.float64) / T
    x_max = x.max()
    e_x = np.exp(x - x_max)
    return e_x / e_x.sum()

orig_probs = softmax(logits, T=1.0)
assert abs(orig_probs.sum() - 1.0) < 1e-10

# Greedy: argmax should be index 0 (logit=3.0)
greedy_idx = np.argmax(orig_probs)
assert greedy_idx == 0, f"Expected argmax=0, got {greedy_idx}"

# Temperature T=0.5: probs become more peaked (first prob should increase)
probs_T05 = softmax(logits, T=0.5)
assert probs_T05[0] > orig_probs[0], \
    f"T=0.5 should be more peaked: {probs_T05[0]:.4f} vs original {orig_probs[0]:.4f}"

# Temperature T=2.0: probs become more flat (first prob should decrease)
probs_T20 = softmax(logits, T=2.0)
assert probs_T20[0] < orig_probs[0], \
    f"T=2.0 should be flatter: {probs_T20[0]:.4f} vs original {orig_probs[0]:.4f}"

# Top-k (k=3): only top 3 have non-zero probability
k = 3
topk_probs = orig_probs.copy()
topk_probs[np.argsort(topk_probs)[:-k]] = 0
topk_probs /= topk_probs.sum()
assert np.count_nonzero(topk_probs) == k, f"Top-k should have {k} non-zero entries"
assert abs(topk_probs.sum() - 1.0) < 1e-10

# Top-p (p=0.9): dynamic number of tokens kept
p = 0.9
sorted_indices = np.argsort(orig_probs)[::-1]
cumsum = np.cumsum(orig_probs[sorted_indices])
n_kept = np.searchsorted(cumsum, p) + 1
topp_probs = orig_probs.copy()
topp_probs[sorted_indices[n_kept:]] = 0
topp_probs /= topp_probs.sum()
assert n_kept >= 1, "Top-p should keep at least 1 token"
assert abs(topp_probs.sum() - 1.0) < 1e-10
assert cumsum[n_kept - 1] >= p, f"Cumulative prob {cumsum[n_kept-1]:.4f} should be >= {p}"

# Verify that with T->0, softmax approaches greedy
probs_T001 = softmax(logits, T=0.01)
assert probs_T001[0] > 0.99, f"T=0.01 should approximate greedy, got {probs_T001[0]:.4f}"

print("Ch15.6 OK")
print(f"  Original probs: {orig_probs.round(4)}")
print(f"  T=0.5:          {probs_T05.round(4)} -- more peaked [OK]")
print(f"  T=2.0:          {probs_T20.round(4)} -- flatter [OK]")
print(f"  Top-k=3:        {topk_probs.round(4)} -- {k} non-zero [OK]")
print(f"  Top-p=0.9:      {topp_probs.round(4)} -- {n_kept} tokens kept [OK]")
print(f"  T->0 (greedy):  {probs_T001.round(4)} -- mass on argmax [OK]")

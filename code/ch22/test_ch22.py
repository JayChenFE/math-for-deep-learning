"""Validate Chapter 22: Softmax stability, Log-Sum-Exp, Logit Bias."""
import numpy as np

# 22.1-22.3 Naive vs stable softmax
def softmax_stable(x):
    x = np.array(x, dtype=np.float64)
    x = x - np.max(x)
    e = np.exp(x)
    return e / e.sum()

# Naive crashes on large inputs (float32)
x_big_f32 = np.array([1000.0, 1001.0, 1002.0], dtype=np.float32)
assert np.any(np.isnan(np.exp(x_big_f32) / np.exp(x_big_f32).sum()))

# Stable works
result = softmax_stable(x_big_f32)
assert not np.any(np.isnan(result))
assert abs(result.sum() - 1.0) < 1e-10

# Stable matches naive for safe inputs
x_safe = np.array([2.0, 1.0, 0.1])
naive = np.exp(x_safe) / np.exp(x_safe).sum()
assert np.allclose(naive, softmax_stable(x_safe))

# 22.6 Log-Sum-Exp + log_softmax
def log_sum_exp(x):
    x = np.array(x, dtype=np.float64)
    c = np.max(x)
    return c + np.log(np.sum(np.exp(x - c)))

def log_softmax_stable(x):
    return x - log_sum_exp(x)

lsm = log_softmax_stable(x_big_f32)
assert not np.any(np.isnan(lsm))
assert abs(np.exp(lsm).sum() - 1.0) < 1e-10

# 22.5 Logit Bias: -inf forces probability to 0
vocab_size = 12
logits = np.random.randn(vocab_size)
bias = np.full(vocab_size, -np.inf)
bias[0] = 0.0  # only token 0 allowed
probs = softmax_stable(logits + bias)
assert probs[0] > 0.99
assert probs[1] < 1e-10  # effectively 0

print("Ch22 OK -- stable softmax, log-sum-exp, logit bias all verified")

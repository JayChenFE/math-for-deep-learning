"""Validate Chapter 15: Distributions & Sampling."""
import numpy as np
np.random.seed(42)
data = np.random.randn(1000) * 2 + 5
assert abs(data.mean() - 5) < 0.5
assert abs(data.std() - 2) < 0.3
def softmax(x, T=1.0):
    x = np.array(x, dtype=float)
    e = np.exp((x - x.max()) / T)
    return e / e.sum()
logits = [2.0, 1.0, 0.5, 0.1, -1.0]
p = softmax(logits, 1.0)
assert abs(p.sum() - 1.0) < 1e-8 and (p > 0).all()
p_hot = softmax(logits, 0.1)
assert p_hot[0] > 0.9
p_cool = softmax(logits, 5.0)
assert max(p_cool) < 0.4
print("Ch15 ALL PASSED")

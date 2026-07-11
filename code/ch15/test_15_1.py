"""Validate Ch15.1: Discrete random variable — dice simulation."""
import numpy as np

np.random.seed(42)
n = 3000
rolls = np.random.randint(1, 7, size=n)

# Verify all values in range [1,6]
assert rolls.min() >= 1 and rolls.max() <= 6

# Verify probabilities sum to 1
full_counts = np.array([np.sum(rolls == k) for k in range(1, 7)])
probs = full_counts / n
assert abs(probs.sum() - 1.0) < 1e-10

# Each probability should be close to 1/6 (~0.167)
for p in probs:
    assert abs(p - 1/6) < 0.03, f"Probability {p:.4f} too far from 1/6"

print(f"Ch15.1 OK — dice probs: {probs.round(3)}, sum={probs.sum():.1f}")

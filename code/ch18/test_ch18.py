"""Validate Chapter 18: LLN & CLT."""
import numpy as np
np.random.seed(42)
# LLN: frequency converges
coins = np.random.binomial(1, 0.5, 10000)
freq = np.cumsum(coins) / np.arange(1, 10001)
assert abs(freq[-1] - 0.5) < 0.02
# CLT: means of exponential ~ normal
means = [np.random.exponential(scale=2.0, size=30).mean() for _ in range(2000)]
assert 1.8 < np.mean(means) < 2.2
print("Ch18 ALL PASSED")

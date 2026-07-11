"""Validate Ch15.4: Bernoulli distribution."""
import numpy as np

np.random.seed(42)

p = 0.3
n_trials = 10000
samples = (np.random.random(n_trials) < p).astype(int)

# Verify sample mean ≈ p
assert abs(samples.mean() - p) < 0.02, f"mean={samples.mean():.3f} vs p={p}"

# Verify sample variance ≈ p(1-p)
expected_var = p * (1 - p)
assert abs(samples.var() - expected_var) < 0.02, \
    f"var={samples.var():.3f} vs expected={expected_var:.3f}"

# All values should be 0 or 1
assert set(np.unique(samples)).issubset({0, 1})

print(f"Ch15.4 OK — mean={samples.mean():.3f} (p={p}), var={samples.var():.3f} (p(1-p)={expected_var:.3f})")

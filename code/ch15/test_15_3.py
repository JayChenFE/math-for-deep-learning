"""Validate Ch15.3: Uniform distribution & weight initialization."""
import numpy as np

np.random.seed(42)

fan_in, fan_out = 784, 256
limit = np.sqrt(6 / (fan_in + fan_out))
weights = np.random.uniform(-limit, limit, size=(fan_in, fan_out))

# Weights should be within [-limit, limit]
assert weights.min() >= -limit - 1e-10
assert weights.max() <= limit + 1e-10

# Mean should be close to 0
assert abs(weights.mean()) < 0.01, f"mean={weights.mean():.6f} not close to 0"

# Theoretical variance of Uniform[-L, L] = (2L)^2 / 12
theoretical_var = (2 * limit)**2 / 12
actual_var = weights.var()
# Within 10% of theoretical
assert abs(actual_var - theoretical_var) / theoretical_var < 0.1, \
    f"var={actual_var:.6f} vs theoretical={theoretical_var:.6f}"

print(f"Ch15.3 OK — limit={limit:.4f}, mean={weights.mean():.6f}, var={actual_var:.6f} (theory={theoretical_var:.6f})")

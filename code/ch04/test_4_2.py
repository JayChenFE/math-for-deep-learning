"""Validate Ch4.2: Scalar multiplication."""
import numpy as np

v = np.array([3.0, 4.0])
v_len = np.sqrt(np.sum(v**2))
assert abs(v_len - 5.0) < 1e-10

# Scale by 2: length doubles
v2 = 2.0 * v
assert abs(np.sqrt(np.sum(v2**2)) - 10.0) < 1e-10
# Direction unchanged: ratios equal
assert np.allclose(v2 / v, np.array([2.0, 2.0]))

# Scale by 0.5: length halves
v05 = 0.5 * v
assert abs(np.sqrt(np.sum(v05**2)) - 2.5) < 1e-10

# Scale by -1: same length, opposite direction
v_neg = -1.0 * v
assert abs(np.sqrt(np.sum(v_neg**2)) - 5.0) < 1e-10
assert np.array_equal(v_neg, np.array([-3.0, -4.0]))

# Scale by 0: zero vector
v0 = 0.0 * v
assert np.array_equal(v0, np.array([0.0, 0.0]))

print("Ch4.2 OK -- scalar multiplication: length scales by |c|, direction preserved")

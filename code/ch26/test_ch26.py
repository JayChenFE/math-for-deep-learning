"""Validate Chapter 26: Linear regression."""
import numpy as np

np.random.seed(42)
n = 100
X_raw = np.random.uniform(-3, 3, size=n)
y = 3.0 * X_raw - 2.0 + np.random.randn(n) * 1.5
X = np.column_stack([X_raw, np.ones(n)])

# Closed-form
w_closed = np.linalg.inv(X.T @ X) @ X.T @ y
assert abs(w_closed[0] - 3.0) < 0.5
assert abs(w_closed[1] - (-2.0)) < 0.5

# Gradient descent
w_gd = np.array([0.0, 0.0])
for _ in range(500):
    w_gd -= 0.01 * (2/n) * X.T @ (X @ w_gd - y)
assert abs(w_gd[0] - 3.0) < 0.5
assert abs(w_gd[1] - (-2.0)) < 0.5

# Both should agree
assert np.allclose(w_closed, w_gd, atol=0.1)

print("Ch26 OK -- closed-form and GD converge to true params")

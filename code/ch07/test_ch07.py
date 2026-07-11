"""Validate Chapter 7: Inverse matrix & linear systems."""
import numpy as np

# 7.1 Identity
I3 = np.eye(3)
A = np.array([[2.,1.,3.],[1.,4.,0.],[5.,2.,1.]])
x = np.array([3.,1.,2.])
assert np.allclose(I3 @ x, x)
assert np.allclose(A @ I3, A)
assert np.allclose(I3 @ A, A)

# 7.2 Inverse
A2 = np.array([[2.,1.],[1.,3.]])
A2_inv = np.linalg.inv(A2)
assert np.allclose(A2 @ A2_inv, np.eye(2), atol=1e-10)
assert np.allclose(A2_inv @ A2, np.eye(2), atol=1e-10)
x2 = np.array([5.,2.])
assert np.allclose(A2_inv @ (A2 @ x2), x2)  # recover

# Singular matrix
B = np.array([[1.,2.],[2.,4.]])
assert abs(np.linalg.det(B)) < 1e-10
try:
    np.linalg.inv(B)
    assert False, "Should have raised"
except np.linalg.LinAlgError:
    pass

# 7.3 Solve equations
A3 = np.array([[2.,3.],[5.,4.]])
b3 = np.array([8.,13.])
w_inv = np.linalg.inv(A3) @ b3
w_solve = np.linalg.solve(A3, b3)
assert np.allclose(w_inv, w_solve)
assert abs(2*w_inv[0] + 3*w_inv[1] - 8) < 1e-10
assert abs(5*w_inv[0] + 4*w_inv[1] - 13) < 1e-10

# 7.4 Pseudoinverse
X = np.array([[1.,2.],[3.,4.],[5.,6.]])
y = np.array([4.,10.,16.])
w_pinv = np.linalg.pinv(X) @ y
assert w_pinv.shape == (2,)

# 7.5 Normal equation
np.random.seed(42)
n = 100
X_raw = np.random.uniform(-3, 3, size=n)
y_data = 3.0 * X_raw - 2.0 + np.random.randn(n) * 1.5
X_design = np.column_stack([X_raw, np.ones(n)])
w_lr = np.linalg.inv(X_design.T @ X_design) @ X_design.T @ y_data
assert abs(w_lr[0] - 3.0) < 0.5  # slope ~3
assert abs(w_lr[1] - (-2.0)) < 0.5  # intercept ~-2

print("Ch7 OK -- inverse, solve, pinv, normal equation all pass")

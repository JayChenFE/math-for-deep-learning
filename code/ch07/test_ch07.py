"""Validate Chapter 7: Inverse, Linear Systems & Closed-Form Solution."""
import numpy as np

# 7.1 Inverse
A = np.array([[4, 7], [2, 6]])
assert np.allclose(np.linalg.inv(A) @ A, np.eye(2))

# 7.2 Solve linear system
x_sol = np.linalg.solve(np.array([[2, 3], [5, 4]]), np.array([8, 13]))
assert abs(2 * x_sol[0] + 3 * x_sol[1] - 8) < 0.1

# 7.3 Singular matrix + pseudoinverse
try:
    np.linalg.inv(np.array([[1, 2], [2, 4]]))
except np.linalg.LinAlgError:
    pass
A_sing = np.array([[1, 2], [2, 4]])
A_pinv = np.linalg.pinv(A_sing)
assert np.allclose(A_sing @ A_pinv @ A_sing, A_sing)

# 7.4 Closed-form linear regression
np.random.seed(42)
X = np.random.randn(50, 2)
y = X @ [3.0, 2.0] + 1.0 + np.random.randn(50) * 0.3
Xa = np.column_stack([X, np.ones(50)])
w = np.linalg.inv(Xa.T @ Xa) @ Xa.T @ y
assert abs(w[0] - 3) < 1.0

print("Ch7 ALL PASSED")

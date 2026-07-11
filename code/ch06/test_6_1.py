"""Validate Ch6.1: Matrix-vector multiplication."""
import numpy as np

W = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
x = np.array([2.0, 1.0])

y_at = W @ x
y_manual = np.array([np.dot(W[i], x) for i in range(3)])

assert y_at.shape == (3,)
assert np.allclose(y_at, y_manual)
assert np.allclose(y_at, np.array([4.0, 10.0, 16.0]))  # [1*2+2*1, 3*2+4*1, 5*2+6*1]

print("Ch6.1 OK -- matrix-vector: (3x2) @ (2,) -> (3,), each row dot with x")

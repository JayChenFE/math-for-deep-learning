"""Validate Chapter 6: Matrix Multiplication & Broadcasting."""
import numpy as np

# 6.1 Matrix-vector
W = np.array([[1, 2, 0, -1], [0, 1, 3, 2], [2, 0, 1, 0]])
x = np.array([2, 1, 3, 4])
y = W @ x
assert y.shape == (3,)
for i in range(3):
    assert abs(np.dot(W[i], x) - y[i]) < 1e-10

# 6.2 Matrix-matrix
A = np.random.randn(50, 60)
B = np.random.randn(60, 70)
assert (A @ B).shape == (50, 70)

# 6.4 Linear layer
X = np.random.randn(32, 784)
W6 = np.random.randn(256, 784) * 0.01
b6 = np.zeros(256)
out = X @ W6.T + b6
assert out.shape == (32, 256)

# 6.5 Broadcasting
A_br = np.ones((3, 4))
v_br = np.array([10, 20, 30, 40])
assert (A_br + v_br).shape == (3, 4)

print("Ch6 ALL PASSED")

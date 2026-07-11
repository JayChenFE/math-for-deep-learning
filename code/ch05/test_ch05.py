"""Validate Chapter 5: Dot Product & Cosine Similarity."""
import numpy as np

a = np.array([2, 3, 1])
b = np.array([4, 0, 5])
assert np.dot(a, b) == 13
assert a @ b == 13
assert (a * b).sum() == 13

v1 = np.array([1.0, 0.0])
v2 = np.array([0.0, 1.0])
v3 = np.array([-1.0, 0.0])
assert abs(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))) < 1e-10
assert abs(np.dot(v1, v3) / (np.linalg.norm(v1) * np.linalg.norm(v3)) - (-1)) < 1e-10

Q = np.random.randn(4, 3)
K = np.random.randn(6, 3)
assert (Q @ K.T).shape == (4, 6)

print("Ch5 ALL PASSED")

"""Validate Ch3.3: Matrix — 2-D array."""
import numpy as np

batch_size, num_features = 32, 784
X = np.random.randn(batch_size, num_features)
assert X.ndim == 2
assert X.shape == (32, 784)

# Each row is a sample vector
assert X[0].shape == (784,)
assert X[15].shape == (784,)

# Each column is a feature
assert X[:, 0].shape == (32,)

# Indexing
M = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
assert M[0, 0] == 1
assert np.array_equal(M[1, :], np.array([4, 5, 6]))
assert np.array_equal(M[:, 2], np.array([3, 6, 9]))
assert np.array_equal(M[:2, 1:], np.array([[2, 3], [5, 6]]))

# Linear layer params
d_in, d_out = 784, 256
W = np.random.randn(d_in, d_out) * 0.01
b = np.zeros(d_out)
assert W.shape == (784, 256)
assert b.shape == (256,)

print("Ch3.3 OK -- matrix: 2-D, shape=(rows, cols), X[i,:] extracts row i")

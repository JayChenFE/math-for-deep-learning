"""Validate Ch3.2: Vector — 1-D array."""
import numpy as np

user_features = np.array([28, 15000, 0.7, 42])
assert user_features.ndim == 1
assert user_features.shape == (4,)

word_embedding = np.random.randn(768)
assert word_embedding.shape == (768,)

bias = np.zeros(256)
assert bias.shape == (256,)
assert bias.sum() == 0

v = np.array([10, 20, 30, 40, 50])
assert v[0] == 10
assert v[-1] == 50
assert np.array_equal(v[1:4], np.array([20, 30, 40]))
assert np.array_equal(v[:3], np.array([10, 20, 30]))

# L2 norm
norm = np.sqrt(np.sum(user_features.astype(float) ** 2))
assert norm > 0

# Distinguish (n,) vs (n,1) vs (1,n)
a = np.array([1, 2, 3])
assert a.shape == (3,)
assert a.reshape(3, 1).shape == (3, 1)
assert a.reshape(1, 3).shape == (1, 3)

print("Ch3.2 OK -- vector: 1-D, shape=(n,), indexing same as list")

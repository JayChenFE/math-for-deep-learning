"""Validate Ch3.5: Axis — what does axis=k actually do?"""
import numpy as np

np.random.seed(42)
tensor = np.random.randn(2, 3, 4)

# axis=0: eliminates dim 0
s0 = tensor.sum(axis=0)
assert s0.shape == (3, 4)

# axis=1: eliminates dim 1
s1 = tensor.sum(axis=1)
assert s1.shape == (2, 4)

# axis=2: eliminates dim 2
s2 = tensor.sum(axis=2)
assert s2.shape == (2, 3)

# axis=-1 same as axis=2 for 3D
s_neg1 = tensor.sum(axis=-1)
assert np.array_equal(s2, s_neg1)

# Matrix visualization
M = np.array([[1, 2, 3], [4, 5, 6]])
assert np.array_equal(M.sum(axis=0), np.array([5, 7, 9]))
assert M.sum(axis=0).shape == (3,)
assert np.array_equal(M.sum(axis=1), np.array([6, 15]))
assert M.sum(axis=1).shape == (2,)

# mean along axis
m = tensor.mean(axis=-1)
assert m.shape == (2, 3)

# verify values: sum and divide manually
manual_mean = tensor.sum(axis=-1) / tensor.shape[-1]
assert np.allclose(m, manual_mean)

print("Ch3.5 OK -- axis=k eliminates the k-th dimension, axis=-1 always the last")

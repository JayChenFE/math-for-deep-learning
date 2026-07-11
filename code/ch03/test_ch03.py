"""Validate Chapter 3 code blocks"""
import numpy as np
from pathlib import Path
PROJ = Path(__file__).resolve().parent.parent.parent

# 3.1 scalar
loss_np = np.array(3.14159)
assert loss_np.ndim == 0 and loss_np.shape == ()
print("3.1 scalar OK")

# 3.2 vector
user = np.array([25, 175, 8000, 0.85])
assert user.shape == (4,)
x = np.linspace(0, 1, 5)
assert len(x) == 5 and abs(x[-1] - 1.0) < 1e-10
print("3.2 vector OK")

# 3.3 matrix
X = np.random.randn(32, 784)
assert X.shape == (32, 784)
assert X[0].shape == (784,)
assert X[:, 0].shape == (32,)
print("3.3 matrix OK")

# 3.4 tensor
X_t = np.random.randn(4, 10, 512)
assert X_t.shape == (4, 10, 512)
assert X_t[0, 3, :].shape == (512,)
images = np.random.randn(8, 3, 224, 224)
assert images.ndim == 4
print("3.4 tensor OK")

# 3.5 axis
A = np.array([[1, 2, 3], [4, 5, 6]])
assert A.sum(axis=0).shape == (3,)
assert A.sum(axis=1).shape == (2,)
T = np.random.randn(4, 10, 512)
assert T.sum(axis=0).shape == (10, 512)
assert T.sum(axis=1).shape == (4, 512)
assert T.sum(axis=2).shape == (4, 10)
print("3.5 axis OK")

print("\nChapter 3 ALL PASSED")

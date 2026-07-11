"""Validate Chapter 25: Weight initialization."""
import numpy as np

np.random.seed(42)
n_layers, dim, batch = 20, 128, 256

# Kaiming: std ~ 1 after many ReLU layers
x = np.random.randn(batch, dim)
for _ in range(n_layers):
    W = np.random.randn(dim, dim) * np.sqrt(2.0 / dim)
    x = np.maximum(0, x @ W)
assert 0.1 < x.std() < 3.0

# Xavier + ReLU: std should decay (no sqrt(2) compensation)
x2 = np.random.randn(batch, dim)
for _ in range(n_layers):
    limit = np.sqrt(6.0 / (dim + dim))
    W = np.random.uniform(-limit, limit, (dim, dim))
    x2 = np.maximum(0, x2 @ W)
assert x2.std() < x.std()

# Zero init: all outputs identical
X = np.random.randn(3, 4)
Wz = np.zeros((4, 3))
h = X @ Wz
assert np.allclose(h[:, 0], h[:, 1])

print("Ch25 OK -- Kaiming preserves variance, Xavier decays, zero init = dead")

"""Validate Chapter 25: Weight initialization."""
import numpy as np

np.random.seed(42)
n_layers, dim = 20, 128
batch = 256

# Kaiming: std ~ sqrt(2/fan_in)
x = np.random.randn(batch, dim)
for _ in range(n_layers):
    W = np.random.randn(dim, dim) * np.sqrt(2.0 / dim)
    x = np.maximum(0, x @ W)  # ReLU
# Kaiming maintains std ~ 1
assert 0.1 < x.std() < 3.0, f"Kaiming std={x.std():.4f}"

# Xavier (no ReLU compensation): std should decay
x2 = np.random.randn(batch, dim)
for _ in range(n_layers):
    limit = np.sqrt(6.0 / (dim + dim))
    W = np.random.uniform(-limit, limit, (dim, dim))
    x2 = np.maximum(0, x2 @ W)  # ReLU
# Xavier without ReLU compensation -> signal shrinks
assert x2.std() < 0.5, f"Xavier+ReLU std={x2.std():.4f} should be small"

# Zero init: all neurons identical
W_zero = np.zeros((dim, dim))
x3 = np.random.randn(batch, dim)
x3 = np.maximum(0, x3 @ W_zero)  # all outputs are 0
assert np.allclose(x3, 0.0)

# Kaiming > Xavier for ReLU
assert x.std() > x2.std() * 2  # Kaiming preserves more variance

print(f"Ch25 OK -- Kaiming std={x.std():.3f}, Xavier+ReLU std={x2.std():.4f}, zero=dead")

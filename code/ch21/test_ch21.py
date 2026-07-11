"""Validate Chapter 21: LayerNorm & RMSNorm."""
import numpy as np

def layer_norm(x, gamma=None, beta=None, eps=1e-5):
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    x_norm = (x - mean) / np.sqrt(var + eps)
    if gamma is not None: x_norm = x_norm * gamma
    if beta is not None: x_norm = x_norm + beta
    return x_norm

def rms_norm(x, gamma=None, eps=1e-5):
    rms = np.sqrt(np.mean(x**2, axis=-1, keepdims=True))
    x_norm = x / (rms + eps)
    if gamma is not None: x_norm = x_norm * gamma
    return x_norm

np.random.seed(42)
X = np.random.randn(2, 3, 4)
gamma = np.ones(4); beta = np.zeros(4)

# LayerNorm
X_ln = layer_norm(X, gamma, beta)
token_means = X_ln.mean(axis=-1)
token_vars = X_ln.var(axis=-1)
assert np.allclose(token_means, 0.0, atol=1e-6)
# Variances should be ~1.0 after LN (may be 0.9999x due to float)
assert np.all(token_vars > 0.99) and np.all(token_vars < 1.01)

# RMSNorm
X_rms = rms_norm(X, gamma)
# RMSNorm: mean is NOT zero, but RMS ≈ 1
rms_vals = np.sqrt(np.mean(X_rms**2, axis=-1))
assert np.all(rms_vals > 0.99) and np.all(rms_vals < 1.01)
# LayerNorm mean should be closer to 0 than RMSNorm
assert abs(X_ln.mean()) < abs(X_rms.mean())

# RMSNorm is simpler (no mean subtraction)
assert layer_norm(X, gamma, beta).shape == X.shape
assert rms_norm(X, gamma).shape == X.shape

# Compare with PyTorch if available
try:
    import torch
    import torch.nn as nn
    X_t = torch.tensor(X, dtype=torch.float32)
    ln = nn.LayerNorm(4, eps=1e-5, elementwise_affine=False)
    with torch.no_grad():
        result = ln(X_t).numpy()
    assert np.allclose(X_ln, result, atol=1e-5)
    print("Ch21 OK -- LayerNorm matches PyTorch, RMSNorm RMS=1")
except ImportError:
    print("Ch21 OK -- LayerNorm mu=0/std=1, RMSNorm RMS=1 (no PyTorch)")

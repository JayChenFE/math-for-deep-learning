"""Validate Ch6.4: Fully connected layer — output = X @ W.T + b."""
import numpy as np

np.random.seed(42)
batch_size, d_in, d_out = 32, 784, 256
X = np.random.randn(batch_size, d_in)
W = np.random.randn(d_out, d_in) * np.sqrt(2.0 / d_in)
b = np.zeros(d_out)

output = X @ W.T + b
assert output.shape == (batch_size, d_out)

# Verify: output[i] = X[i] @ W.T + b = dot(X[i], W[j]) for each output neuron j
for i in range(3):
    for j in range(5):
        manual = np.dot(X[i], W[j]) + b[j]
        assert abs(output[i, j] - manual) < 1e-10

# Compare with PyTorch if available
try:
    import torch
    import torch.nn as nn
    torch.manual_seed(42)
    layer = nn.Linear(d_in, d_out, bias=True)
    with torch.no_grad():
        layer.weight.copy_(torch.tensor(W, dtype=torch.float32))
        layer.bias.zero_()
    X_torch = torch.tensor(X, dtype=torch.float32)
    output_torch = layer(X_torch)
    assert np.allclose(output, output_torch.detach().numpy(), atol=1e-5)
    print("Ch6.4 OK -- NumPy FC matches PyTorch nn.Linear")
except ImportError:
    print("Ch6.4 OK -- NumPy FC layer: X@W.T+b verified (no PyTorch)")

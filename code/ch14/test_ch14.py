"""Validate Chapter 14: Computational Graph & Autograd."""
import numpy as np

# Test autograd concepts with PyTorch if available
try:
    import torch
    # y = (x+2)*3, dy/dx = 3
    x = torch.tensor(1.0, requires_grad=True)
    y = (x + 2) * 3
    y.backward()
    assert abs(x.grad.item() - 3.0) < 1e-6
    # y = sin(x^2) at x=2
    x2 = torch.tensor(2.0, requires_grad=True)
    y2 = torch.sin(x2 ** 2)
    y2.backward()
    expected = torch.cos(torch.tensor(4.0)) * 4
    assert abs(x2.grad.item() - expected.item()) < 1e-4
    # no_grad test
    with torch.no_grad():
        z = x2 ** 2
        assert not z.requires_grad
    print("Ch14 PyTorch OK")
except ImportError:
    print("Ch14 SKIP (no torch)")

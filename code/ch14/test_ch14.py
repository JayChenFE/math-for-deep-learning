"""Validate Chapter 14: Computational graph & autograd."""
import numpy as np

# 14.1-14.4: Manual backprop vs PyTorch
try:
    import torch
    x = torch.tensor(1.0, requires_grad=True)
    u = x + 2; y = u * 3
    y.backward()
    assert abs(x.grad.item() - 3.0) < 1e-6  # dy/dx = 3

    # sin(x^2) at x=2
    x2 = torch.tensor(2.0, requires_grad=True)
    y2 = torch.sin(x2**2)
    y2.backward()
    expected = torch.cos(torch.tensor(4.0)) * 4
    assert abs(x2.grad.item() - expected.item()) < 1e-4

    # 14.6 detach vs no_grad
    w = torch.tensor([2.0, 3.0], requires_grad=True)
    x3 = torch.tensor([1.0, 2.0])
    y3 = (w * x3).sum()
    yv = y3.detach()
    assert not yv.requires_grad
    assert w.requires_grad  # unchanged

    # no_grad
    x_big = torch.randn(300, 300)
    with torch.no_grad():
        z = x_big @ x_big.T
        assert not z.requires_grad

    # detach creates new tensor, original unchanged
    orig = torch.tensor([5.0], requires_grad=True)
    detached = orig.detach()
    assert orig.requires_grad  # still True
    assert not detached.requires_grad

    print("Ch14 OK -- PyTorch autograd, detach, no_grad")
except ImportError:
    # Validate logic without PyTorch
    # y=(x+2)*3: x=1 -> u=3 -> y=9, dy/dx=du/dx*dy/du=1*3=3
    x = 1.0; u = x+2; y = u*3
    assert y == 9.0  # forward
    dy_du, du_dx = 3, 1
    assert dy_du * du_dx == 3  # backprop

    # y=(x*2+1)^2 at x=3
    x = 3.0; a = x*2; b = a+1; y = b**2
    assert a == 6 and b == 7 and y == 49
    # back: dy/db=2b=14, db/da=1, da/dx=2 -> 14*1*2=28
    assert abs((2*b)*1*2 - 28.0) < 1e-10

    print("Ch14 OK -- manual computational graph verified (no PyTorch)")

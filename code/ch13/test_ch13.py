"""Validate Chapter 13: Chain rule."""
import numpy as np

def nd(f, x, h=1e-5):
    return (f(x+h) - f(x-h)) / (2*h)

# 13.1 Two-layer: h(x)=sin(x^2)
h = lambda x: np.sin(x**2)
u = lambda x: x**2
dh_du = lambda u: np.cos(u)
du_dx = lambda x: 2*x
x0 = 2.0
chain_2 = dh_du(u(x0)) * du_dx(x0)
assert abs(chain_2 - nd(h, x0)) < 1e-5
assert abs(chain_2 - np.cos(4.0)*4.0) < 1e-10

# 13.3 Three-layer: h(x)=ln(cos(x^3))
def chain_3(x):
    u = x**3; v = np.cos(u)
    dh_dv = 1/v; dv_du = -np.sin(u); du_dx = 3*x**2
    return dh_dv * dv_du * du_dx

for x_test in [0.5, 0.8, 1.0]:  # cos(x^3)>0 required for ln
    c3 = chain_3(x_test)
    n3 = nd(lambda x: np.log(np.cos(x**3)), x_test)
    assert abs(c3 - n3) < 1e-5, f"x={x_test}: chain={c3:.6f} num={n3:.6f}"

# Exercise concepts
# h(x)=exp(3x^2+2): u=3x^2+2, dh/du=exp(u), du/dx=6x
x1 = 1.0
u1 = 3*x1**2 + 2  # =5
chain_exp = np.exp(u1) * (6*x1)  # exp(5)*6
assert abs(chain_exp - nd(lambda x: np.exp(3*x**2+2), x1)) < 1e-3

# PyTorch verification if available
try:
    import torch
    x_t = torch.tensor(0.8, requires_grad=True)
    h_t = torch.log(torch.cos(x_t**3))
    h_t.backward()
    assert abs(x_t.grad.item() - chain_3(1.5)) < 1e-5
    print("Ch13 OK -- chain rule + PyTorch autograd verified")
except ImportError:
    print("Ch13 OK -- chain rule verified (no PyTorch)")

# Chain rule intuition: derivative = product of local gradients
# Backprop = chain rule executed from loss backwards
print("  Chain rule = multiply local gradients layer by layer")
print("  Backprop = chain rule on computation graph at scale")

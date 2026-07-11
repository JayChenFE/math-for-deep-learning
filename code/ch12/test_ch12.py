"""Validate Chapter 12: Gradient Vector."""
import numpy as np

def f12(x):
    """Rosenbrock-like narrow valley function."""
    return x[0]**2 + 10 * x[1]**2

def grad(f, x, h=1e-5):
    """Numerical gradient for arbitrary f: R^n -> R."""
    g = np.zeros_like(x)
    for i in range(len(x)):
        xp = x.copy()
        xp[i] += h
        xm = x.copy()
        xm[i] -= h
        g[i] = (f(xp) - f(xm)) / (2 * h)
    return g

# Gradient at (3, 2) should be ~(6, 40)
g12 = grad(f12, np.array([3.0, 2.0]))
assert abs(g12[0] - 6.0) < 0.1
assert abs(g12[1] - 40.0) < 0.5

# Gradient descent on L(w) = (w-3)^2
L = lambda w: (w - 3) ** 2
dL = lambda w: 2 * (w - 3)
w = 0.0
for _ in range(100):
    w -= 0.1 * dL(w)
assert abs(w - 3.0) < 1e-5

print("Ch12 ALL PASSED")

"""Validate Chapter 11: Partial Derivatives."""
import numpy as np

def f11(x, y):
    return x**2 + y**2

def partial_x(f, x, y, h=1e-5):
    return (f(x + h, y) - f(x - h, y)) / (2 * h)

# Verify partial derivatives of f(x,y) = x^2 + y^2
assert abs(partial_x(f11, 3.0, 4.0) - 6.0) < 1e-3

# np.gradient on 2D grid
x = np.linspace(-3, 3, 20)
y = np.linspace(-3, 3, 20)
X, Y = np.meshgrid(x, y)
Z = X**2 + Y**2
dZ_dy, dZ_dx = np.gradient(Z, y, x)
assert dZ_dx.shape == (20, 20)
assert dZ_dy.shape == (20, 20)
# Near origin (center of grid), gradient should be ~0
assert abs(dZ_dx[10, 10]) < 0.5
assert abs(dZ_dy[10, 10]) < 0.5

print("Ch11 ALL PASSED")

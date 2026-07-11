"""Validate Chapter 11: Partial derivatives."""
import numpy as np

def f(x, y): return x**2 + y**2

def partial_x(f, x, y, h=1e-5):
    return (f(x+h, y) - f(x-h, y)) / (2*h)

def partial_y(f, x, y, h=1e-5):
    return (f(x, y+h) - f(x, y-h)) / (2*h)

x0, y0 = 3.0, 4.0
assert abs(partial_x(f, x0, y0) - 6.0) < 1e-5
assert abs(partial_y(f, x0, y0) - 8.0) < 1e-5

# partial_x only depends on x, not y
for y_test in [0.0, 5.0, 10.0]:
    assert abs(partial_x(f, x0, y_test) - 6.0) < 1e-5

# np.gradient: returns (dZ/dy, dZ/dx)
x = np.linspace(-3, 3, 50); y = np.linspace(-3, 3, 50)
X, Y = np.meshgrid(x, y); Z = X**2 + Y**2
dZ_dy, dZ_dx = np.gradient(Z, y, x)
assert dZ_dx.shape == (50, 50) and dZ_dy.shape == (50, 50)

# At origin: gradient ~ 0
assert abs(dZ_dx[25, 25]) < 0.2
assert abs(dZ_dy[25, 25]) < 0.2
# At corner (x~3,y~3): gradient ~ 2x=6, 2y=6
assert 5.5 < dZ_dx[-1, -1] < 6.5
assert 5.5 < dZ_dy[-1, -1] < 6.5

# Gradient perpendicular to contour
# For f=x^2+2y^2, contour is ellipse: gradient = [2x, 4y]
Z2 = X**2 + 2*Y**2
dZ2_dy, dZ2_dx = np.gradient(Z2, y, x)
# At point (2, 1): gradient should be [4, 4]
idx_x = np.argmin(np.abs(x - 2.0))
idx_y = np.argmin(np.abs(y - 1.0))
assert abs(dZ2_dx[idx_y, idx_x] - 4.0) < 0.2
assert abs(dZ2_dy[idx_y, idx_x] - 4.0) < 0.2

print("Ch11 OK -- partial derivatives, np.gradient, contour perpendicularity")

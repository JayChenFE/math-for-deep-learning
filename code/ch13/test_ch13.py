"""Validate Chapter 13: Chain Rule."""
import numpy as np

def num_deriv(f, x, h=1e-5):
    return (f(x + h) - f(x - h)) / (2 * h)

# h(x) = sin(x^2)
h13 = lambda x: np.sin(x ** 2)
x0 = 2.0
chain = np.cos(x0 ** 2) * (2 * x0)
numeric = num_deriv(h13, x0)
assert abs(chain - numeric) < 1e-5

# h(x) = ln(cos(x^3)) at x=1.0
x1 = 1.0
h13b = lambda x: np.log(np.cos(x ** 3))
u13 = x1 ** 3
v13 = np.cos(u13)
chain3 = (1 / v13) * (-np.sin(u13)) * (3 * x1 ** 2)
numeric3 = num_deriv(h13b, x1)
assert abs(chain3 - numeric3) < 1e-4

print("Ch13 ALL PASSED")

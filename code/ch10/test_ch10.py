"""Validate Chapter 10: Derivatives."""
import numpy as np

def num_deriv(f, x, h=1e-5):
    """Central difference numerical derivative."""
    return (f(x + h) - f(x - h)) / (2 * h)

def num_second(f, x, h=1e-4):
    """Central difference second derivative."""
    return (f(x + h) - 2 * f(x) + f(x - h)) / (h ** 2)

# Power function x^3 at x=2
f_cube = lambda x: x ** 3
assert abs(num_deriv(f_cube, 2.0) - 12) < 1e-3

# Exponential e^x at x=1
assert abs(num_deriv(np.exp, 1.0) - np.e) < 1e-3

# Second derivative x^3 at x=2 (f'' = 6x, at 2 = 12)
assert abs(num_second(f_cube, 2.0) - 12) < 0.1

# Sigmoid at x=0 (derivative = 0.25)
sigmoid = lambda x: 1 / (1 + np.exp(-x))
assert abs(num_deriv(sigmoid, 0.0) - 0.25) < 1e-3

# ln(x) at x=2 (derivative = 0.5)
assert abs(num_deriv(np.log, 2.0) - 0.5) < 1e-3

# x^4 at x=2 (derivative = 32)
assert abs(num_deriv(lambda x: x ** 4, 2.0) - 32) < 1e-2

print("Ch10 ALL PASSED")

"""Validate Chapter 10: Derivatives — analytical formulas + numerical verification."""
import numpy as np

def numerical_derivative(f, x, h=1e-5):
    return (f(x + h) - f(x - h)) / (2 * h)

# 10.1 Numerical derivative accuracy
assert abs(numerical_derivative(lambda x: x**2, 3.0) - 6.0) < 1e-8

# 10.2-10.4 Analytical derivatives
# x^3 at x=2: 3*2^2 = 12
assert abs(3 * 2**2 - 12.0) < 1e-10
# e^x at x=1: e^1
assert abs(np.exp(1.0) - np.e) < 1e-10
# sigmoid at x=0: s(0)*(1-s(0)) = 0.5*0.5 = 0.25
s0 = 1/(1+np.exp(0))
assert abs(s0 * (1-s0) - 0.25) < 1e-10
# ReLU deriv: x>0 -> 1, x<0 -> 0
assert np.where(3.0 > 0, 1.0, 0.0) == 1.0
assert np.where(-2.0 > 0, 1.0, 0.0) == 0.0

# Verify all with numerical
for name, f, f_deriv, x, expected in [
    ("x^3", lambda x: x**3, lambda x: 3*x**2, 2.0, 12.0),
    ("e^x", np.exp, np.exp, 1.0, np.e),
    ("sigmoid", lambda x: 1/(1+np.exp(-x)),
     lambda x: (s:=1/(1+np.exp(-x))) and s*(1-s), 0.0, 0.25),
]:
    nd = numerical_derivative(f, x)
    ad = f_deriv(x)
    assert abs(nd - ad) < 1e-5, f"{name}: nd={nd:.6f} ad={ad:.6f}"

# 10.5 Second derivative
def numerical_second_derivative(f, x, h=1e-4):
    return (f(x+h) - 2*f(x) + f(x-h)) / (h**2)
assert abs(numerical_second_derivative(lambda x: x**2, 3.0) - 2.0) < 1e-4
assert abs(numerical_second_derivative(lambda x: x**3, 2.0) - 12.0) < 0.01  # 6*2=12

# 10.6 ReLU at x=0
relu = lambda x: np.maximum(0, x)
# numerical derivative at x=0 gives ~0.5 (center difference straddles discontinuity)
nd_relu_0 = numerical_derivative(relu, 0.0, h=1e-6)
assert 0.4 < nd_relu_0 < 0.6  # roughly 0.5
# at x=0.1, derivative should be ~1
assert abs(numerical_derivative(relu, 0.1) - 1.0) < 1e-5
# at x=-0.1, derivative should be ~0
assert abs(numerical_derivative(relu, -0.1)) < 1e-5

print("Ch10 OK -- numerical & analytical derivatives, 2nd derivative, ReLU at x=0")

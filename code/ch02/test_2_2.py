"""Validate Ch2.2: Numerical differentiation — h convergence and breakdown."""
import numpy as np

def f(x):
    return x ** 2

def secant_slope_h(f, a, h):
    return (f(a + h) - f(a)) / h

a = 2.0
true_derivative = 2 * a  # f'(x)=2x -> f'(2)=4

# Test optimal h range (~1e-5 to 1e-8)
h_good = 1e-5
slope_good = secant_slope_h(f, a, h_good)
assert abs(slope_good - true_derivative) < 1e-4, \
    f"At h={h_good}, slope={slope_good:.6f} should be ~{true_derivative}"

h_good2 = 1e-6
slope_good2 = secant_slope_h(f, a, h_good2)
assert abs(slope_good2 - true_derivative) < 1e-5, \
    f"At h={h_good2}, slope={slope_good2:.6f} too far from {true_derivative}"

# Test breakdown at tiny h
h_tiny = 1e-16
slope_tiny = secant_slope_h(f, a, h_tiny)
# At h=1e-16, slope should be clearly wrong (either 0 or far from 4)
assert abs(slope_tiny - true_derivative) > 1.0, \
    f"At h={h_tiny}, slope should break down but got {slope_tiny:.6f}"

# Test large h: slope should be noticeably different
h_large = 1.0
slope_large = secant_slope_h(f, a, h_large)
assert abs(slope_large - true_derivative) > 0.5, \
    f"At h={h_large}, slope should differ from true value"

print(f"Ch2.2 OK")
print(f"  h=1e-5:  slope={slope_good:.8f}  (near {true_derivative})")
print(f"  h=1e-6:  slope={slope_good2:.8f}  (near {true_derivative})")
print(f"  h=1:     slope={slope_large:.8f}  (rough approximation)")
print(f"  h=1e-16: slope={slope_tiny:.8f}  (broken by rounding error)")

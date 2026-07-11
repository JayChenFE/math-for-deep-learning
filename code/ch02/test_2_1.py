"""Validate Ch2.1: Secant line slope — average rate of change."""
import numpy as np

def f(x):
    return x ** 2

def secant_slope(f, a, b):
    return (f(b) - f(a)) / (b - a)

# Fixed left endpoint a=1
a = 1.0
b_values = [3.0, 2.0, 1.5, 1.1, 1.01]
slopes = [secant_slope(f, a, b) for b in b_values]

# Slopes should monotonically approach 2.0 (the true derivative at x=1)
assert slopes[0] > slopes[1] > slopes[2] > slopes[3] > slopes[4], \
    f"Slopes should monotonically decrease, got {slopes}"

# The last slope should be very close to 2.0
assert abs(slopes[-1] - 2.0) < 0.02, f"Last slope {slopes[-1]:.4f} too far from 2.0"

# Verify secant slope formula manually
assert abs(secant_slope(f, 1, 3) - 4.0) < 1e-10  # (9-1)/(3-1)=4

print(f"Ch2.1 OK -- slopes approach 2.0: {[round(s, 4) for s in slopes]}")

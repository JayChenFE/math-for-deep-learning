"""Validate Ch1.1: Integer, float, bool — type checking and float precision."""
import math

# Type checks
a, b, c = 42, 3.14159, True
assert isinstance(a, int)
assert isinstance(b, float)
assert isinstance(c, bool)
assert True + True == 2  # bool is int subclass

# Float precision trap
x = 0.1 + 0.2
assert x != 0.3  # This is the whole point!
assert abs(x - 0.3) < 1e-10  # Tolerance comparison
assert math.isclose(0.1 + 0.2, 0.3)

# Float limits
assert 1.0 + 1e-16 == 1.0  # Below float64 precision
assert 1.0 + 1e-15 != 1.0  # Just above precision threshold

# Python int has no upper bound
big = 2 ** 1000
assert len(str(big)) == 302

print(f"Ch1.1 OK -- 0.1+0.2={x}, abs error={x-0.3:.2e}, int has no overflow")

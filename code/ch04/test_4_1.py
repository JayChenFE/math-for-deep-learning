"""Validate Ch4.1: Vector addition."""
import numpy as np

a = np.array([3.0, 0.0])
b = np.array([0.0, 4.0])
c = a + b
assert np.array_equal(c, np.array([3.0, 4.0]))
assert abs(np.sqrt(c[0]**2 + c[1]**2) - 5.0) < 1e-10

# Commutativity
assert np.array_equal(a + b, b + a)
# Associativity
d = np.array([1.0, 1.0])
assert np.allclose((a + b) + d, a + (b + d))
# Zero vector
zero = np.zeros(2)
assert np.array_equal(a + zero, a)

print("Ch4.1 OK -- vector addition: commutativity, associativity, zero vector")

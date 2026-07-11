"""Validate Ch5.1: Dot product — algebraic definition."""
import numpy as np

a = np.array([2.0, 3.0, 1.0])
b = np.array([4.0, 1.0, 5.0])

dot_manual = sum(a[i] * b[i] for i in range(len(a)))
dot_np = np.dot(a, b)
dot_at = a @ b

# 2*4 + 3*1 + 1*5 = 8 + 3 + 5 = 16
assert dot_manual == 16.0
assert dot_np == 16.0
assert dot_at == 16.0

# Sign intuition
a_pos = np.array([3.0, 4.0])
assert a_pos @ np.array([2.0, 1.0]) == 10.0  # positive
assert a_pos @ np.array([-4.0, 1.0]) == -8.0  # negative

print("Ch5.1 OK -- dot product: sum(a_i*b_i), positive/negative reveals angle")

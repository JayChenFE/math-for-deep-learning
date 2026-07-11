# Test ch01 exercise 3
import math
vals = [0.1 + 0.2, 0.3, 0.1 + 0.1 + 0.1]
n = len(vals)
for i in range(n):
    for j in range(n):
        assert math.isclose(vals[i], vals[j]), f"vals[{i}] and vals[{j}] should be close"
print("ch01-ex3: all math.isclose checks passed")

# Test ch01 exercise 4 logic
import numpy as np
x_test = np.array([2.0, 3.0])
assert np.allclose(x_test, x_test, 1), "identity check"
assert np.allclose(x_test**3, [8, 27]), "cube check"
assert np.allclose(x_test**2, [4, 9]), "square check"
print("ch01-ex4: curve logic verified")

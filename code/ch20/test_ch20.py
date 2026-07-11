"""Validate Chapter 20: Floating Point."""
import numpy as np
r32 = np.float32(1.0) + np.float32(1e-8)
assert abs(float(r32) - 1.0) < 1e-6  # float32 can't represent 1e-8 difference
r64 = 1.0 + 1e-8
assert abs(r64 - 1.0) > 1e-9  # float64 CAN represent it
w=np.float32(1.0); g=np.float32(1e-8); lr=np.float32(0.01)
assert w == w - lr*g  # no change due to precision
print("Ch20 ALL PASSED")

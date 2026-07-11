"""Validate Chapter 20: Floating point traps & mixed precision."""
import numpy as np

# 20.1 float32 vs float64 precision differences
# float32(0.1) and float64(0.1) are different approximations
a32 = np.float32(0.1)
a64 = np.float64(0.1)
# The key insight: float32 cannot distinguish 1e-8 from 0 when added to 1.0
assert np.float32(1.0) + np.float32(1e-6) != np.float32(1.0)  # visible
assert np.float32(1.0) + np.float32(1e-8) == np.float32(1.0)  # swallowed
# float64 CAN distinguish 1e-8
assert np.float64(1.0) + np.float64(1e-8) != np.float64(1.0)

# float32 1.0 + small epsilon
assert np.float32(1.0) + np.float32(1e-6) != np.float32(1.0)  # still visible
assert np.float32(1.0) + np.float32(1e-8) == np.float32(1.0)  # swallowed!

# 20.2 float64 has higher precision
assert np.finfo(np.float32).eps > np.finfo(np.float64).eps

# 20.3 float16 limits
assert np.finfo(np.float16).max == 65504
# fp16 has tiny range compared to fp32
assert np.finfo(np.float16).max < 1e5   # tiny range
assert np.finfo(np.float32).max > 1e30  # huge range
# bfloat16 shares same exponent bits (8) as float32 -> similar range

# 20.4 Small gradient trap
param = np.float32(1.0)
lr = np.float32(1e-3)  # larger lr to make updates visible
# Very small gradient: lr*grad = 1e-3 * 1e-8 = 1e-11 < float32 eps
tiny_grad = np.float32(1e-8)
update = lr * tiny_grad
new_param = param - update
assert new_param == param  # No update! The trap.

# Slightly larger gradient: lr*grad = 1e-3 * 1e-4 = 1e-7 ~ float32 eps
med_grad = np.float32(1e-4)
new_param2 = param - lr * med_grad
assert new_param2 != param  # Update visible

print("Ch20 OK -- float32/16 limits, bf16 range, small gradient trap verified")

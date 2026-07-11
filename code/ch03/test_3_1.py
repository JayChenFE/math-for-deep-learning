"""Validate Ch3.1: Scalar — 0-D tensor."""
import numpy as np

loss_py = 2.37
assert isinstance(loss_py, float)

loss_np = np.array(2.37)
assert loss_np.ndim == 0
assert loss_np.shape == ()
assert abs(loss_np.item() - 2.37) < 1e-10

# Scalar arithmetic
lr = 0.01
gradient = np.array(0.5)
new_loss = loss_np - lr * gradient
assert new_loss.ndim == 0
assert abs(new_loss.item() - (2.37 - 0.01 * 0.5)) < 1e-10

print("Ch3.1 OK -- scalar: 0-D, shape=(), item() extracts Python float")

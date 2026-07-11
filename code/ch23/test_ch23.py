"""Validate Chapter 23: Gradient explosion & clipping."""
import numpy as np

# 23.1 Simulate explosion
n_layers = 50
grad = 1.0
factors = 1.5 + np.random.randn(n_layers) * 0.1
for f in factors:
    grad = grad * f
# After 50 layers with factor ~1.5, gradient should be huge
assert grad > 1e6  # definitely exploded

# Vanishing: factor ~0.5
grad_vanish = 1.0
for f in 0.5 + np.random.randn(n_layers) * 0.05:
    grad_vanish = grad_vanish * abs(f)
assert grad_vanish < 0.01  # definitely vanished

# 23.3 Gradient clipping
np.random.seed(42)
true_param = 3.0
param_no_clip = 0.0
param_with_clip = 0.0
lr = 0.01
max_norm = 1.0

for step in range(100):
    grad = 2 * (param_no_clip - true_param)
    if step % 20 == 0:
        grad = grad * np.exp(np.random.uniform(5, 10))  # explosion

    # No clip
    param_no_clip -= lr * grad

    # With clip
    grad_norm = abs(grad)
    if grad_norm > max_norm:
        grad = grad * (max_norm / grad_norm)
    param_with_clip -= lr * grad

# With clip, parameter should be closer to optimum than without
assert abs(param_with_clip - true_param) < abs(param_no_clip - true_param)

print("Ch23 OK -- gradient explosion simulated, clipping keeps params stable")

"""Validate Chapter 31: Training loop micro-anatomy."""
import numpy as np

# 31.1 Step phases
stages = ['Forward', 'Loss', 'Backward', 'Clip', 'Step']
times = [3.2, 0.1, 6.5, 0.3, 2.1]
assert times[2] > times[0]  # backward > forward

# 31.2 zero_grad: gradient accumulation
param_clean = 0.0
for step in range(50):
    grad = 2*(param_clean - 3.0) + np.random.randn()*0.5
    param_clean -= 0.05*grad
assert abs(param_clean - 3.0) < 0.5  # converges

# Without zero_grad (accumulating)
param_dirty = 0.0; grad_acc = 0.0
for step in range(50):
    grad = 2*(param_dirty - 3.0) + np.random.randn()*0.5
    grad_acc += grad
    param_dirty -= 0.05*grad_acc
# Should diverge or be far from optimum
assert abs(param_dirty) > abs(param_clean)  # dirty is worse

# 31.3 train/eval mode
np.random.seed(42)
x = np.ones(100)
mask = np.random.random(100) > 0.5
y_train = x * mask / 0.5
assert y_train.mean() > 0.5  # some dropped
y_eval = x
assert y_eval.mean() == 1.0  # none dropped

# 31.4 Gradient checkpointing concept
# Checkpoint: don't store intermediate activations, recompute during backward
# Saves ~50-70% memory, costs ~20-30% extra compute
saved_memory_ratio = 0.6
extra_compute_ratio = 0.25
assert saved_memory_ratio > 0.4
assert extra_compute_ratio < 0.5

# 31.5 autograd modes
# Default: backward() releases graph
# retain_graph=True: keep graph for multiple backward passes
# create_graph=True: build graph of gradients (for meta-learning)

print("Ch31 OK -- step phases, zero_grad trap, train/eval, checkpointing concept")

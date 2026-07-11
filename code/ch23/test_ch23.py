"""Validate Chapter 23: Gradient Explosion & Clipping."""
import numpy as np
g=np.array([1.])
for _ in range(20): g=g*2.
assert g[0] > 1000  # exploded
np.random.seed(42); grad=np.random.randn(100)*100
norm=np.linalg.norm(grad)
if norm>1.: grad=grad*1./norm
assert np.linalg.norm(grad) <= 1.01
print("Ch23 ALL PASSED")

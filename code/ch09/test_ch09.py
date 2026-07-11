"""Validate Chapter 9: Singular Value Decomposition & LoRA."""
import numpy as np

# 9.1 SVD decomposition & reconstruction
np.random.seed(42)
A = np.random.randn(5, 4)
U, S, Vt = np.linalg.svd(A, full_matrices=False)
Sigma = np.diag(S)
assert np.allclose(U @ Sigma @ Vt, A)

# 9.2 Energy concentration
energy = np.cumsum(S ** 2) / np.sum(S ** 2)
assert energy[-1] > 0.999

# 9.3 Low-rank image compression
h, w = 30, 40
img = np.random.randn(h, w)
Ui, Si, Vti = np.linalg.svd(img, full_matrices=False)
k = 5
recon = Ui[:, :k] @ np.diag(Si[:k]) @ Vti[:k, :]
assert recon.shape == (h, w)

# 9.5 LoRA concept
d, r = 100, 4
W = np.random.randn(d, d) * 0.01
B = np.random.randn(d, r) * 0.01
A2 = np.random.randn(r, d) * 0.01
x = np.random.randn(d)
assert ((W + B @ A2) @ x).shape == (d,)

print("Ch9 ALL PASSED")

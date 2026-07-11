"""Validate Chapter 8: Eigenvalues & Eigenvectors."""
import numpy as np

# 8.1 Eigen decomposition
A = np.array([[4, 1], [2, 3]])
eigvals, eigvecs = np.linalg.eig(A)
for i in range(2):
    v = eigvecs[:, i]
    lam = eigvals[i]
    assert np.allclose(A @ v, lam * v)

# 8.3 Eigenvalue decay (symmetric matrix)
np.random.seed(42)
B = np.random.randn(5, 5)
A_sym = B.T @ B
ev = np.sort(np.linalg.eigvalsh(A_sym))[::-1]
assert len(ev) == 5
assert ev[0] >= ev[-1]

print("Ch8 ALL PASSED")

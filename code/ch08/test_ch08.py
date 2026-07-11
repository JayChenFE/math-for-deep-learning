"""Validate Chapter 8: Eigenvalues & eigenvectors."""
import numpy as np

# 8.1 A@v = lambda*v
A = np.array([[4.,1.],[1.,3.]])
eigvals, eigvecs = np.linalg.eig(A)
for i in range(2):
    v = eigvecs[:, i]
    lam = eigvals[i]
    assert np.allclose(A @ v, lam * v, atol=1e-10)

# 8.2 Eigenvalue sorting & energy
np.random.seed(42)
M = np.random.randn(5, 5)
A5 = M.T @ M  # symmetric positive definite
eigvals5, eigvecs5 = np.linalg.eig(A5)
eigvals5 = eigvals5.real
# All eigenvalues should be positive for PSD matrix
assert np.all(eigvals5 > 0)
# Sort descending
idx = np.argsort(eigvals5)[::-1]
eigvals_sorted = eigvals5[idx]
assert np.all(np.diff(eigvals_sorted) <= 0)
# First 2 should contain significant energy
total = eigvals_sorted.sum()
assert eigvals_sorted[:2].sum() / total > 0.5  # first 2 > 50%

# 8.3 Geometry: unit circle -> ellipse
theta = np.linspace(0, 2*np.pi, 100)
circle = np.column_stack([np.cos(theta), np.sin(theta)])
ellipse = (A @ circle.T).T
# Verify ellipse is different from circle (A is not identity)
assert not np.allclose(circle, ellipse)

# Determinant = product of eigenvalues
det_A = np.linalg.det(A)
prod_eig = np.prod(eigvals)
assert abs(det_A - prod_eig) < 1e-10

print("Ch8 OK -- eigenvectors, energy sorting, circle->ellipse, det=prod(eig)")

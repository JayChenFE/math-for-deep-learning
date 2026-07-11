"""Validate Chapter 17: Expectation, Variance, Covariance, PCA."""
import numpy as np
np.random.seed(42)
X = np.random.randn(100, 4)
X[:,1] = X[:,0]*0.8 + np.random.randn(100)*0.3
X[:,2] = -X[:,0]*0.6 + np.random.randn(100)*0.3
cov_mat = np.cov(X.T)
assert cov_mat[0,1] > 0.3   # positive correlation
assert cov_mat[0,2] < -0.2  # negative correlation
Xc = X - X.mean(axis=0); cov = Xc.T @ Xc / (len(X)-1)
evals, evecs = np.linalg.eigh(cov)
top2 = evecs[:, -2:]; X_pca = Xc @ top2
assert X_pca.shape == (100, 2)
print("Ch17 ALL PASSED")

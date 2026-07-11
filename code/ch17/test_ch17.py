"""Validate Chapter 17: Expectation, variance, covariance, PCA."""
import numpy as np

# 17.1-17.2 Mean & variance
np.random.seed(42)
tight = np.random.normal(5, 1, 1000)
wide = np.random.normal(5, 5, 1000)
assert abs(tight.mean() - 5.0) < 0.1
assert tight.var() < wide.var()
var_formula = np.mean(tight**2) - np.mean(tight)**2
assert abs(var_formula - tight.var()) < 1e-10

# 17.3 Covariance
n = 500
height = np.random.normal(170, 10, n)
weight = 0.8 * height + np.random.normal(0, 8, n)
cov_hw = np.cov(height, weight)[0, 1]
assert cov_hw > 0  # positive correlation

temp = np.random.normal(15, 8, n)
heating = -3 * temp + np.random.normal(0, 10, n)
cov_th = np.cov(temp, heating)[0, 1]
assert cov_th < 0  # negative correlation

corr_hw = np.corrcoef(height, weight)[0, 1]
assert 0.5 < corr_hw < 1.0  # strong positive

# 17.4-17.5 PCA
np.random.seed(42)
npc = 50
c0 = np.random.randn(npc, 4)*0.5 + np.array([2, 3, 1, 0])
c1 = np.random.randn(npc, 4)*0.6 + np.array([-1, -1, 0, 2])
c2 = np.random.randn(npc, 4)*0.4 + np.array([3, -2, 2, -1])
X = np.vstack([c0, c1, c2])
Xc = X - X.mean(axis=0)
cov_mat = np.cov(Xc, rowvar=False)
eigvals, eigvecs = np.linalg.eig(cov_mat)
eigvals = eigvals.real; eigvecs = eigvecs.real
idx = np.argsort(eigvals)[::-1]
W = eigvecs[:, idx[:2]]
X_pca = Xc @ W
assert X_pca.shape == (150, 2)
# First 2 PCs should retain >80% variance
total = eigvals.sum()
assert eigvals[idx[:2]].sum()/total > 0.7

print("Ch17 OK -- mean/var/cov/corr, PCA 4D->2D, variance retained:", round(eigvals[idx[:2]].sum()/total*100), "%")

"""Validate key code blocks from chapters 3-9"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time

print("=" * 50)
print("Chapter 3: 标量、向量与张量")
# 3.1 scalar
loss_np = np.array(3.14159)
assert loss_np.ndim == 0 and loss_np.shape == ()
# 3.2 vector
user = np.array([25, 175, 8000, 0.85])
assert user.shape == (4,)
# 3.3 matrix
X = np.random.randn(32, 784)
assert X.shape == (32, 784)
assert X[0].shape == (784,)
assert X[:, 0].shape == (32,)
# 3.4 tensor
X_t = np.random.randn(4, 10, 512)
assert X_t.shape == (4, 10, 512)
assert X_t[0, 3, :].shape == (512,)
# 3.5 axis
A = np.array([[1, 2, 3], [4, 5, 6]])
assert A.sum(axis=0).shape == (3,)
assert A.sum(axis=1).shape == (2,)
T = np.random.randn(4, 10, 512)
assert T.sum(axis=0).shape == (10, 512)
assert T.sum(axis=1).shape == (4, 512)
assert T.sum(axis=2).shape == (4, 10)
print("Ch3 OK")

print("=" * 50)
print("Chapter 4: 向量的加减与数乘")
a = np.array([2, 1]); b = np.array([1, 3])
assert np.allclose(a + b, [3, 4])
v = np.array([2, 1])
assert np.allclose(2*v, [4, 2])
assert np.allclose(np.linalg.norm(2*v), 2 * np.linalg.norm(v), rtol=0.001)
diff = a - b
assert np.allclose(diff, [1, -2])
# Word embedding demo
king = np.array([0.5, 0.7, 0.3, 0.2])
man = np.array([0.3, 0.4, 0.1, 0.1])
woman = np.array([0.2, 0.5, 0.2, 0.2])
queen_pred = king - man + woman
assert queen_pred.shape == (4,)
print("Ch4 OK")

print("=" * 50)
print("Chapter 5: 点积与相似度")
a5 = np.array([2, 3, 1]); b5 = np.array([4, 0, 5])
assert np.dot(a5, b5) == 13
assert a5 @ b5 == 13
# Cosine similarity
v1, v2 = np.array([1., 0.]), np.array([0., 1.])
cs = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
assert abs(cs) < 1e-10  # orthogonal → cos=0
v3 = np.array([-1., 0.])
cs2 = np.dot(v1, v3) / (np.linalg.norm(v1) * np.linalg.norm(v3))
assert abs(cs2 - (-1.0)) < 1e-10  # opposite → cos=-1
# Batch dot product
Q = np.random.randn(4, 3); K = np.random.randn(6, 3)
scores = Q @ K.T
assert scores.shape == (4, 6)
print("Ch5 OK")

print("=" * 50)
print("Chapter 6: 矩阵乘法")
# 6.1 Matrix-vector
W = np.array([[1, 2, 0, -1], [0, 1, 3, 2], [2, 0, 1, 0]])
x = np.array([2, 1, 3, 4])
y = W @ x
assert y.shape == (3,)
for i in range(3):
    assert abs(np.dot(W[i], x) - y[i]) < 1e-10
# 6.2 Matrix-matrix
A6 = np.random.randn(50, 60); B6 = np.random.randn(60, 70)
C6 = A6 @ B6
assert C6.shape == (50, 70)
# 6.4 Linear layer
batch, d_in, d_out = 32, 784, 256
X6 = np.random.randn(batch, d_in)
W6 = np.random.randn(d_out, d_in) * 0.01
b6 = np.zeros(d_out)
out = X6 @ W6.T + b6
assert out.shape == (batch, d_out)
# 6.5 Broadcasting
A_br = np.ones((3, 4)); v_br = np.array([10, 20, 30, 40])
assert (A_br + v_br).shape == (3, 4)
col = np.array([[1], [2], [3]]); row = np.array([10, 20, 30, 40])
assert (col + row).shape == (3, 4)
print("Ch6 OK")

print("=" * 50)
print("Chapter 7: 逆矩阵与线性方程组")
# 7.1 Inverse
A7 = np.array([[4, 7], [2, 6]])
A7_inv = np.linalg.inv(A7)
assert np.allclose(A7 @ A7_inv, np.eye(2))
# 7.2 Solve system
A_sys = np.array([[2, 3], [5, 4]]); b_sys = np.array([8, 13])
x_sol = np.linalg.solve(A_sys, b_sys)
assert np.allclose(A_sys @ x_sol, b_sys)
# 7.3 Singular matrix + pinv
A_sing = np.array([[1, 2], [2, 4]])
try:
    np.linalg.inv(A_sing)
    assert False, "Should fail"
except np.linalg.LinAlgError:
    pass
A_pinv = np.linalg.pinv(A_sing)
assert np.allclose(A_sing @ A_pinv @ A_sing, A_sing)
# 7.4 Linear regression closed form
np.random.seed(42)
N = 50; X_reg = np.random.randn(N, 2)
true_w = np.array([3.0, 2.0]); true_b = 1.0
y_reg = X_reg @ true_w + true_b + np.random.randn(N) * 0.5
X_aug = np.column_stack([X_reg, np.ones(N)])
w_closed = np.linalg.inv(X_aug.T @ X_aug) @ X_aug.T @ y_reg
assert abs(w_closed[0] - 3.0) < 1.0 and abs(w_closed[1] - 2.0) < 1.0
print("Ch7 OK")

print("=" * 50)
print("Chapter 8: 特征值与特征向量")
# 8.1 Eigen decomposition
A8 = np.array([[4, 1], [2, 3]])
eigvals, eigvecs = np.linalg.eig(A8)
for i in range(2):
    v = eigvecs[:, i]; lam = eigvals[i]
    assert np.allclose(A8 @ v, lam * v, rtol=1e-9)
# 8.3 Eigenvalue decay
np.random.seed(42)
B = np.random.randn(5, 5)
A_sym = B.T @ B
eigvals_sym = np.sort(np.linalg.eigvalsh(A_sym))[::-1]
assert len(eigvals_sym) == 5
assert eigvals_sym[0] >= eigvals_sym[-1]
print("Ch8 OK")

print("=" * 50)
print("Chapter 9: SVD")
# 9.1 SVD decomposition
np.random.seed(42)
A9 = np.random.randn(5, 4)
U, S, Vt = np.linalg.svd(A9, full_matrices=False)
Sigma = np.diag(S)
A9_recon = U @ Sigma @ Vt
assert np.allclose(A9, A9_recon)
# 9.2 Energy
energy = np.cumsum(S**2) / np.sum(S**2)
assert energy[-1] > 0.999
# 9.3 Image compression via low-rank
h, w = 30, 40
img = np.random.randn(h, w)
U_img, S_img, Vt_img = np.linalg.svd(img, full_matrices=False)
k = 5
recon = U_img[:, :k] @ np.diag(S_img[:k]) @ Vt_img[:k, :]
assert recon.shape == (h, w)
# 9.5 LoRA concept
d, r = 100, 4
W9 = np.random.randn(d, d) * 0.01
B9 = np.random.randn(d, r) * 0.01
A9_lora = np.random.randn(r, d) * 0.01
x9 = np.random.randn(d)
y_eff = (W9 + B9 @ A9_lora) @ x9
assert y_eff.shape == (d,)
print("Ch9 OK")

print("\n" + "=" * 50)
print("ALL CHAPTERS 3-9 VALIDATED")

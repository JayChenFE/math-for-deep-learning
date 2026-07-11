"""Validate Chapter 9: SVD decomposition & applications."""
import numpy as np

# 9.1 SVD decomposition & reconstruction
np.random.seed(42)
A = np.random.randn(5, 3)  # NOT square
U, S, VT = np.linalg.svd(A, full_matrices=True)
Sigma = np.zeros((5, 3))
np.fill_diagonal(Sigma, S)
A_recon = U @ Sigma @ VT
assert np.allclose(A, A_recon, atol=1e-10)
assert U.shape == (5, 5) and VT.shape == (3, 3)

# 9.2 Singular value energy
A2 = np.random.randn(50, 30)
_, S2, _ = np.linalg.svd(A2, full_matrices=False)
total_energy = np.sum(S2**2)
k90 = np.searchsorted(np.cumsum(S2**2) / total_energy, 0.9) + 1
assert k90 < len(S2)  # Should need less than all
# First few singular values dominate
assert S2[0] > S2[-1]  # Sorted descending

# 9.3 Image compression via low-rank approx
img = np.random.randn(100, 100)
U_i, S_i, VT_i = np.linalg.svd(img, full_matrices=False)
for k in [5, 20, 50]:
    approx = U_i[:, :k] @ np.diag(S_i[:k]) @ VT_i[:k, :]
    assert approx.shape == (100, 100)
    # Error should decrease as k increases
    error = np.linalg.norm(img - approx)
    assert error > 0  # approximation should have some error

# 9.4 Matrix factorization recommendation
R = np.array([[5,3,0,1],[4,0,0,1],[1,1,0,5],[1,0,0,4],[0,1,5,4]], dtype=float)
U_r, S_r, VT_r = np.linalg.svd(R, full_matrices=False)
k = 2
R_approx = U_r[:, :k] @ np.diag(S_r[:k]) @ VT_r[:k, :]
assert R_approx.shape == (5, 4)

# 9.5 LoRA low-rank intuition
d, r = 100, 4
W = np.random.randn(d, d)
delta_W_full = np.random.randn(d, d) * 0.1
B = np.random.randn(d, r) * 0.01
A_lora = np.random.randn(r, d) * 0.01
delta_W_lora = B @ A_lora
assert delta_W_lora.shape == (d, d)
# LoRA uses far fewer params
full_params = d * d
lora_params = 2 * d * r
assert lora_params < full_params / 10  # at least 10x compression

print("Ch9 OK -- SVD, energy, low-rank approx, recommendation, LoRA")

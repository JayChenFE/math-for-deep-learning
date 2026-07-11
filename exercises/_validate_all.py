"""Validate all exercise answer code for chapters 4-9."""
import numpy as np

# ch04
np.random.seed(0)
words = {w: np.random.randn(10) for w in ['cat', 'dog', 'puppy', 'kitten', 'pet']}
src = words['dog']
tgt = words['cat']
diff = words['dog'] - words['pet'] + words['kitten']
cs_before = np.dot(src, tgt) / (np.linalg.norm(src) * np.linalg.norm(tgt))
cs_after = np.dot(diff, tgt) / (np.linalg.norm(diff) * np.linalg.norm(tgt))
print(f'ch04 OK: cos_before={cs_before:.4f} cos_after={cs_after:.4f}')

# ch05
Q = np.random.randn(10, 8)
K = np.random.randn(20, 8)
assert (Q @ K.T).shape == (10, 20)
print('ch05 OK')

# ch06
A = np.random.randn(10, 15)
B = np.random.randn(15, 20)
C = np.zeros((10, 20))
for i in range(10):
    for j in range(20):
        for t in range(15):
            C[i, j] += A[i, t] * B[t, j]
assert np.allclose(C, A @ B, atol=1e-8)
print('ch06 OK')

# ch07
A7 = np.array([[3, 1], [1, 2]])
b7 = np.array([9, 8])
assert np.allclose(np.linalg.inv(A7) @ b7, np.linalg.solve(A7, b7))
print('ch07 OK')

# ch08
np.random.seed(0)
B8 = np.random.randn(3, 3)
A8 = B8.T @ B8
evals, evecs = np.linalg.eig(A8)
for i in range(3):
    assert np.allclose(A8 @ evecs[:, i], evals[i] * evecs[:, i], rtol=1e-9)
print('ch08 OK')

# ch09
np.random.seed(0)
A9 = np.random.randn(50, 40)
U, S, Vt = np.linalg.svd(A9, full_matrices=False)
e5 = np.linalg.norm(A9 - U[:, :5] @ np.diag(S[:5]) @ Vt[:5, :], 'fro')
e30 = np.linalg.norm(A9 - U[:, :30] @ np.diag(S[:30]) @ Vt[:30, :], 'fro')
assert e30 < e5
print('ch09 OK')

print('ALL ANSWERS VALIDATED')

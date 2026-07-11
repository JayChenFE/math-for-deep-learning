"""Validate Ch6.2: Matrix-matrix multiplication."""
import numpy as np

m, k, n = 50, 40, 30
np.random.seed(42)
A = np.random.randn(m, k)
B = np.random.randn(k, n)

# Triple loop
C_loop = np.zeros((m, n))
for i in range(m):
    for j in range(n):
        C_loop[i, j] = sum(A[i, kk] * B[kk, j] for kk in range(k))

C_at = A @ B

assert C_at.shape == (m, n)
assert np.allclose(C_loop, C_at, atol=1e-10)

# Dimension rule: (m,k) @ (k,n) -> (m,n), k must match
assert C_at.shape == (50, 30)

print(f"Ch6.2 OK -- matmul: ({m},{k}) @ ({k},{n}) -> ({m},{n}), loops match @ operator")

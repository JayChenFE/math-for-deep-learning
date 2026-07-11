"""Validate Chapter 8"""
import numpy as np
A=np.array([[4,1],[2,3]]); eigvals,eigvecs=np.linalg.eig(A)
for i in range(2): assert np.allclose(A@eigvecs[:,i],eigvals[i]*eigvecs[:,i])
np.random.seed(42); B=np.random.randn(5,5); As=B.T@B
ev=np.sort(np.linalg.eigvalsh(As))[::-1]; assert len(ev)==5 and ev[0]>=ev[-1]
print("Ch8 ALL PASSED")

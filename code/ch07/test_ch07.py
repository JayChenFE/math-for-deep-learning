"""Validate Chapter 7"""
import numpy as np
A=np.array([[4,7],[2,6]]); assert np.allclose(np.linalg.inv(A)@A, np.eye(2))
x_sol=np.linalg.solve(np.array([[2,3],[5,4]]),np.array([8,13])); assert np.allclose([2,3]@x_sol,8,atol=0.1)
try: np.linalg.inv(np.array([[1,2],[2,4]]))
except np.linalg.LinAlgError: pass
A_pinv=np.linalg.pinv(np.array([[1,2],[2,4]])); assert np.allclose(np.array([[1,2],[2,4]])@A_pinv@np.array([[1,2],[2,4]]),np.array([[1,2],[2,4]]))
np.random.seed(42); X=np.random.randn(50,2); y=X@[3.,2.]+1+np.random.randn(50)*.3
Xa=np.column_stack([X,np.ones(50)]); w=np.linalg.inv(Xa.T@Xa)@Xa.T@y
assert abs(w[0]-3)<1.0; print("Ch7 ALL PASSED")

"""Validate Chapter 26: Linear Regression."""
import numpy as np
np.random.seed(42); N=100; X=np.random.randn(N,1)
y=X.ravel()*2.0+1.0+np.random.randn(N)*0.5
Xa=np.column_stack([X,np.ones(N)])
w=np.linalg.inv(Xa.T@Xa)@Xa.T@y
assert abs(w[0]-2.0)<0.3 and abs(w[1]-1.0)<0.3
w_gd=np.array([0.,0.])
for _ in range(1000): yp=Xa@w_gd; w_gd-=0.01*(-2/N*Xa.T@(y-yp))
assert abs(w_gd[0]-2.0)<0.3
print("Ch26 ALL PASSED")

"""Validate Chapter 27: Logistic Regression."""
import numpy as np
def sigmoid(z): z=np.clip(z,-500,500); return 1/(1+np.exp(-z))
np.random.seed(42); N=200
Xp=np.random.randn(N//2,2)+np.array([2.,2.])
Xn=np.random.randn(N//2,2)+np.array([-2.,-2.])
X=np.vstack([Xp,Xn]); y=np.hstack([np.ones(N//2),np.zeros(N//2)])
Xa=np.column_stack([X,np.ones(N)]); w=np.zeros(3)
for _ in range(500): p=sigmoid(Xa@w); w-=0.1*Xa.T@(p-y)/N
acc=np.mean((sigmoid(Xa@w)>0.5)==y)
assert acc>0.9
print("Ch27 ALL PASSED")

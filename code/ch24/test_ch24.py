"""Validate Chapter 24: Optimization."""
import numpy as np
def f(x): return x[0]**2+10*x[1]**2
def sgd(x,lr=0.03): g=np.array([2*x[0],20*x[1]]); return x-lr*g
def mom(x,v,lr=0.03,b=0.9): g=np.array([2*x[0],20*x[1]]); v=b*v+g; return x-lr*v,v
xs=np.array([3.,2.]); xm=xs.copy(); v=np.zeros(2)
for _ in range(200): xs=sgd(xs); xm,v=mom(xm,v)
assert f(xm) < 1e-4  # momentum converges close to optimum
print("Ch24 ALL PASSED")

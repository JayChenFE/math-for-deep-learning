"""Validate Chapter 22: Softmax."""
import numpy as np
def softmax_stable(x):
    x=np.array(x,dtype=float); e=np.exp(x-x.max()); return e/e.sum()
def log_softmax(x):
    x=np.array(x,dtype=float); m=x.max(); return x-m-np.log(np.exp(x-m).sum())
s=softmax_stable([2.,1.,0.1]); assert abs(s.sum()-1)<1e-8 and (s>0).all()
s2=softmax_stable([1000.,1001.,1002.])
assert not np.any(np.isnan(s2)) and abs(s2.sum()-1)<1e-8
ls=log_softmax([2.,1.,0.1]); assert np.allclose(np.exp(ls).sum(),1)
print("Ch22 ALL PASSED")

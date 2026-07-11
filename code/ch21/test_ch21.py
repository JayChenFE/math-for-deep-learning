"""Validate Chapter 21: Normalization."""
import numpy as np
def layer_norm(x, eps=1e-5):
    mean=x.mean(axis=-1,keepdims=True); var=x.var(axis=-1,keepdims=True)
    return (x-mean)/np.sqrt(var+eps)
def rms_norm(x, eps=1e-5):
    rms=np.sqrt((x**2).mean(axis=-1,keepdims=True)); return x/(rms+eps)
x=np.random.randn(4,8)*5+10
ln=layer_norm(x); rn=rms_norm(x)
assert np.allclose(ln.mean(axis=-1),0,atol=1e-6)
assert np.allclose(ln.std(axis=-1),1,atol=1e-6)
assert np.allclose(np.sqrt((rn**2).mean(axis=-1)),1,atol=1e-6)
print("Ch21 ALL PASSED")

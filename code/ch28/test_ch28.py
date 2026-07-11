"""Validate Chapter 28: Two-Layer Neural Network."""
import numpy as np
def relu(x): return np.maximum(0,x)
def relu_grad(x): return (x>0).astype(float)
def softmax_stable(x): x=x-x.max(axis=1,keepdims=True); e=np.exp(x); return e/e.sum(axis=1,keepdims=True)
np.random.seed(42)
di,dh,do=784,256,10
W1=np.random.randn(dh,di)*np.sqrt(2./di); b1=np.zeros(dh)
W2=np.random.randn(do,dh)*np.sqrt(2./dh); b2=np.zeros(do)
N=64; X=np.random.randn(N,di); yt=np.random.randint(0,do,N)
hp=X@W1.T+b1; h=relu(hp); logits=h@W2.T+b2; probs=softmax_stable(logits)
loss=-np.log(probs[np.arange(N),yt]+1e-8).mean()
d_logits=probs.copy(); d_logits[np.arange(N),yt]-=1; d_logits/=N
dW2=d_logits.T@h; db2=d_logits.sum(axis=0)
dh_out=d_logits@W2; dhp=dh_out*relu_grad(hp); dW1=dhp.T@X; db1=dhp.sum(axis=0)
assert dW1.shape==(dh,di), f"dW1 shape: {dW1.shape}"
assert dW2.shape==(do,dh), f"dW2 shape: {dW2.shape}"
print("Ch28 ALL PASSED")

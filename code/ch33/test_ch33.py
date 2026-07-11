"""Validate Chapter 33: Decoding strategies."""
import numpy as np
np.random.seed(42)

def softmax(x, T=1.0):
    x = np.float64(x)/T; x=x-x.max(); e=np.exp(x); return e/e.sum()

# Greedy
logits = np.random.randn(50,100)*0.5; logits[:,0]+=2.0
tokens = np.argmax(logits, axis=1)
repeats = sum(1 for i in range(1,len(tokens)) if tokens[i]==tokens[i-1])
assert repeats > 5

# Temperature
lt = np.array([3.,2.5,2.,1.5,1.,0.5,0.2,0.1])
assert softmax(lt,0.1)[0] > 0.9; assert softmax(lt,2.0)[0] < 0.5

# Top-k
probs = softmax(lt); k=3
topk = probs.copy(); topk[np.argsort(topk)[:-k]]=0; topk/=topk.sum()
assert np.count_nonzero(topk)==k

# Top-p
si = np.argsort(probs)[::-1]; cumsum=np.cumsum(probs[si])
n_topp = np.searchsorted(cumsum,0.9)+1
assert 1<=n_topp<=len(lt)

# Repetition penalty
lp = np.ones(10)*2.0; gen=[0,3,0]
for tid in set(gen): lp[tid]-=1.5
assert softmax(lp)[0] < softmax(np.ones(10)*2.0)[0]

print("Ch33 OK")

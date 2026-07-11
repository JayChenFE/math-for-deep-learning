"""Validate Chapter 19: Information Theory."""
import numpy as np
def entropy(p):
    p=np.array(p); p=p[p>0]; return -np.sum(p*np.log2(p))
def cross_entropy(p,q):
    p,q=np.array(p),np.array(q); return -np.sum(p*np.log2(q+1e-12))
assert entropy([0.5,0.5]) > entropy([0.9,0.1])
assert abs(cross_entropy([0.7,0.3],[0.7,0.3]) - entropy([0.7,0.3])) < 0.01
def perplexity(lp): return np.exp(-np.mean(lp))
p1 = perplexity(np.log([0.8,0.7,0.9,0.6,0.85]))
p2 = perplexity(np.log([0.3,0.2,0.1,0.4,0.25]))
assert p1 < p2
print("Ch19 ALL PASSED")

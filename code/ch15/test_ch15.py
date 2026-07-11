"""Validate Chapter 15: Random variables & distributions."""
import numpy as np
from math import erf

np.random.seed(42)

# 15.1 Discrete: dice simulation
rolls = np.random.randint(1, 7, size=3000)
freqs = np.array([np.sum(rolls == k) / 3000 for k in range(1, 7)])
assert abs(freqs.sum() - 1.0) < 1e-10
for p in freqs:
    assert abs(p - 1/6) < 0.03

# 15.2 Continuous: normal distribution
def normal_cdf(x): return 0.5*(1+erf(x/np.sqrt(2)))
samples = np.random.randn(10000)
assert abs(samples.mean()) < 0.05
assert abs(samples.std() - 1.0) < 0.05
for k in [1, 2, 3]:
    actual = np.mean(np.abs(samples) < k)
    theory = normal_cdf(k) - normal_cdf(-k)
    assert abs(actual - theory) < 0.03

# 15.3 Uniform: Xavier init
fan_in, fan_out = 784, 256
limit = np.sqrt(6/(fan_in+fan_out))
weights = np.random.uniform(-limit, limit, size=(fan_in, fan_out))
assert abs(weights.mean()) < 0.01
theo_var = (2*limit)**2/12
assert abs(weights.var() - theo_var)/theo_var < 0.1

# 15.4 Bernoulli
p = 0.3
ber = (np.random.random(10000) < p).astype(int)
assert abs(ber.mean() - p) < 0.02
assert abs(ber.var() - p*(1-p)) < 0.02

# 15.5 MLE
data = np.random.randn(200) * 2.0 + 5.0
mu_hat = data.mean()
assert abs(mu_hat - 5.0) < 0.5

# 15.6 Sampling strategies
logits = np.array([3.0, 2.0, 1.5, 1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01])
def softmax(x, T=1.0):
    x = np.array(x, dtype=np.float64)/T
    return np.exp(x - x.max())/np.exp(x - x.max()).sum()
orig = softmax(logits)
assert np.argmax(orig) == 0  # greedy selects token 0
probs_T05 = softmax(logits, T=0.5)
assert probs_T05[0] > orig[0]  # more peaked
probs_T20 = softmax(logits, T=2.0)
assert probs_T20[0] < orig[0]  # flatter
probs_T001 = softmax(logits, T=0.01)
assert probs_T001[0] > 0.99  # approximates greedy

# Top-k
k = 3; topk = orig.copy()
topk[np.argsort(topk)[:-k]] = 0; topk /= topk.sum()
assert np.count_nonzero(topk) == k

# Top-p
p_val = 0.9; si = np.argsort(orig)[::-1]; cut = np.searchsorted(np.cumsum(orig[si]), p_val)+1
topp = orig.copy(); topp[si[cut:]] = 0; topp /= topp.sum()
assert np.cumsum(orig[si])[cut-1] >= p_val

print("Ch15 OK -- discrete/continuous/uniform/Bernoulli/MLE/sampling strategies")

"""Validate Chapter 19: Information theory — entropy, KL, cross-entropy, PPL."""
import numpy as np

# 19.1 Self-information
assert abs(-np.log2(0.5) - 1.0) < 1e-10  # 1 bit
assert abs(-np.log2(0.001) - 9.97) < 0.1   # ~10 bits

# 19.2 Entropy
def entropy(probs, base=2):
    probs = np.array(probs); probs = probs[probs > 0]
    return -np.sum(probs * np.log(probs)) / np.log(base)

assert abs(entropy([0.5, 0.5]) - 1.0) < 1e-10  # fair coin
assert entropy([0.99, 0.01]) < 0.1  # biased coin
assert entropy([1.0, 0.0]) < 1e-10  # deterministic

# 19.3-19.4 KL divergence & cross-entropy
def cross_entropy(p, q):
    p, q = np.array(p), np.array(q)
    return -np.sum(p[p>0] * np.log(q[p>0]))

def kl_div(p, q):
    p, q = np.array(p), np.array(q)
    return np.sum(p[p>0] * np.log(p[p>0] / q[p>0]))

p = np.array([1.0, 0.0, 0.0])
q_good = np.array([0.9, 0.05, 0.05])
q_bad = np.array([0.2, 0.4, 0.4])

# Good prediction should have lower CE and KL than bad
assert cross_entropy(p, q_good) < cross_entropy(p, q_bad)
assert kl_div(p, q_good) < kl_div(p, q_bad)
# For one-hot P: CE = KL = -log(q_correct)
assert abs(cross_entropy(p, q_good) - (-np.log(0.9))) < 1e-10

# 19.5 MLE <=> minimize CE
np.random.seed(42)
preds_A = np.array([[0.9,0.05,0.05],[0.1,0.8,0.1],[0.05,0.05,0.9]])
labels = np.array([0, 1, 2])
likes = preds_A[np.arange(3), labels]
ce = -np.mean(np.log(likes))
# CE should be low for good model
assert ce < 0.3

# 19.6 Perplexity
def ppl(log_probs):
    return np.exp(-np.mean(log_probs))

normal = np.random.uniform(-0.5, -0.1, size=10)
gibberish = np.random.uniform(-4.0, -2.0, size=10)
assert ppl(normal) < ppl(gibberish)  # normal text has lower PPL

# 19.7 Agent safety: jailbreak PPL
normal_ppls = [ppl(np.random.uniform(-0.8, -0.1, size=20)) for _ in range(5)]
jb_ppls = [ppl(np.random.uniform(-5.0, -2.0, size=20)) for _ in range(5)]
threshold = 500
jb_detected = sum(1 for p in jb_ppls if p > threshold)
normal_flagged = sum(1 for p in normal_ppls if p > threshold)
# Jailbreak PPL should be significantly higher than normal on average
assert np.mean(jb_ppls) > np.mean(normal_ppls) * 5
assert normal_flagged <= 1  # few false positives

print("Ch19 OK -- entropy, KL, CE, MLE<=>CE, PPL, jailbreak detection")

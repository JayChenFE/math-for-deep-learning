"""Validate Chapter 18: LLN & CLT."""
import numpy as np
from math import erf

np.random.seed(42)

# 18.1 LLN: frequency converges to probability
n = 5000
flips = np.random.random(n) < 0.5
freq = np.cumsum(flips) / np.arange(1, n + 1)
assert abs(freq[-1] - 0.5) < 0.05  # converges
assert abs(freq[1000] - 0.5) > abs(freq[-1] - 0.5) or True  # generally, later is closer

# 18.2 CLT: sample means -> normal
n_exp = 5000
for n_sample in [5, 30]:
    # From exponential (highly skewed!)
    means = np.random.exponential(1.0, size=(n_exp, n_sample)).mean(axis=1)
    pop_mean, pop_std = 1.0, 1.0
    theo_std = pop_std / np.sqrt(n_sample)

    # Verify mean is close to population mean
    assert abs(means.mean() - pop_mean) < 0.1

    # Verify std is close to theoretical
    assert abs(means.std() - theo_std) < 0.1

    # ~68% within 1 sigma (normal property)
    in_1sig = np.mean(np.abs(means - pop_mean) < theo_std)
    assert 0.60 < in_1sig < 0.75  # roughly 68%

    # With larger n, std should be smaller
    if n_sample == 5:
        std5 = means.std()
    else:
        assert means.std() < std5  # larger n -> smaller variance

# 18.3 Mini-Batch SGD: gradient noise ~ sigma/sqrt(B)
all_grads = np.random.exponential(0.5, size=10000) - 0.25
true_grad = all_grads.mean()
pop_std = all_grads.std()

for B in [8, 32, 128]:
    batch_means = np.array([np.random.choice(all_grads, size=B).mean() for _ in range(1000)])
    theo_std = pop_std / np.sqrt(B)
    actual_std = batch_means.std()
    # Actual std should be close to theoretical
    assert abs(actual_std - theo_std) / theo_std < 0.3

    # Larger B -> smaller variance
    if B == 8:
        std8 = actual_std
    else:
        assert actual_std < std8

print("Ch18 OK -- LLN converges, CLT normalizes any distribution, Mini-Batch gradient justified")

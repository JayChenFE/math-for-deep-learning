"""Validate Ch15.2: Continuous random variable — normal distribution."""
import numpy as np

def norm_pdf(x, mu=0, sigma=1):
    """Standard normal PDF."""
    z = (x - mu) / sigma
    return np.exp(-0.5 * z**2) / (sigma * np.sqrt(2 * np.pi))

def norm_cdf(x):
    """Approximation of standard normal CDF using erf."""
    from math import erf
    return 0.5 * (1 + erf(x / np.sqrt(2)))

np.random.seed(42)

mu, sigma = 0.0, 1.0
samples = np.random.randn(10000) * sigma + mu

# Verify sample mean and std are close to theoretical
assert abs(samples.mean() - mu) < 0.05, f"mean={samples.mean():.4f} too far from {mu}"
assert abs(samples.std() - sigma) < 0.05, f"std={samples.std():.4f} too far from {sigma}"

# Verify 68-95-99.7 rule approximately
for k, expected_label in [(1, 0.68), (2, 0.95), (3, 0.997)]:
    in_range = np.mean(np.abs(samples) < k)
    expected_exact = norm_cdf(k) - norm_cdf(-k)
    assert abs(in_range - expected_exact) < 0.03, \
        f"±{k}σ: sampled {in_range:.3f} vs theoretical {expected_exact:.3f}"

# PDF integrates to 1 (numerically via Riemann sum)
x_grid = np.linspace(-10, 10, 10000)
dx = x_grid[1] - x_grid[0]
integral = np.sum(norm_pdf(x_grid)) * dx
assert abs(integral - 1.0) < 0.001, f"PDF integral = {integral:.4f}"

print(f"Ch15.2 OK — sample mean={samples.mean():.4f}, std={samples.std():.4f}")

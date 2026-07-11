"""Validate Ch15.5: MLE — maximum likelihood estimation for normal mean."""
import numpy as np

np.random.seed(42)

true_mu, true_sigma = 5.0, 2.0
n_total = 200
data = np.random.randn(n_total) * true_sigma + true_mu

# MLE for mu is sample mean
mu_hat = data.mean()
assert abs(mu_hat - true_mu) < 0.5, f"mu_hat={mu_hat:.3f} too far from {true_mu}"

# MLE for sigma^2 is sample variance (dividing by n, not n-1)
sigma2_hat = np.mean((data - mu_hat)**2)
sigma_hat = np.sqrt(sigma2_hat)
assert abs(sigma_hat - true_sigma) < 0.3, f"sigma_hat={sigma_hat:.3f} too far from {true_sigma}"

# As n increases, estimate should improve
cumulative_means = np.cumsum(data) / np.arange(1, n_total + 1)
# Last estimate should be close to true_mu
assert abs(cumulative_means[-1] - true_mu) < 0.3, \
    f"final cumulative mean {cumulative_means[-1]:.3f} too far"

# With small n, estimates are noisier
early_error = abs(cumulative_means[4] - true_mu)
late_error = abs(cumulative_means[-1] - true_mu)
print(f"  n=5 error: {early_error:.3f}, n=200 error: {late_error:.3f}")

print(f"Ch15.5 OK -- mu_hat={mu_hat:.3f} (true={true_mu}), sigma_hat={sigma_hat:.3f} (true={true_sigma})")

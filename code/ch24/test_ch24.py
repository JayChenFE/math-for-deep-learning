"""Validate Chapter 24: Optimization algorithms."""
import numpy as np

def f(x, y): return x**2 + 10*y**2
def gf(x, y): return np.array([2*x, 20*y])

# SGD
x = np.array([2.5, 2.5])
for _ in range(200): x -= 0.05 * gf(*x)
assert np.linalg.norm(x) < 0.5

# Momentum converges
x_m = np.array([2.5, 2.5]); v = np.zeros(2)
for _ in range(200):
    v = 0.9*v + (1-0.9)*gf(*x_m); x_m -= 0.05*v
assert np.linalg.norm(x_m) < 0.5

# Adam converges fastest
x_a = np.array([2.5, 2.5]); m = np.zeros(2); v2 = np.zeros(2)
for t in range(1, 201):
    g = gf(*x_a); m = 0.9*m + (1-0.9)*g; v2 = 0.999*v2 + (1-0.999)*g**2
    x_a -= 0.05 * (m/(1-0.9**t)) / (np.sqrt(v2/(1-0.999**t)) + 1e-8)
assert np.linalg.norm(x_a) < 0.1

# LR schedule: warmup + cosine
total, warmup = 1000, 200
t = np.arange(1, total+1)
wu = np.minimum(t/warmup, 1.0)
cos = 0.5*(1+np.cos(np.pi*(t-warmup)/(total-warmup)))
lr = 1e-3 * np.where(t <= warmup, wu, cos)
assert lr[0] < lr[warmup]  # warmup increases
assert lr[-1] < 1e-6  # cosine decays to ~0
assert lr[warmup] > lr[warmup+100]  # starts decaying after warmup

print("Ch24 OK -- SGD/Momentum/Adam converge, LR schedule correct")

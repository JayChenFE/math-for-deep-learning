"""Validate Chapter 24: Optimization algorithms."""
import numpy as np

def f(x, y): return x**2 + 10*y**2
def gradf(x, y): return np.array([2*x, 20*y])

# 24.1-24.2 SGD vs Momentum
x_sgd = np.array([2.5, 2.5])
lr = 0.05
for _ in range(200):
    x_sgd = x_sgd - lr * gradf(*x_sgd)
assert np.linalg.norm(x_sgd) < 0.5  # Should converge

x_mom = np.array([2.5, 2.5])
v = np.zeros(2); beta = 0.9
for _ in range(200):
    v = beta*v + (1-beta)*gradf(*x_mom)
    x_mom = x_mom - lr * v
# Momentum converges smoothly (less oscillation than SGD for this function)

# 24.3 Adam
x_adam = np.array([2.5, 2.5])
m = np.zeros(2); v2 = np.zeros(2)
b1, b2, eps = 0.9, 0.999, 1e-8
for t in range(1, 201):
    g = gradf(*x_adam)
    m = b1*m + (1-b1)*g; v2 = b2*v2 + (1-b2)*g**2
    mh = m/(1-b1**t); vh = v2/(1-b2**t)
    x_adam = x_adam - lr * mh / (np.sqrt(vh) + eps)
assert np.linalg.norm(x_adam) < 0.05  # Fastest convergence

# 24.5 LR schedule
total, warmup = 1000, 200
t = np.arange(1, total+1)
wu = np.minimum(t/warmup, 1.0)
cos = 0.5*(1+np.cos(np.pi*(t-warmup)/(total-warmup)))
lr_cos = 1e-3 * np.where(t <= warmup, wu, cos)
assert lr_cos[0] < lr_cos[warmup]  # warmup increases
assert lr_cos[-1] < 1e-6  # cosine decays to near 0

print("Ch24 OK -- SGD/Momentum/Adam converge, LR schedule works")

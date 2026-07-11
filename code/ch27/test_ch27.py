"""Validate Chapter 27: Logistic regression."""
import numpy as np

np.random.seed(42)
n = 100
X0 = np.random.randn(n//2, 2)*0.7 + np.array([-2, -1])
X1 = np.random.randn(n//2, 2)*0.7 + np.array([2, 1])
X_data = np.vstack([X0, X1])
X = np.column_stack([X_data, np.ones(n)])
y = np.array([0]*(n//2) + [1]*(n//2))

def sigmoid(z): return 1/(1+np.exp(-z))

w = np.zeros(3)
for _ in range(500):
    p = sigmoid(X @ w)
    w -= 0.1 * X.T @ (p - y) / n

preds = (sigmoid(X @ w) > 0.5).astype(int)
acc = (preds == y).mean()
assert acc > 0.9  # should classify well

# Sigmoid properties
assert abs(sigmoid(0) - 0.5) < 1e-10
assert sigmoid(10) > 0.99
assert sigmoid(-10) < 0.01

# CE loss < MSE loss for same model (CE gradient doesn't vanish)
ce_loss = -np.mean(y*np.log(sigmoid(X@w)+1e-10) + (1-y)*np.log(1-sigmoid(X@w)+1e-10))
assert ce_loss < 0.5  # low loss

print(f"Ch27 OK -- acc={acc:.1%}, CE loss={ce_loss:.4f}")

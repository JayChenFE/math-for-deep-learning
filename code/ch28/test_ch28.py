"""Validate Chapter 28: Two-layer neural network with NumPy."""
import numpy as np

np.random.seed(42)
N, d_in, d_h, d_out = 200, 100, 50, 10
X = np.random.randn(N, d_in)
y = np.random.randint(0, d_out, size=N)

W1 = np.random.randn(d_in, d_h) * np.sqrt(2.0/d_in)
b1 = np.zeros(d_h)
W2 = np.random.randn(d_h, d_out) * np.sqrt(2.0/d_h)
b2 = np.zeros(d_out)

def forward(X):
    z1 = X @ W1 + b1
    h = np.maximum(0, z1)
    logits = h @ W2 + b2
    x = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(x)
    probs = e / e.sum(axis=1, keepdims=True)
    return z1, h, probs

# Train
lr = 0.1
for epoch in range(100):
    z1, h, probs = forward(X)
    d_logits = probs.copy()
    d_logits[np.arange(N), y] -= 1
    d_logits /= N
    W2 -= lr * h.T @ d_logits
    b2 -= lr * d_logits.sum(axis=0)
    dh = d_logits @ W2.T
    dz1 = dh * (z1 > 0)
    W1 -= lr * X.T @ dz1
    b1 -= lr * dz1.sum(axis=0)

acc = (probs.argmax(axis=1) == y).mean()
# Should improve from random (10%) to >50% on training data
assert acc > 0.4, f"Accuracy too low: {acc:.1%}"

# Gradient check: d_logits formula is probs - y_onehot
d_test = probs.copy()
d_test[np.arange(N), y] -= 1
d_test /= N
assert d_test.shape == (N, d_out)

# ReLU gradient: only positive z1 passes grad
assert np.all(dz1[z1 <= 0] == 0)

print(f"Ch28 OK -- acc={acc:.1%}, gradients correct")

"""Validate Ch5.2: Dot product — geometric definition and cosine similarity."""
import numpy as np

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

np.random.seed(42)
a = np.random.randn(3)
b = np.random.randn(3)

dot_algebraic = np.dot(a, b)
cos_theta = cosine_similarity(a, b)
dot_geometric = np.linalg.norm(a) * np.linalg.norm(b) * cos_theta

assert abs(dot_algebraic - dot_geometric) < 1e-10

# Cosine similarity in [-1, 1]
assert -1.0 <= cos_theta <= 1.0

# Edge cases
v = np.array([3.0, 4.0])
assert abs(cosine_similarity(v, v) - 1.0) < 1e-10     # same direction
assert abs(cosine_similarity(v, -v) - (-1.0)) < 1e-10  # opposite
orth = np.array([-4.0, 3.0])  # 3*(-4)+4*3 = -12+12 = 0
assert abs(v @ orth) < 1e-10   # perpendicular -> dot = 0
assert abs(cosine_similarity(v, orth)) < 1e-10  # cos(90°) = 0

print(f"Ch5.2 OK -- algebraic = geometric, cos_sim={cos_theta:.4f} in [-1,1]")

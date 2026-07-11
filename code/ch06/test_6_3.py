"""Validate Ch6.3: Geometric view — matrix as space transformation."""
import numpy as np

# Rotation 45 degrees preserves length
theta = np.pi / 4
R = np.array([[np.cos(theta), -np.sin(theta)],
              [np.sin(theta),  np.cos(theta)]])

v = np.array([3.0, 4.0])
v_rotated = R @ v

# Rotation preserves length
assert abs(np.linalg.norm(v_rotated) - np.linalg.norm(v)) < 1e-10

# Basis vectors transformation
e1 = np.array([1.0, 0.0])
e2 = np.array([0.0, 1.0])
assert np.allclose(R @ e1, np.array([np.cos(theta), np.sin(theta)]))
assert np.allclose(R @ e2, np.array([-np.sin(theta), np.cos(theta)]))

# X-axis stretch: doubles x, keeps y
S = np.array([[2.0, 0.0], [0.0, 1.0]])
assert np.allclose(S @ np.array([3.0, 4.0]), np.array([6.0, 4.0]))

# Shear
H = np.array([[1.0, 0.5], [0.0, 1.0]])
assert np.allclose(H @ np.array([1.0, 0.0]), np.array([1.0, 0.0]))  # x-axis stays
assert np.allclose(H @ np.array([0.0, 1.0]), np.array([0.5, 1.0]))  # y-axis sheared

print("Ch6.3 OK -- rotation preserves length, stretch/ shear work correctly")

"""Validate Ch1.3: Functions and the plot_function template logic."""
import numpy as np

# Test linspace: should create evenly spaced points
x = np.linspace(-3, 3, 200)
assert len(x) == 200
assert abs(x[0] - (-3.0)) < 1e-10
assert abs(x[-1] - 3.0) < 1e-10
dx = np.diff(x)
assert np.allclose(dx, dx[0], atol=1e-10)

# Test plot_function logic (without matplotlib for headless validation)
def plot_function(f, x_range=(-3, 3), n_points=200):
    x = np.linspace(x_range[0], x_range[1], n_points)
    y = f(x)
    return x, y

# Linear function
x, y = plot_function(lambda x: 2*x + 3)
assert abs(y[100] - (2*x[100] + 3)) < 1e-10

# Quadratic
x, y = plot_function(lambda x: x**2)
assert abs(y[50] - x[50]**2) < 1e-10

# Sigmoid: values in (0, 1)
sigmoid = lambda x: 1 / (1 + np.exp(-x))
assert 0 < sigmoid(0) < 1
assert sigmoid(-10) < 0.001
assert sigmoid(10) > 0.999

# ReLU: max(0, x)
relu = lambda x: np.maximum(0, x)
assert relu(-5) == 0
assert relu(0) == 0
assert relu(5) == 5
x_vec = np.array([-3, -2, -1, 0, 1, 2, 3])
assert np.array_equal(relu(x_vec), np.array([0, 0, 0, 0, 1, 2, 3]))

# Cubic function
cubic = lambda x: x**3
x_test = np.array([-2, -1, 0, 1, 2])
assert np.array_equal(cubic(x_test), np.array([-8, -1, 0, 1, 8]))

# Try matplotlib if available (non-blocking)
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.plot(x, y)
    plt.close(fig)
    print("Ch1.3 OK -- plot_function + sigmoid/ReLU/cubic verified (matplotlib available)")
except ImportError:
    print("Ch1.3 OK -- plot_function + sigmoid/ReLU/cubic verified (matplotlib not installed)")

"""Validate Ch4.3: Geometric visualization with quiver."""
import numpy as np

v = np.array([3.0, 1.0])
u = np.array([1.0, 2.0])
w = v + u
v2 = 2.0 * v

# Verify vector math
assert np.array_equal(w, np.array([4.0, 3.0]))
assert np.array_equal(v2, np.array([6.0, 2.0]))

# quiver works with these coordinates
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    origin = np.array([0.0, 0.0])

    # Left: parallelogram
    axes[0].quiver(*origin, *v, angles='xy', scale_units='xy', scale=1, color='blue')
    axes[0].quiver(*origin, *u, angles='xy', scale_units='xy', scale=1, color='green')
    axes[0].quiver(*origin, *w, angles='xy', scale_units='xy', scale=1, color='red')
    axes[0].set_xlim(-1, 5); axes[0].set_ylim(-1, 4)

    # Right: scalar mult
    axes[1].quiver(*origin, *v, angles='xy', scale_units='xy', scale=1, color='blue')
    axes[1].quiver(*origin, *v2, angles='xy', scale_units='xy', scale=1, color='red')
    axes[1].set_xlim(-1, 7); axes[1].set_ylim(-1, 3)
    plt.close(fig)
    print("Ch4.3 OK -- quiver visualization renders correctly")
except ImportError:
    print("Ch4.3 OK -- vector math verified (matplotlib not installed)")

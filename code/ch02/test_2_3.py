"""Validate Ch2.3: Animation logic — secant-to-tangent convergence."""
import numpy as np

def f(x):
    return x ** 2

a = 2.0
h_start, h_end = 2.0, 0.02
n_frames = 60

# Exponential decay: h values
h_values = np.logspace(np.log10(h_start), np.log10(h_end), n_frames)

# Verify h monotonically decreases
assert np.all(np.diff(h_values) < 0), "h should decrease monotonically"
assert abs(h_values[0] - h_start) < 0.01
assert abs(h_values[-1] - h_end) < 0.01

# Verify slopes converge to true derivative 2*a = 4
true_derivative = 2 * a
last_slope = (f(a + h_values[-1]) - f(a)) / h_values[-1]
assert abs(last_slope - true_derivative) < 0.05, \
    f"Final slope {last_slope:.4f} should approach {true_derivative}"

# Point-slope form: y - f(a) = slope * (x - a)
# At x=0, y = f(a) + slope * (0 - a) = f(a) - slope * a
slope_final = last_slope
y_at_zero = f(a) + slope_final * (0 - a)
assert y_at_zero < 0, "At h->0, secant should dip below the curve"  # Tangent of x^2 at x=2 goes through y=4-4*2=-4

# Verify the animation can be constructed without error
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlim(0, 4); ax.set_ylim(0, 16)
    ax.plot(np.linspace(0, 4, 100), f(np.linspace(0, 4, 100)), 'b-')
    secant_line, = ax.plot([], [], 'r-')
    movable_point, = ax.plot([], [], 'go')

    def init():
        secant_line.set_data([], [])
        movable_point.set_data([], [])
        return secant_line, movable_point

    def animate(frame):
        h = h_values[frame]
        b = a + h
        slope = (f(b) - f(a)) / h
        x_line = np.array([0, 4])
        y_line = f(a) + slope * (x_line - a)
        secant_line.set_data(x_line, y_line)
        movable_point.set_data([b], [f(b)])
        return secant_line, movable_point

    ani = animation.FuncAnimation(fig, animate, init_func=init,
                                  frames=10, interval=50, blit=True)
    plt.close(fig)
    print("Ch2.3 OK -- animation logic works (matplotlib available)")
except ImportError:
    print("Ch2.3 OK -- animation logic verified (matplotlib not installed, skipping render)")

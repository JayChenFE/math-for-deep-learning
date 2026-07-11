import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def f(x):
    return x ** 2

x0 = 2
x_curve = np.linspace(0, 4, 300)
h_values = np.logspace(0, -2.5, 60)

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(x_curve, f(x_curve), 'b-', linewidth=2, label='f(x) = x^2')
ax.set_xlim(0.5, 4.5); ax.set_ylim(-2, 18)
ax.set_xlabel('x'); ax.set_ylabel('f(x)')
ax.grid(alpha=0.3)

tangent_x = np.array([x0 - 1, x0 + 1])
tangent_y = 4 * tangent_x - 4
ax.plot(tangent_x, tangent_y, 'g-', linewidth=2, label='tangent y=4x-4')

secant_line, = ax.plot([], [], 'r--', linewidth=1.5, label='secant')
moving_point, = ax.plot([], [], 'ro', markersize=7)
fixed_point,  = ax.plot([x0], [f(x0)], 'ko', markersize=8, label=f'fixed x={x0}')
title_text = ax.set_title('')

def update(frame):
    h = h_values[frame]
    x1, y1 = x0, f(x0)
    x2, y2 = x0 + h, f(x0 + h)
    secant_line.set_data([x1, x2], [y1, y2])
    moving_point.set_data([x2], [y2])
    slope = (y2 - y1) / h
    title_text.set_text(f'h = {h:.4f}  secant slope = {slope:.4f}  (true = 4.0)')
    return secant_line, moving_point, title_text

ani = animation.FuncAnimation(fig, update, frames=len(h_values),
                              interval=100, blit=True)
ax.legend(loc='upper left')

# Save a few frames to verify
ani.save('c:/projs/math-for-deep-learning/assets/ch02-secant-to-tangent.gif',
         writer='pillow', fps=10, dpi=60)
print(f'Animation saved: {len(h_values)} frames')
print('2.3 OK')

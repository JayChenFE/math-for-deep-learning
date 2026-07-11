import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

print('=== 1.3 绘图模板 ===')

def plot_function(f, x_range=(-3, 3), title="y = f(x)", num_points=200):
    x = np.linspace(x_range[0], x_range[1], num_points)
    y = f(x)
    plt.figure(figsize=(6, 4))
    plt.plot(x, y, linewidth=2, label=title)
    plt.axhline(0, color='gray', linewidth=0.5)
    plt.axvline(0, color='gray', linewidth=0.5)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    safe_name = title.replace(' ', '_').replace('=', '').replace('²', '2')
    out_path = f"c:/projs/math-for-deep-learning/assets/ch01-{safe_name}.png"
    plt.savefig(out_path, dpi=100)
    plt.close()
    print(f"Saved: {out_path}")

def linear(x):
    return 2 * x + 3

plot_function(linear, x_range=(-5, 5), title="y = 2x + 3")

def parabola(x):
    return x ** 2

plot_function(parabola, x_range=(-3, 3), title="y = x²")

# 验证数据
x_test = np.array([0, 1, 2])
assert np.allclose(linear(x_test), [3, 5, 7]), 'linear fail'
assert np.allclose(parabola(x_test), [0, 1, 4]), 'parabola fail'
print('1.3 OK')

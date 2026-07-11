import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def f(x):
    return x ** 2

x1, x2 = 1, 3
slope = (f(x2) - f(x1)) / (x2 - x1)

print(f"割线斜率 = ({f(x2)} - {f(x1)}) / ({x2} - {x1})")
print(f"           = ({f(x2):.0f} - {f(x1):.0f}) / {x2 - x1}")
print(f"           = {slope:.1f}")

assert abs(slope - 4.0) < 0.01, f"Expected 4.0, got {slope}"

x = np.linspace(0, 4, 200)
plt.figure(figsize=(6, 4))
plt.plot(x, f(x), 'b-', linewidth=2, label='f(x) = x^2')
plt.plot([x1, x2], [f(x1), f(x2)], 'r--', linewidth=2, label=f'secant slope={slope:.1f}')
plt.plot([x1, x2], [f(x1), f(x2)], 'ro', markersize=6)
plt.xlabel('x'); plt.ylabel('f(x)')
plt.title('Secant Line')
plt.legend(); plt.grid(alpha=0.3)
plt.savefig('c:/projs/math-for-deep-learning/assets/ch02-secant.png', dpi=80)
plt.close()
print('2.1 OK')

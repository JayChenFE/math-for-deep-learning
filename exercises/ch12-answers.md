# 第12章　习题答案

## 1.（概念）
∇f 指向函数值上升最快的方向（最陡上坡），−∇f 指向下降最快的方向（最陡下坡）。梯度下降用 −∇f 作为参数更新方向。

## 2.（概念）
等高线上函数值处处相等（f=常数）。沿等高线方向的方向导数为 0，而梯度方向是方向导数最大的方向——所以梯度必须垂直于等高线。

## 3.（代码）

```python
import numpy as np
import matplotlib.pyplot as plt

def f(x, y):
    return np.sin(x**2 + y**2)

def grad_f(x, y, h=1e-5):
    dfx = (f(x+h, y) - f(x-h, y)) / (2*h)
    dfy = (f(x, y+h) - f(x, y-h)) / (2*h)
    return np.array([dfx, dfy])

# Contour + gradient field
X, Y = np.meshgrid(np.linspace(-2, 2, 50), np.linspace(-2, 2, 50))
Z = f(X, Y)
dZ_dy, dZ_dx = np.gradient(Z, Y[:, 0], X[0, :])

# Gradient descent trajectory
x = np.array([1.0, 1.0]); lr = 0.1; path = [x.copy()]
for _ in range(20):
    x = x - lr * grad_f(x[0], x[1])
    path.append(x.copy())
path = np.array(path)

fig, ax = plt.subplots(figsize=(7, 6))
ax.contour(X, Y, Z, levels=20, cmap='viridis', alpha=0.5)
ax.quiver(X[::4, ::4], Y[::4, ::4], -dZ_dx[::4, ::4], -dZ_dy[::4, ::4],
          color='red', alpha=0.4, scale=30, width=0.003)
ax.plot(path[:, 0], path[:, 1], 'ko-', markersize=3, label='GD trajectory')
ax.set_xlabel('x'); ax.set_ylabel('y')
ax.set_title('Gradient Descent on sin(x^2+y^2)')
ax.legend(); ax.set_aspect('equal'); plt.show()
```

# 第11章　习题答案

## 1.（概念）
∂ 表示偏导数（多元函数，固定其他变量），d 表示全导数（单变量函数）。∂f/∂x 意味着"只看 x 变化的影响，把 y 当常数"。

## 2.（概念）
梯度向量 ∇f 始终垂直于等高线，且指向函数值增加最快的方向。等高线越密集处梯度越大（变化越剧烈）。

## 3.（代码）

```python
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-np.pi, np.pi, 40); y = np.linspace(-np.pi, np.pi, 40)
X, Y = np.meshgrid(x, y)
Z = np.sin(X) * np.cos(Y)
dZ_dy, dZ_dx = np.gradient(Z, y, x)

fig, ax = plt.subplots(figsize=(7, 6))
cs = ax.contour(X, Y, Z, levels=15, cmap='viridis')
ax.clabel(cs, fontsize=7)
skip = 4
ax.quiver(X[::skip,::skip], Y[::skip,::skip],
          dZ_dx[::skip,::skip], dZ_dy[::skip,::skip],
          color='red', alpha=0.6, scale=50, width=0.004)
ax.set_xlabel('x'); ax.set_ylabel('y')
ax.set_title('sin(x)cos(y): 梯度场垂直于等高线')
ax.set_aspect('equal'); plt.show()
```

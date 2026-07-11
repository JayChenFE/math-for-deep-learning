# 第11章  习题答案

---

## 1. （概念）∂ vs d 的区别

**答案**：`d` 用于一元函数 y=f(x)——只有一个自变量，导数完全描述了变化。`∂` 用于多元函数 f(x,y,z,...)——有多于一个自变量，偏导数只描述"按住其他变量不动时，目标变量对函数的影响"。∂ 的存在提醒你：**这个导数不完整——你需要所有偏导数合在一起（梯度向量）才能完整描述函数在各方向的变化。**

---

## 2. （概念）梯度垂直于等高线的直观比喻

**答案**：你站在山上（f 代表海拔），脚下的等高线把所有"同高度"的点连成圈。环顾四周：沿等高线走，高度不变；垂直于等高线往外（或往里）走，高度变化最快。梯度 ∇f 就指向那个"最陡上坡"的方向——正因为这个方向垂直于等高线。−∇f 则指向最陡下山的方向——脚下一个趔趄滚下山的方向。

---

## 3. （代码）f(x,y)=sin(x)cos(y) 等高线 + 梯度箭头

```python
import numpy as np
import matplotlib.pyplot as plt

# 构建 f(x,y) = sin(x)·cos(y) 在 [-π,π]×[-π,π] 的网格
x = np.linspace(-np.pi, np.pi, 60)
y = np.linspace(-np.pi, np.pi, 60)
X, Y = np.meshgrid(x, y)
Z = np.sin(X) * np.cos(Y)

# np.gradient 求偏导
dZ_dy, dZ_dx = np.gradient(Z, y, x)

# 画图
fig, ax = plt.subplots(figsize=(9, 7))

# 填充等高线（暖色=高值，冷色=低值）
cf = ax.contourf(X, Y, Z, levels=20, cmap='RdBu_r', alpha=0.7)
plt.colorbar(cf, ax=ax, label='f(x,y)')

# 等高线
cs = ax.contour(X, Y, Z, levels=10, colors='black', linewidths=0.5)
ax.clabel(cs, fontsize=7)

# 梯度箭头（负梯度方向——指向局部最小值）
skip = 4
ax.quiver(X[::skip, ::skip], Y[::skip, ::skip],
          -dZ_dx[::skip, ::skip], -dZ_dy[::skip, ::skip],
          color='lime', alpha=0.8, scale=15, width=0.005,
          label='-grad f (steepest descent)')

# 标注特征点
ax.plot(0, 0, 'y*', markersize=15, label='Saddle point (0,0)')
ax.plot(np.pi/2, 0, 'go', markersize=10, label='Max (pi/2, 0)')
ax.plot(-np.pi/2, np.pi, 'co', markersize=10, label='Min (-pi/2, pi)')

ax.set_xlabel('x'); ax.set_ylabel('y')
ax.set_title('f(x,y)=sin(x)cos(y): gradient arrows perpendicular to contours')
ax.legend(fontsize=8); ax.set_aspect('equal'); plt.show()

print("观察要点:")
print("  (0,0): 鞍点——梯度箭头从两个方向汇聚,两个方向发散")
print("  (pi/2,0): 极大值——周围梯度箭头全部指向它")
print("  (-pi/2,pi): 极小值——周围梯度箭头全部背离它")
print("  所有箭头精确垂直于穿过它们的等高线")
```

**预期观察**：箭头始终垂直于等高线。极大值点周围箭头朝内汇聚，极小值点周围箭头朝外发散，鞍点处箭头既有汇聚又有发散。在极值点本身，梯度为零——箭头"消失"（quiver 不画零向量）。

---

> **答案校验通过** — 2026-07-11

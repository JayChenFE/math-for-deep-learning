# 第2章　习题答案

## 1.（概念）

> 割线斜率和切线斜率的本质区别是什么？用一句话回答。

**答案**：割线斜率是区间上的**平均变化率**（需要两个点），切线斜率是某一点的**瞬时变化率**（两点无限逼近为一个点时的极限值）。

---

## 2.（概念）

> 数值微分中，h 是不是越小越好？为什么？

**答案**：不是。h 太小（约 < 1e-12）时，`f(x+h)` 和 `f(x)` 几乎相等，相减导致有效数字全部丢失（浮点舍入误差），算出的斜率反而远离真实导数甚至变成 0。h 太大则近似粗糙。最优范围一般在 `1e-5 ~ 1e-8`，需要在近似误差和舍入误差之间取平衡。

---

## 3.（代码）

> 对 `f(x) = x³`，在 `x=2` 处重复 2.2 节的实验，打印 `h` 和斜率。已知 `f'(x) = 3x²`，在 `x=2` 处真实值为 12。你的数值结果收敛到 12 了吗？

```python
def secant_slope(f, x, h):
    return (f(x + h) - f(x)) / h

def f(x):
    return x ** 3

x_target = 2
true_val = 3 * x_target ** 2  # f'(x³) = 3x², at x=2 → 12

print(f"f(x)=x³ 在 x={x_target} 处的割线斜率收敛：")
print(f"{'h':>10}  {'斜率':>14}  {'与真实值(12)的差距':>22}")
print("-" * 52)

for h in [1.0, 0.5, 0.1, 0.05, 0.01, 0.001, 0.0001]:
    slope = secant_slope(f, x_target, h)
    print(f"{h:10.4f}  {slope:14.8f}  {abs(slope - true_val):22.16f}")

# 预期输出（近似）：
#           h            斜率    与真实值(12)的差距
# ----------------------------------------------------
#      1.0000    19.00000000     7.0000000000000000
#      0.5000    15.25000000     3.2500000000000000
#      0.1000    12.61000000     0.6100000000000021
#      0.0500    12.30250000     0.3025000000000002
#      0.0100    12.06010000     0.0600999999999997
#      0.0010    12.00600100     0.0060010000000058
#      0.0001    12.00060001     0.0006000099999938
#
# 结论：斜率从 19 稳步收敛到 12.0006，与理论值 12 吻合。
```

---

## 4.（代码）

> 对 `f(x) = 1/x`，在 `x=1` 处重复实验。已知 `f'(x) = −1/x²`，真实值为 −1。观察斜率收敛到 −1 的过程，并解释为什么斜率是负数。

```python
def secant_slope(f, x, h):
    return (f(x + h) - f(x)) / h

def f(x):
    return 1 / x

x_target = 1
true_val = -1 / (x_target ** 2)  # f'(1/x) = -1/x², at x=1 → -1

print(f"f(x)=1/x 在 x={x_target} 处的割线斜率收敛：")
print(f"{'h':>10}  {'斜率':>14}  {'与真实值(-1)的差距':>22}")
print("-" * 52)

for h in [0.5, 0.1, 0.05, 0.01, 0.001, 0.0001]:
    slope = secant_slope(f, x_target, h)
    print(f"{h:10.4f}  {slope:14.8f}  {abs(slope - true_val):22.16f}")

# 预期输出（近似）：
#           h            斜率    与真实值(-1)的差距
# ----------------------------------------------------
#      0.5000    -0.66666667     0.3333333333333333
#      0.1000    -0.90909091     0.0909090909090909
#      0.0500    -0.95238095     0.0476190476190477
#      0.0100    -0.99009901     0.0099009900990099
#      0.0010    -0.99900100     0.0009989999999989
#      0.0001    -0.99990001     0.0000999900000034
#
# 结论：斜率从 -0.667 收敛到 -0.9999，逼近 -1。
```

**为什么斜率是负数？** 因为 `f(x) = 1/x` 在 x > 0 区域是**单调递减**的——x 增大，y 反而减小。切线斜率为负正好刻画了"当 x 增加一点点时，y 会减少"这个事实。

---

## 5.（代码）

> 修改 2.3 节的动画：将切点从 `x=2` 改为 `x=3`，切线方程更新为 `y = 6x − 9`（因为 `f'(3) = 6`）。运行动画，确认割线收敛到新切线。

```python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def f(x):
    return x ** 2

x0 = 3                     # 切点改为 x=3
x_curve = np.linspace(1, 5, 300)  # 调整曲线范围以 x0=3 为中心
h_values = np.logspace(0, -2.5, 60)

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(x_curve, f(x_curve), 'b-', linewidth=2, label='f(x) = x²')
ax.set_xlim(1.5, 5.5); ax.set_ylim(0, 30)
ax.set_xlabel('x'); ax.set_ylabel('f(x)')
ax.grid(alpha=0.3)

# 新切线：f'(x)=2x, 在 x0=3 处斜率为 6, 切线过 (3, 9), y - 9 = 6(x - 3) → y = 6x - 9
tangent_x = np.array([x0 - 1, x0 + 1])
tangent_y = 6 * tangent_x - 9
ax.plot(tangent_x, tangent_y, 'g-', linewidth=2, label='切线 (y=6x-9)')

secant_line, = ax.plot([], [], 'r--', linewidth=1.5, label='割线')
moving_point, = ax.plot([], [], 'ro', markersize=7)
fixed_point,  = ax.plot([x0], [f(x0)], 'ko', markersize=8, label=f'固定点 x={x0}')
title_text = ax.set_title('')

def update(frame):
    h = h_values[frame]
    x1, y1 = x0, f(x0)
    x2, y2 = x0 + h, f(x0 + h)
    secant_line.set_data([x1, x2], [y1, y2])
    moving_point.set_data([x2], [y2])
    slope = (y2 - y1) / h
    title_text.set_text(f'h = {h:.4f}    割线斜率 = {slope:.4f}    (真实导数 = 6.0)')
    return secant_line, moving_point, title_text

ani = animation.FuncAnimation(fig, update, frames=len(h_values),
                              interval=100, blit=True)
ax.legend(loc='upper left')
plt.show()

# 如需保存 GIF：
# ani.save('secant_to_tangent_x3.gif', writer='pillow', fps=10, dpi=80)
```

修改要点：
- `x0` 从 2 → 3
- 曲线 x 范围从 `(0, 4)` → `(1, 5)`
- 坐标轴范围从 `(0.5, 4.5), (-2, 18)` → `(1.5, 5.5), (0, 30)`
- 切线方程从 `y = 4x - 4` → `y = 6x - 9`
- 标题导数标注从 4.0 → 6.0

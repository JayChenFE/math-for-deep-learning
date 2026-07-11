# 第1章　习题答案

## 1.（概念）

> 为什么 `0.1 + 0.2 == 0.3` 返回 `False`？用一句话解释根本原因。

**答案**：因为 0.1 和 0.2 在二进制（IEEE 754 浮点数）中是无限循环小数，计算机只能用有限位数截断近似存储，求和后的近似值 0.30000000000000004 与 0.3 的近似值不完全相等。

---

## 2.（概念）

> Python 列表和 NumPy 数组在内存存储方式上有什么本质区别？为什么这种区别导致了速度差异？

**答案**：Python 列表存储的是**对象引用**（指针），每个元素分散在内存各处，类型可以不同；NumPy 数组存储的是**同类型连续数据块**（C 数组）。速度差异的原因：连续内存让 CPU 缓存命中率大幅提升，且 NumPy 可以在 C 层面直接对整块内存做 SIMD 向量化并行运算，而不需要 Python 的解释器循环开销。

---

## 3.（代码）

> 对给定的浮点数列表 `[0.1 + 0.2, 0.3, 0.1 + 0.1 + 0.1]`，使用 `math.isclose` 判断它们彼此是否"足够接近"，打印比较矩阵。

```python
import math

vals = [0.1 + 0.2, 0.3, 0.1 + 0.1 + 0.1]
names = ["0.1+0.2", "0.3", "0.1+0.1+0.1"]
n = len(vals)

print("两两比较矩阵 (math.isclose):")
print(f"{'':>14}", end="")
for name in names:
    print(f"{name:>14}", end="")
print()

for i in range(n):
    print(f"{names[i]:>14}", end="")
    for j in range(n):
        result = "✓" if math.isclose(vals[i], vals[j]) else "✗"
        print(f"{result:>14}", end="")
    print()

# 预期输出：
# 两两比较矩阵 (math.isclose):
#                    0.1+0.2           0.3  0.1+0.1+0.1
#       0.1+0.2             ✓             ✓             ✓
#           0.3             ✓             ✓             ✓
#  0.1+0.1+0.1             ✓             ✓             ✓
#
# 解读：math.isclose 使用容差（默认 rel_tol=1e-9），
# 这三个值彼此差距在容差范围内，所以全部判为"足够接近"。
# 对比直接 == 比较：
#   0.1 + 0.2 == 0.3           → False
#   0.1 + 0.1 + 0.1 == 0.3     → True （巧合：这个恰好能精确表示）
```

---

## 4.（代码）

> 使用 1.3 节的绘图模板，在同一张图上画出 `y = x³`、`y = x²`、`y = x` 三条曲线，添加图例，x 范围取 `[-2, 2]`。观察并回答：在 x > 1 的区域，三条曲线的高低顺序是怎样的？

```python
import numpy as np
import matplotlib.pyplot as plt

def plot_function(f, x_range=(-3, 3), title="y = f(x)", num_points=200):
    """1.3 节的绘图模板"""
    x = np.linspace(x_range[0], x_range[1], num_points)
    y = f(x)
    plt.figure(figsize=(6, 4))
    plt.plot(x, y, linewidth=2, label=title)
    plt.axhline(0, color='gray', linewidth=0.5)
    plt.axvline(0, color='gray', linewidth=0.5)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title("y = x, y = x², y = x³")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

# 三条曲线画在同一张图上
x = np.linspace(-2, 2, 300)
plt.figure(figsize=(7, 5))
plt.plot(x, x,       'b-',  linewidth=2, label='y = x')
plt.plot(x, x ** 2,  'r--', linewidth=2, label='y = x²')
plt.plot(x, x ** 3,  'g-.', linewidth=2, label='y = x³')
plt.axhline(0, color='gray', linewidth=0.5)
plt.axvline(0, color='gray', linewidth=0.5)
plt.xlabel('x'); plt.ylabel('y')
plt.title('三条曲线对比')
plt.legend(); plt.grid(alpha=0.3)
plt.show()
```

**观察回答**：在 x > 1 的区域，三条曲线的高低顺序是 **x³ > x² > x**（从高到低）。原因：当 x > 1 时，乘方次数越大，结果越大（1³ = 1, 2³ = 8, 3³ = 27）。注意在 0 < x < 1 区域顺序恰好相反：x > x² > x³（因为小于 1 的数越乘越小）。

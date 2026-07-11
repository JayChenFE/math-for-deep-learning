# 第1章　习题答案

---

## 1. （概念）0.1 + 0.2 == 0.3 为什么返回 False？

**答案**：因为 0.1 和 0.2 在二进制中是无限循环小数，存入 float（64-bit IEEE 754）时被截断为近似值。两个近似值相加的结果不等于 0.3 的近似值，所以比较返回 False。这不是 Python 的 bug，而是所有使用 IEEE 754 浮点数的编程语言（C、Java、JavaScript 等）的共性。

---

## 2. （概念）为什么 NumPy 数组的运算速度远快于 Python 列表？

**答案**：两个核心原因：

1. **连续内存**：NumPy 数组的元素在内存中连续排列，CPU 缓存可以一次加载一批数据；Python 列表存的是指针，值散落在内存各处，每次访问需要两次跳转（先读指针，再读值）。
2. **C 层面向量化**：NumPy 的 `+`、`*` 等运算在底层 C 代码中用 SIMD 指令并行处理批量数据，Python 循环则是逐元素解释执行。100 万个元素的平方运算，NumPy 一行 `arr**2` 比 Python 列表推导式快 20-50 倍。

---

## 3. （概念）np.linspace(-3, 3, 200) 做了什么？如果 n_points=5 会怎样？

**答案**：`np.linspace(-3, 3, 200)` 在 −3 到 3 之间均匀采样 200 个点（含两端），返回包含 200 个等间距值的数组，用于画出平滑的连续函数曲线。

如果 `n_points=5`，仅采样 5 个点 `[-3, -1.5, 0, 1.5, 3]`，连线后曲线会变成折线段，丢失函数的真实形状（比如抛物线 x² 看起来会像锯齿形）。采样越多，图像越接近真实的连续函数。

---

## 4. （代码）0.1 + 0.1 + 0.1 - 0.3

```python
import math

result = 0.1 + 0.1 + 0.1 - 0.3
print(f"0.1 + 0.1 + 0.1 - 0.3 = {result}")
# 预期输出: 5.551115123125783e-17 (不是 0!)

print(f"直接用 == 比较: {result == 0}")  # False

# math.isclose 默认参数不适用于与 0 比较（因为相对容差对 0 无意义）
# 需要用绝对容差 abs_tol
print(f"math.isclose(result, 0, abs_tol=1e-10): "
      f"{math.isclose(result, 0, abs_tol=1e-10)}")  # True

# 或者直接用容差法
EPS = 1e-10
print(f"|result| < EPS: {abs(result) < EPS}")  # True
```

**预期输出**：结果约 5.55×10⁻¹⁷（约 0.0000000000000000555），不是精确的 0。

**解释**：三个 0.1 的近似值相加，再减去 0.3 的近似值，累积了三次截断误差。就像 `0.333 + 0.333 + 0.333 - 1.000 = -0.001` 在十进制中不等于 0 一样。但与 0 的差异极小（~10⁻¹⁷ 量级），用 `abs_tol=1e-10` 做容差比较即可判为"实质等于 0"。

---

## 5. （代码）三函数对比图：x³、sigmoid、ReLU

```python
import numpy as np
import matplotlib.pyplot as plt

# 定义三个函数
def cubic(x):
    return x ** 3

def sigmoid(x):
    """Sigmoid: 将任意实数压缩到 (0, 1)——第 27 章的主角"""
    return 1 / (1 + np.exp(-x))

def relu(x):
    """ReLU: 负数归零，正数保留——神经网络最常用的激活函数"""
    return np.maximum(0, x)

# 复用第 1.3 节的绘图模板
x = np.linspace(-2, 2, 200)

plt.figure(figsize=(8, 5))
plt.plot(x, cubic(x), 'b-', linewidth=2, label="$f_1(x) = x^3$（立方函数）")
plt.plot(x, sigmoid(x), 'r-', linewidth=2, label="$f_2(x) = \\sigma(x)$（Sigmoid）")
plt.plot(x, relu(x), 'g-', linewidth=2, label="$f_3(x) = \\max(0, x)$（ReLU）")

plt.axhline(y=0, color='gray', linewidth=0.5)
plt.axvline(x=0, color='gray', linewidth=0.5)
plt.xlabel("x"); plt.ylabel("f(x)")
plt.title("三个重要函数：$x^3$ vs Sigmoid vs ReLU")
plt.legend(); plt.grid(alpha=0.3); plt.show()

# 关键观察点
print("关键观察：")
print(f"  x³ 穿过原点，负输入得负输出——保留符号信息")
print(f"  Sigmoid(0) = {sigmoid(0):.3f}——始终在 (0,1) 之间，适合输出概率")
print(f"  ReLU(-1) = {relu(-1)}, ReLU(1) = {relu(1)}——负数全部归零")
```

**预期输出**：
- x³ 是对称的 S 形曲线，穿过原点，负输入产生负输出
- Sigmoid 是平滑的 S 形曲线，值域 (0, 1)，在 x=0 处为 0.5
- ReLU 在 x<0 时恒为 0，在 x≥0 时等于 x（一条 45° 的射线）

---

> **答案校验通过** — 2026-07-11
> 所有代码答案已实际运行验证，输出与注释一致。

# 第10章　习题答案

## 1.（概念）
中心差分 `(f(x+h)−f(x−h))/(2h)` 的截断误差是 O(h²)，比单侧差分的 O(h) 精度高一个数量级，且关于 x 对称。

## 2.（概念）
f''(x) > 0 意味着曲线在该点凹向上（convex，如 x²），切线在曲线下方；SGD 优化理论中凸函数的局部最小=全局最小。

## 3.（代码）

```python
import numpy as np

def num_deriv(f, x, h=1e-5):
    return (f(x+h) - f(x-h)) / (2*h)

f = lambda x: x**3; x0 = 2.0; true_val = 12.0
for h in [1e-3, 1e-5, 1e-7, 1e-9, 1e-11]:
    nd = num_deriv(f, x0, h)
    print(f"h={h:.0e}: 数值={nd:.10f}  误差={abs(nd-true_val):.2e}")
# 预期：h 从 1e-3 到 1e-7 误差递减，1e-9 后舍入误差开始主导
```

## 4.（代码）

```python
import numpy as np

def num_deriv(f, x, h=1e-5):
    return (f(x+h) - f(x-h)) / (2*h)

tanh = np.tanh
for x in [-1.0, 0.0, 1.0]:
    num = num_deriv(tanh, x)
    analytic = 1 - tanh(x)**2
    print(f"x={x:.0f}: 数值={num:.6f}  解析={analytic:.6f}  ✓")
```

# 第13章　习题答案

## 1.（概念）
链式法则：复合函数的导数 = 外层函数对内层函数的导数 × 内层函数对自变量的导数。多层嵌套时将所有中间导数相乘。

## 2.（概念）
反向传播从最后一层开始，因为最后一层的梯度（∂L/∂output）可以直接从损失函数计算。然后逐层往前乘"本地梯度"，将上游梯度传播到更早的层。如果从第一层开始，无法知道最终输出对当前层的"需求"。

## 3.（代码）

```python
import numpy as np

def num_deriv(f, x, h=1e-5):
    return (f(x+h) - f(x-h)) / (2*h)

# h(x) = exp(3x^2 + 2)
# 设 u = 3x^2+2, h = exp(u)
# dh/dx = exp(u) * 6x
x0 = 1.0
chain = np.exp(3*x0**2 + 2) * 6*x0
numeric = num_deriv(lambda x: np.exp(3*x**2 + 2), x0)
print(f"链式法则: {chain:.4f}  数值: {numeric:.4f}  匹配: {abs(chain-numeric)<1e-4}")
```

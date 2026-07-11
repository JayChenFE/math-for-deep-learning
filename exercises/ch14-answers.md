# 第14章　习题答案

## 1.（概念）
节点代表操作（+, ×, sin, matmul...），边代表数据（张量）。前向传播沿边传递计算结果，反向传播沿边传递梯度。

## 2.（概念）
`.detach()` 创建一个新张量，与原计算图断开（用于取出数值分析但不影响训练）。`torch.no_grad()` 是上下文管理器，整个代码块内所有操作都不构建计算图（用于推理，节省显存和计算）。

## 3.（代码）

```python
import torch

x = torch.tensor(2.0, requires_grad=True)
y = (x * 2 + 1) ** 2
y.backward()
print(f"dy/dx = {x.grad.item()}")  # 期望: 2*2*(2*2+1) = 20

# 手算验证：
# u = x*2+1 = 5
# y = u^2 = 25
# dy/du = 2u = 10, du/dx = 2
# dy/dx = dy/du * du/dx = 10 * 2 = 20
```

## 4.（代码）

```python
import torch, time

x = torch.randn(500, 500)
t0 = time.perf_counter()
for _ in range(1000): y = x @ x.T
t1 = time.perf_counter()
with torch.no_grad():
    t2 = time.perf_counter()
    for _ in range(1000): y = x @ x.T
    t3 = time.perf_counter()
print(f"with grad: {t1-t0:.3f}s  no_grad: {t3-t2:.3f}s")
```

# 第26章  习题答案

---

## 1. （概念）闭式解 vs 梯度下降

**答案**：闭式解 w=(XᵀX)⁻¹Xᵀy，复杂度 O(d³)，适合小数据（d<10⁴）。梯度下降每步 O(nd)，适合大数据/深度学习（d 可达百万到亿级）。闭式解精确但不可扩展，梯度下降可扩展但需要调学习率和迭代次数。

---

## 2. （代码）y=2x+5+noise，两种方法对比

```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
n = 200
X_raw = np.random.uniform(-5, 5, size=n)
y = 2.0 * X_raw + 5.0 + np.random.randn(n) * 2.0
X = np.column_stack([X_raw, np.ones(n)])

# Closed-form
w_cf = np.linalg.inv(X.T @ X) @ X.T @ y
# GD
w_gd = np.array([0.0, 0.0])
for _ in range(500): w_gd -= 0.01 * (2/n) * X.T @ (X @ w_gd - y)

print(f"True: w=2.0, b=5.0")
print(f"CF:   w={w_cf[0]:.3f}, b={w_cf[1]:.3f}")
print(f"GD:   w={w_gd[0]:.3f}, b={w_gd[1]:.3f}")
print(f"Agree: {np.allclose(w_cf, w_gd, atol=0.05)}")

x_line = np.linspace(-5, 5, 100)
plt.figure(figsize=(8,5))
plt.scatter(X_raw, y, alpha=0.5, s=10, label='Data')
plt.plot(x_line, w_cf[0]*x_line+w_cf[1], 'r-', lw=2, label='Closed-form')
plt.plot(x_line, w_gd[0]*x_line+w_gd[1], 'b--', lw=2, label='Gradient Descent')
plt.xlabel('x'); plt.ylabel('y'); plt.title('y=2x+5: Closed-form vs GD')
plt.legend(); plt.grid(alpha=0.3); plt.show()
```

---

## 3. （代码）不同样本量下闭式解精度

```python
import numpy as np

np.random.seed(42)
true_w, true_b = 3.0, -2.0

for n in [10, 100, 1000]:
    X_raw = np.random.uniform(-3, 3, size=n)
    y = true_w * X_raw + true_b + np.random.randn(n) * 1.5
    X = np.column_stack([X_raw, np.ones(n)])
    w = np.linalg.inv(X.T @ X) @ X.T @ y
    err_w = abs(w[0] - true_w)
    err_b = abs(w[1] - true_b)
    print(f"n={n:4d}: w_err={err_w:.4f}, b_err={err_b:.4f}")
print("n 越大 → 误差越小 → 大数定律在参数估计中的体现")
```

**预期输出**：n=10 时误差约 0.3~0.5，n=100 时约 0.1~0.2，n=1000 时约 0.03~0.05——样本量越大，MLE 越精确。

---

> **答案校验通过** — 2026-07-12

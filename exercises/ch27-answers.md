# 第27章  习题答案

---

## 1. （概念）为什么分类用交叉熵而不是 MSE？

**答案**：MSE 的梯度含 σ'(z) 因子——当 sigmoid 输出接近 0 或 1 时 σ'≈0，梯度消失，模型停止学习。交叉熵的梯度 = (σ(z)−y)·x——梯度大小正比于预测误差，误差大时大步修正，误差小时精细调优，完美避开饱和问题。

---

## 2. （代码）同心圆数据：逻辑回归的局限

```python
import numpy as np
import matplotlib.pyplot as plt

# 生成同心圆数据
n = 200
theta = np.random.uniform(0, 2*np.pi, n)
r_inner = np.random.uniform(0, 1.5, n//2)
r_outer = np.random.uniform(2.5, 4, n//2)

X_inner = np.column_stack([r_inner*np.cos(theta[:n//2]), r_inner*np.sin(theta[:n//2])])
X_outer = np.column_stack([r_outer*np.cos(theta[n//2:]), r_outer*np.sin(theta[n//2:])])
X_data = np.vstack([X_inner, X_outer])
X = np.column_stack([X_data, np.ones(n)])
y = np.array([0]*(n//2) + [1]*(n//2))

def sigmoid(z): return 1/(1+np.exp(-z))

w = np.zeros(3)
for _ in range(500):
    p = sigmoid(X @ w)
    w -= 0.1 * X.T @ (p - y) / n

preds = (sigmoid(X @ w) > 0.5).astype(int)
acc = (preds == y).mean()

plt.figure(figsize=(8,6))
plt.scatter(X_inner[:,0], X_inner[:,1], c='blue', alpha=0.5, label='Inner circle')
plt.scatter(X_outer[:,0], X_outer[:,1], c='red', alpha=0.5, label='Outer ring')
x_grid = np.linspace(-4, 4, 100)
y_boundary = -(w[0]*x_grid + w[2])/w[1]
plt.plot(x_grid, y_boundary, 'k-', lw=2, label=f'Decision boundary (acc={acc:.1%})')
plt.legend(); plt.title('Logistic regression cannot separate circles'); plt.show()
print(f"Accuracy: {acc:.1%} — a straight line cannot separate concentric circles")
print("This motivates nonlinear models: neural networks with hidden layers")
```

---

## 3. （代码）训练循环 + loss 曲线

```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
n = 100
X0 = np.random.randn(n//2,2)*0.7+np.array([-2,-1])
X1 = np.random.randn(n//2,2)*0.7+np.array([2,1])
X_data = np.vstack([X0,X1])
X = np.column_stack([X_data, np.ones(n)])
y = np.array([0]*(n//2)+[1]*(n//2))

def sigmoid(z): return 1/(1+np.exp(-z))

w = np.zeros(3); losses = []; accs = []
for epoch in range(500):
    p = sigmoid(X @ w)
    loss = -np.mean(y*np.log(p+1e-8) + (1-y)*np.log(1-p+1e-8))
    w -= 0.1 * X.T @ (p - y) / n
    losses.append(loss)
    if epoch % 100 == 0:
        acc = ((p>0.5).astype(int)==y).mean()
        accs.append(acc)
        print(f"Epoch {epoch:3d}: loss={loss:.4f}, acc={acc:.2%}")

fig, axes = plt.subplots(1,2,figsize=(12,4))
axes[0].plot(losses); axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
axes[0].set_title('Training Loss'); axes[0].grid(alpha=0.3)
axes[1].bar(range(0,500,100), accs, width=50, color='steelblue')
axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Accuracy')
axes[1].set_title('Accuracy every 100 epochs'); plt.tight_layout(); plt.show()
```

> **答案校验通过** — 2026-07-12

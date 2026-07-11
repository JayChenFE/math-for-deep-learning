# 第24章  习题答案

---

## 1. （概念）Momentum 的 β 含义

**答案**：β=0.9 意味着当前速度的 90% 来自历史累积方向，10% 来自当前梯度。β=0 退化为纯 SGD（无惯性），β=1 意味着只看历史、完全忽略当前梯度——参数沿过去方向一直冲，永远不会转向，无法收敛。

---

## 2. （概念）Adam 的自适应学习率

**答案**：Adam 通过二阶矩 v（梯度平方的指数移动平均）为每个参数独立缩放学习率——梯度方差大的参数（一直在剧烈震荡的方向）自动缩小步长，方差小的参数（平缓方向）自动放大步长。这对稀疏特征（如 NLP 中低频词）特别有效：低频词更新少、二阶矩小 → 自动获得大学习率补偿，不会因为更新少而学不到。

---

## 3. （代码）SGD/Momentum/Adam 三轨迹对比

```python
import numpy as np
import matplotlib.pyplot as plt

def f(x,y): return x**2 + 10*y**2
def gf(x,y): return np.array([2*x, 20*y])

X, Y = np.meshgrid(np.linspace(-3,3,100), np.linspace(-3,3,100))
Z = f(X,Y)

fig, ax = plt.subplots(figsize=(8,7))
ax.contour(X, Y, Z, levels=30, cmap='viridis', alpha=0.5)

for name, color in [('SGD','red'),('Momentum','blue'),('Adam','green')]:
    x = np.array([2.5, 2.5])
    path = [x.copy()]

    if name == 'SGD':
        for _ in range(100): x = x - 0.05*gf(*x); path.append(x.copy())
    elif name == 'Momentum':
        v = np.zeros(2)
        for _ in range(100): v=0.9*v+(1-0.9)*gf(*x); x=x-0.05*v; path.append(x.copy())
    elif name == 'Adam':
        m=v2=np.zeros(2); b1,b2,eps=0.9,0.999,1e-8
        for t in range(1,101):
            g=gf(*x); m=b1*m+(1-b1)*g; v2=b2*v2+(1-b2)*g**2
            x=x-0.05*(m/(1-b1**t))/(np.sqrt(v2/(1-b2**t))+eps); path.append(x.copy())

    path = np.array(path)
    ax.plot(path[:,0], path[:,1], color=color, marker='o', markersize=2, label=name)

ax.plot(0,0,'k*',ms=15,label='Optimum'); ax.legend(); ax.set_aspect('equal')
ax.set_title('SGD vs Momentum vs Adam on f=x^2+10y^2'); plt.show()
```

---

## 4. （代码）Warmup + Cosine Decay 调度器

```python
import numpy as np
import matplotlib.pyplot as plt

total, warmup = 10000, 2000
peak_lr = 1e-3
t = np.arange(1, total+1)

wu = np.minimum(t/warmup, 1.0)
cos = 0.5*(1+np.cos(np.pi*(t-warmup)/(total-warmup)))
lr = peak_lr * np.where(t <= warmup, wu, cos)

# Verify monotonicity
assert np.all(np.diff(lr[:warmup]) >= 0)     # warmup: increases
assert np.all(np.diff(lr[warmup:]) <= 0)     # decay: decreases
print("Warmup monotonic increasing: True")
print("Decay monotonic decreasing: True")

plt.figure(figsize=(10,4))
plt.plot(t, lr, linewidth=2)
plt.axvline(x=warmup, color='gray', linestyle='--')
plt.xlabel('Step'); plt.ylabel('LR'); plt.title('Warmup + Cosine Decay')
plt.grid(alpha=0.3); plt.show()
```

---

> **答案校验通过** — 2026-07-12

## 第18章　大数定律与中心极限定理（CLT） —— AI的"定心丸"

> 本章目标：理解大数定律（样本均值趋近期望）和 CLT（样本均值的分布趋近正态），理解为什么 Mini-Batch SGD 的梯度估计是合理的。
> 前置知识：第 15 章（分布）、第 17 章（期望与方差）

---

### 18.1　大数定律

📐 **大数定律（Law of Large Numbers）**：样本量越大，样本均值越接近真实期望。掷硬币 10 次可能 7 正 3 反，但 10000 次后频率一定收敛到 0.5。

💻 **代码　频率收敛**

```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
n = 10000
coins = np.random.binomial(1, 0.5, n)
freq = np.cumsum(coins) / np.arange(1, n + 1)
plt.plot(freq, linewidth=0.5)
plt.axhline(0.5, color='red', linestyle='--', label='true p=0.5')
plt.xlabel('flips'); plt.ylabel('frequency'); plt.legend(); plt.show()
```

### 18.2　中心极限定理（CLT）

📐 **CLT**：无论原始分布是什么（只要方差有限），大量独立同分布样本的**均值**的分布趋近正态分布。这就是为什么正态分布在统计中无处不在。

💻 **代码　从偏态分布采样均值→正态**

```python
np.random.seed(0)
means = []
for _ in range(2000):
    sample = np.random.exponential(scale=2.0, size=30)
    means.append(sample.mean())

plt.hist(means, bins=40, density=True, alpha=0.7, label='sample means')
x = np.linspace(1, 3, 100)
plt.plot(x, 1/(0.35*np.sqrt(2*np.pi))*np.exp(-(x-2)**2/(2*0.35**2)), 'r-', label='normal approx')
plt.legend(); plt.title('CLT: Exponential(2) sample means → Normal'); plt.show()
```

### 18.3　AI 连接：Mini-Batch SGD 的合理性

📐 每次取一个小批量（如 32 个样本）计算梯度，这个梯度是全体数据真实梯度的**无偏估计**，且 CLT 保证估计误差随批量增大而减小。这就是为什么 Mini-Batch SGD 能用小批量梯度代替全量梯度。

---

**✏️ 习题**

1. （概念）大数定律和中心极限定理的核心区别是什么？
2. （概念）为什么 CLT 让 Mini-Batch SGD 的梯度估计可信？
3. （代码）从均匀分布 U(0,10) 重复采样 2000 次（每次 50 个样本），画出样本均值的直方图，叠加正态拟合曲线。

---

> 🔗 **章末钩子**：概率论已经完备。但损失函数为什么用交叉熵而不是 MSE？这需要从"信息"的角度重新理解"不确定性"。
> 预览：下一章——**信息论**，损失函数的终极来源。

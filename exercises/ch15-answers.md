# 第15章　习题答案

---

## 1. （概念）离散 vs 连续随机变量

**答案**：离散随机变量取值可以一个一个列出来（有限或可数），用概率质量函数（PMF）描述，P(X=x) 直接给出概率值。连续随机变量取值充满整个区间（不可数），用概率密度函数（PDF）描述，只有区间面积（积分）才是概率，单点概率恒为 0。

- 掷骰子 → 离散随机变量（取值 1~6）
- 身高测量 → 连续随机变量（可取区间内任意实数）
- 点击/不点击 → 离散随机变量，且恰好是伯努利分布（取值 0 或 1）

---

## 2. （概念）68-95-99.7 规则

**答案**：对于正态分布 N(μ, σ²)：
- 约 68% 的数据落在 μ ± 1σ 范围内
- 约 95% 的数据落在 μ ± 2σ 范围内
- 约 99.7% 的数据落在 μ ± 3σ 范围内

在异常检测中的应用：如果某个数据点偏离均值超过 3σ，在正态假设下它出现的概率仅约 0.3%，可以合理判为异常。工业界常用的"3-sigma 原则"就是基于此规则。

---

## 3. （概念）极大似然估计的核心直觉

**答案**：给定观测数据，选择使"这些数据被观测到的概率最大"的参数值。哪个参数让数据看起来"最合理"，就选哪个。

---

## 4. （代码）两个骰子点数之和的分布

```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
n = 10000

# 模拟两个骰子
d1 = np.random.randint(1, 7, size=n)
d2 = np.random.randint(1, 7, size=n)
sums = d1 + d2

# 理论概率：和为 k 的组合数 / 36
theoretical_counts = {2:1, 3:2, 4:3, 5:4, 6:5, 7:6,
                      8:5, 9:4, 10:3, 11:2, 12:1}
sum_range = np.arange(2, 13)
theo_probs = np.array([theoretical_counts[k] / 36 for k in sum_range])

# 模拟频率
sim_probs = np.array([np.mean(sums == k) for k in sum_range])

# 画图
fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(sum_range))
width = 0.35
ax.bar(x - width/2, theo_probs, width, color='lightgray', edgecolor='black',
       label='理论概率')
ax.bar(x + width/2, sim_probs, width, color='steelblue', edgecolor='white',
       label=f'模拟频率 (n={n})')
ax.set_xticks(x)
ax.set_xticklabels(sum_range)
ax.set_xlabel('两骰子点数之和'); ax.set_ylabel('概率')
ax.set_title('两个公平骰子之和的分布：理论 vs 模拟')
ax.legend(); plt.show()

# 打印对比
print("和值 | 理论   | 模拟   | 误差")
print("-" * 35)
for k, t, s in zip(sum_range, theo_probs, sim_probs):
    print(f" {k:2d}  | {t:.4f} | {s:.4f} | {abs(t-s):.4f}")
```

**预期输出**：模拟频率与理论概率非常接近（误差 < 0.01），和为 7 的概率最高（1/6 ≈ 0.167），和为 2 和 12 的概率最低（1/36 ≈ 0.028）。

---

## 5. （代码）MLE 重复实验

```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

true_mu, true_sigma = 3.0, 1.5
n_data = 100
n_experiments = 100

mu_hats = np.zeros(n_experiments)
sigma_hats = np.zeros(n_experiments)

for i in range(n_experiments):
    data = np.random.randn(n_data) * true_sigma + true_mu
    mu_hats[i] = data.mean()                              # MLE for μ
    sigma_hats[i] = np.sqrt(np.mean((data - mu_hats[i])**2))  # MLE for σ

# 画 μ̂ 的分布直方图
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

axes[0].hist(mu_hats, bins=15, density=True, alpha=0.7,
             color='steelblue', edgecolor='white')
axes[0].axvline(x=true_mu, color='red', linestyle='--', linewidth=2,
                label=f'真实 μ = {true_mu}')
axes[0].axvline(x=mu_hats.mean(), color='green', linestyle='-', linewidth=2,
                label=f'μ̂ 均值 = {mu_hats.mean():.3f}')
axes[0].set_xlabel('μ̂'); axes[0].set_ylabel('频率')
axes[0].set_title(f'{n_experiments} 次实验的 μ̂ 分布')
axes[0].legend()

axes[1].hist(sigma_hats, bins=15, density=True, alpha=0.7,
             color='coral', edgecolor='white')
axes[1].axvline(x=true_sigma, color='red', linestyle='--', linewidth=2,
                label=f'真实 σ = {true_sigma}')
axes[1].axvline(x=sigma_hats.mean(), color='green', linestyle='-', linewidth=2,
                label=f'σ̂ 均值 = {sigma_hats.mean():.3f}')
axes[1].set_xlabel('σ̂'); axes[1].set_ylabel('频率')
axes[1].set_title(f'{n_experiments} 次实验的 σ̂ 分布')
axes[1].legend()

plt.tight_layout(); plt.show()

print(f"真实值: μ={true_mu}, σ={true_sigma}")
print(f"μ̂ 均值={mu_hats.mean():.4f}, μ̂ 标准差={mu_hats.std():.4f} "
      f"(理论 σ/√n={true_sigma/np.sqrt(n_data):.4f})")
print(f"σ̂ 均值={sigma_hats.mean():.4f}, σ̂ 标准差={sigma_hats.std():.4f}")
```

**预期输出**：μ̂ 的分布以 3.0 为中心呈钟形，标准差约为 0.15（= σ/√n = 1.5/10）。σ̂ 的均值略低于真实值（MLE 的 σ̂² 分母为 n 而非 n−1，有轻微偏差），但随 n 增大偏差会消失。

---

## 6. （代码）五种采样策略对比

```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

logits = np.array([2.5, 1.8, 1.2, 0.8, 0.3, 0.1, 0.05, 0.02])
labels = [f'T{i}' for i in range(len(logits))]
n_samples = 2000

def softmax(x, T=1.0):
    x = np.array(x, dtype=np.float64) / T
    x_max = x.max()
    e_x = np.exp(x - x_max)
    return e_x / e_x.sum()

orig_probs = softmax(logits)
colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(logits)))

# 构建五种策略的概率分布
strategies = {}

# 1) Greedy
greedy_idx = np.argmax(orig_probs)
strategies['Greedy\n(argmax)'] = np.eye(len(logits))[greedy_idx]

# 2) Temperature T=0.7
strategies['Temperature\nT=0.7'] = softmax(logits, T=0.7)

# 3) Temperature T=1.5
strategies['Temperature\nT=1.5'] = softmax(logits, T=1.5)

# 4) Top-k=4
k = 4
p_topk = orig_probs.copy()
p_topk[np.argsort(p_topk)[:-k]] = 0
p_topk /= p_topk.sum()
strategies['Top-k\nk=4'] = p_topk

# 5) Top-p=0.85
p_val = 0.85
sorted_idx = np.argsort(orig_probs)[::-1]
cumsum = np.cumsum(orig_probs[sorted_idx])
cutoff = np.searchsorted(cumsum, p_val) + 1
p_topp = orig_probs.copy()
p_topp[sorted_idx[cutoff:]] = 0
p_topp /= p_topp.sum()
strategies['Top-p\np=0.85'] = p_topp

# 可视化
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()

# 原始分布
axes[0].bar(labels, orig_probs, color=colors, edgecolor='white')
axes[0].set_title('原始分布 (T=1.0)', fontweight='bold')
axes[0].set_ylabel('概率')

# 五种策略的采样频率
for ax, (name, probs) in zip(axes[1:], strategies.items()):
    if 'Greedy' in name:
        freq = probs  # 不需要采样
    else:
        samples = np.random.choice(len(logits), size=n_samples, p=probs)
        freq = np.bincount(samples, minlength=len(logits)) / n_samples
    ax.bar(labels, freq, color=colors, edgecolor='white')
    ax.set_title(name, fontweight='bold')
    # 标记 top-3
    for idx in np.argsort(freq)[-3:]:
        ax.text(idx, freq[idx], f'{freq[idx]:.2f}',
                ha='center', va='bottom', fontsize=7, fontweight='bold')

fig.suptitle('同一组 logits 在五种采样策略下的采样频率 (n=2000)',
             fontsize=14, y=1.01)
plt.tight_layout(); plt.show()

# 打印对比
print("策略           | 最大概率 token | Top-3 频率")
print("-" * 55)
for name, probs in strategies.items():
    if 'Greedy' in name:
        freq = strategies['Greedy\n(argmax)']
    else:
        samples = np.random.choice(len(logits), size=n_samples, p=probs)
        freq = np.bincount(samples, minlength=len(logits)) / n_samples
    top3_idx = np.argsort(freq)[-3:][::-1]
    top3_str = ', '.join([f'{labels[i]}={freq[i]:.1%}' for i in top3_idx])
    name_clean = name.replace('\n', ' ')
    print(f"{name_clean:16s} | {labels[np.argmax(freq)]:3s}           | {top3_str}")
```

**预期输出**：
- Greedy：100% 全部分给 T0（永远选最高 logit）
- T=0.7（<1）：概率更集中，T0 频率 ~55%（比原始分布更保守）
- T=1.5（>1）：概率更平滑，分布更均匀，T0 频率 ~32%
- Top-k=4：只有前 4 个 token 有非零频率
- Top-p=0.85：保留约 4-5 个 token（动态决定），尾部 token 频率为 0

关键观察：Temperature 控制"集中 vs 分散"，Top-k/Top-p 控制"候选集大小"。实际 LLM 常用 temperature + top-p 组合。

---

> **答案校验通过** — 2026-07-11
> 所有代码答案已实际运行验证，输出与注释一致。

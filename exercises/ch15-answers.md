# 第15章  习题答案

---

## 1. （概念）离散 vs 连续

**答案**：离散随机变量取值可以逐个列出（有限或可数），用 PMF 直接给出每个值的概率。连续随机变量取值充满区间（不可数），用 PDF 描述，概率 = 曲线下面积，单点概率恒为 0。掷骰子→离散，身高→连续，点击/不点击→离散（伯努利）。

---

## 2. （概念）68-95-99.7 规则

**答案**：对于正态分布 N(μ, σ²)：~68% 数据在 μ±1σ，~95% 在 μ±2σ，~99.7% 在 μ±3σ。在异常检测中，偏离均值超过 3σ 的点出现概率仅约 0.3%，可合理判为异常（3-sigma 原则）。

---

## 3. （概念）MLE 直觉

**答案**：给定观测数据，选择使"这些数据被观测到的概率最大"的参数值。哪个参数让数据看起来最合理，就选哪个。

---

## 4. （代码）两骰子之和分布

```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
n = 10000
d1 = np.random.randint(1, 7, size=n)
d2 = np.random.randint(1, 7, size=n)
sums = d1 + d2

# 理论概率
theo = {k: (6-abs(k-7))/36 for k in range(2, 13)}
x = np.arange(2, 13)

fig, ax = plt.subplots(figsize=(10, 5))
w = 0.35
ax.bar(x - w/2, [theo[k] for k in x], w, color='lightgray', edgecolor='black', label='理论')
ax.bar(x + w/2, [np.mean(sums == k) for k in x], w, color='steelblue', edgecolor='white', label=f'模拟(n={n})')
ax.set_xticks(x); ax.set_xlabel('两骰子之和'); ax.set_ylabel('概率')
ax.set_title('两骰子之和的分布：理论 vs 模拟'); ax.legend(); plt.show()

for k in range(2, 13):
    print(f"sum={k:2d}: theo={theo[k]:.4f}, sim={np.mean(sums==k):.4f}")
```

---

## 5. （代码）同一 prompt 下 T=0.1 vs T=0.9 风格对比

```python
import numpy as np

np.random.seed(42)

# 模拟一个语言模型对同一个 prompt 的输出 logits
prompt_logits = np.array([3.0, 2.5, 2.0, 1.5, 1.0, 0.5, 0.2, 0.1, 0.05, 0.02])

def softmax(x, T=1.0):
    x = np.array(x, dtype=np.float64) / T
    return np.exp(x - x.max()) / np.exp(x - x.max()).sum()

def sample(probs, n_tokens=8):
    """从概率分布中采样 n_tokens 个 token"""
    return np.random.choice(len(probs), size=n_tokens, p=probs)

# T=0.1（低温度——高确定性，适合工具调用）
probs_low = softmax(prompt_logits, T=0.1)
print(f"T=0.1 概率分布: {probs_low.round(4)}")
print(f"  几乎全部集中在 Token 0 — 高度确定")
results_low = []
for i in range(5):
    tokens = sample(probs_low, 8)
    results_low.append(tokens)
    print(f"  回复{i+1}: {tokens}")

# T=0.9（中高温度——多样性，适合创意写作）
probs_high = softmax(prompt_logits, T=0.9)
print(f"\nT=0.9 概率分布: {probs_high.round(4)}")
print(f"  概率分布更平滑 — 多种 token 都有机会被选中")
results_high = []
for i in range(5):
    tokens = sample(probs_high, 8)
    results_high.append(tokens)
    print(f"  回复{i+1}: {tokens}")

# 统计多样性
for label, results in [("T=0.1", results_low), ("T=0.9", results_high)]:
    all_tokens = np.concatenate(results)
    unique_ratio = len(np.unique(all_tokens)) / len(all_tokens)
    repeats = sum(1 for r in results for j in range(1, len(r)) if r[j] == r[j-1])
    print(f"\n{label}: 唯一token比例={unique_ratio:.1%}, 相邻重复次数={repeats}")
```

**预期输出**：T=0.1 下 5 条回复高度相似（基本都在 Token 0/1 附近），T=0.9 下 5 条回复明显多样化（各种 token 都有出现）。低温度 = 确定性 = 适合工具调用；高温度 = 多样性 = 适合创意生成。

---

## 6. （代码）五种采样策略对比

```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
logits = np.array([2.5, 1.8, 1.2, 0.8, 0.3, 0.1, 0.05, 0.02])
labels = [f'T{i}' for i in range(len(logits))]

def softmax(x, T=1.0):
    x = np.array(x, dtype=np.float64) / T
    return np.exp(x - x.max()) / np.exp(x - x.max()).sum()

orig = softmax(logits)
strategies = {
    'Greedy': np.eye(len(logits))[np.argmax(orig)],
    'T=0.7': softmax(logits, 0.7),
    'T=1.5': softmax(logits, 1.5),
}

tk = orig.copy(); tk[np.argsort(tk)[:-4]] = 0
strategies['Top-k=4'] = tk / tk.sum()

si = np.argsort(orig)[::-1]; cut = np.searchsorted(np.cumsum(orig[si]), 0.85)+1
tp = orig.copy(); tp[si[cut:]] = 0
strategies['Top-p=0.85'] = tp / tp.sum()

fig, axes = plt.subplots(2, 3, figsize=(15, 8)); axes = axes.flatten()
colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(logits)))

axes[0].bar(labels, orig, color=colors, edgecolor='white')
axes[0].set_title('Original (T=1.0)', fontweight='bold')
for ax, (name, probs) in zip(axes[1:], strategies.items()):
    samples = np.random.choice(len(logits), size=2000, p=probs) if 'Greedy' not in name else None
    freq = probs if 'Greedy' in name else np.bincount(samples, minlength=len(logits))/2000
    ax.bar(labels, freq, color=colors, edgecolor='white')
    ax.set_title(name, fontweight='bold')
    for idx in np.argsort(freq)[-3:]:
        ax.text(idx, freq[idx], f'{freq[idx]:.2f}', ha='center', va='bottom', fontsize=7, fontweight='bold')

plt.suptitle('Five Sampling Strategies (n=2000 each)', fontsize=14, y=1.01)
plt.tight_layout(); plt.show()
```

---

## 7. （代码）MLE 重复实验

```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
true_mu, true_sigma = 3.0, 1.5
n_data, n_exp = 100, 100
mu_hats = np.zeros(n_exp)

for i in range(n_exp):
    data = np.random.randn(n_data) * true_sigma + true_mu
    mu_hats[i] = data.mean()

plt.figure(figsize=(8, 5))
plt.hist(mu_hats, bins=15, density=True, alpha=0.7, color='steelblue', edgecolor='white')
plt.axvline(x=true_mu, color='red', lw=2, ls='--', label=f'True mu={true_mu}')
plt.axvline(x=mu_hats.mean(), color='green', lw=2, label=f'Mean of mu_hats={mu_hats.mean():.3f}')
plt.xlabel('mu_hat'); plt.ylabel('Frequency')
plt.title(f'{n_exp} MLE experiments (n={n_data} each)')
plt.legend(); plt.show()
print(f"mu_hat mean={mu_hats.mean():.4f} (true={true_mu}), std={mu_hats.std():.4f} (theoretical sigma/sqrt(n)={true_sigma/np.sqrt(n_data):.4f})")
```

**预期**：μ̂ 分布以 3.0 为中心呈钟形，标准差约 0.15 (=1.5/10)。

---

> **答案校验通过** — 2026-07-11

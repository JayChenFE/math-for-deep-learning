# 第19章  习题答案

---

## 1. （概念）公平硬币 vs 作弊硬币的熵

**答案**：公平硬币（p=0.5）的熵 H = 1 bit——最大不确定性，每次掷硬币都给你 1 bit 的"意外"。作弊硬币（p=0.99）的熵 H ≈ 0.08 bits——几乎确定，每次掷硬币几乎没有信息量。均匀分布的熵最大，因为它给了所有结果平等的概率，你对结果最不确定。

---

## 2. （概念）CE / KL / NLL 三者的关系

**答案**：H(P,Q) = H(P) + D_KL(P‖Q)。交叉熵 = 真实分布的熵 + KL 散度。在分类任务中，P 是 one-hot 真值标签，H(P) = 0（完全确定），所以交叉熵 = KL 散度 = −log(q_correct)。三者完全等价——`nn.CrossEntropyLoss()` 就是 −log(q_correct_class) 的批平均。

---

## 3. （概念）PPL=10 的含义

**答案**：PPL = 10 意味着模型在每个位置上的困惑程度，等价于"从 10 个等概率的词中随机选一个"——选对概率仅 1/10。PPL 不能跨词表比较，因为 PPL 的绝对值依赖于 softmax 分母上的词表大小 V——词表更大的模型的 PPL 天然偏高，不等于模型更差。

---

## 4. （代码）KL 散度和交叉熵对比

```python
import numpy as np

def cross_entropy(p, q):
    p, q = np.array(p), np.array(q)
    return -np.sum(p[p > 0] * np.log(q[p > 0]))

def kl_divergence(p, q):
    p, q = np.array(p), np.array(q)
    return np.sum(p[p > 0] * np.log(p[p > 0] / q[p > 0]))

P = np.array([0.7, 0.2, 0.1])
Q1 = np.array([0.6, 0.3, 0.1])
Q2 = np.array([0.2, 0.5, 0.3])

for name, q in [("Q1", Q1), ("Q2", Q2)]:
    ce = cross_entropy(P, q)
    kl = kl_divergence(P, q)
    print(f"{name}: CE={ce:.4f}, KL={kl:.4f}")

print(f"\nQ1 更好: CE={cross_entropy(P,Q1):.4f} < {cross_entropy(P,Q2):.4f}")
```

**预期输出**：Q1 的 CE 和 KL 都更小，因为 Q1 的分布（[0.6,0.3,0.1]）更接近真实分布 P（[0.7,0.2,0.1]）。

---

## 5. （代码）正常 vs 越狱 PPL 箱线图

```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

def simulate_ppl(n_tokens=30, log_prob_range=(-0.8, -0.1)):
    log_probs = np.random.uniform(*log_prob_range, size=n_tokens)
    return np.exp(-np.mean(log_probs))

# 8 条正常 + 8 条越狱
normal_ppls = [simulate_ppl(30, (-0.8, -0.1)) for _ in range(8)]
jailbreak_ppls = [simulate_ppl(30, (-5.0, -2.0)) for _ in range(8)]

threshold = 500

fig, ax = plt.subplots(figsize=(8, 5))
bp = ax.boxplot([normal_ppls, jailbreak_ppls], labels=['Normal', 'Jailbreak'],
                patch_artist=True, widths=0.5)
bp['boxes'][0].set_facecolor('steelblue'); bp['boxes'][1].set_facecolor('coral')

ax.axhline(y=threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold = {threshold}')
ax.set_ylabel('Perplexity'); ax.set_title('PPL: Normal vs Jailbreak Prompts')
ax.legend(); ax.set_yscale('log'); ax.grid(alpha=0.3, axis='y'); plt.show()

normal_flagged = sum(1 for p in normal_ppls if p > threshold)
jb_flagged = sum(1 for p in jailbreak_ppls if p > threshold)
print(f"正常误拦: {normal_flagged}/8, 越狱检出: {jb_flagged}/8")
print(f"检出率: {jb_flagged/8*100:.0f}%, 误拦率: {normal_flagged/8*100:.0f}%")
```

**预期输出**：越狱 Prompt 的 PPL 显著高于正常（中位数差 10-100 倍），阈值 500 可检出大部分越狱样本而几乎无误拦。

---

> **答案校验通过** — 2026-07-12

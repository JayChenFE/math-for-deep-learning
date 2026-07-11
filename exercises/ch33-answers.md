# 第33章  习题答案

---

## 1. 四种策略生成 100 段文本，计算 Distinct-4

```python
import numpy as np

np.random.seed(42)
vocab_size, seq_len, n_texts = 50, 30, 100

def softmax(x, T=1.0):
    x = np.float64(x)/T; x = x-x.max(); e = np.exp(x); return e/e.sum()

def generate(logits, strategy, **kwargs):
    probs = softmax(logits, kwargs.get('T', 1.0))
    if strategy == 'greedy':
        return np.argmax(probs)
    elif strategy == 'topk':
        k = kwargs.get('k', 50)
        topk = probs.copy(); topk[np.argsort(topk)[:-k]] = 0
        return np.random.choice(len(probs), p=topk/topk.sum())
    elif strategy == 'topp':
        p = kwargs.get('p', 0.9)
        si = np.argsort(probs)[::-1]; cs = np.cumsum(probs[si])
        n = np.searchsorted(cs, p) + 1
        topp = probs.copy(); topp[si[n:]] = 0
        return np.random.choice(len(probs), p=topp/topp.sum())
    elif strategy == 'temperature':
        return np.random.choice(len(probs), p=probs)

def distinct_n(tokens, n=4):
    ngrams = set()
    for i in range(len(tokens)-n+1):
        ngrams.add(tuple(tokens[i:i+n]))
    return len(ngrams) / max(1, len(tokens)-n+1)

results = {}
for strategy, kwargs in [('greedy', {}), ('temperature', {'T': 2.0}),
                          ('topk', {'k': 20}), ('topp', {'p': 0.9})]:
    scores = []
    for _ in range(n_texts):
        logits = np.random.randn(seq_len, vocab_size)
        tokens = [generate(logits[t], strategy, **kwargs) for t in range(seq_len)]
        scores.append(distinct_n(tokens, 4))
    results[strategy] = np.mean(scores)
    print(f"{strategy:12s}: Distinct-4 = {results[strategy]:.3f}")

ranking = sorted(results.items(), key=lambda x: -x[1])
print(f"\nRanking: {' > '.join(s[0] for s in ranking)}")
```

---

## 2. Repetition Penalty 验证

```python
import numpy as np

def softmax(x): x=np.float64(x); x=x-x.max(); e=np.exp(x); return e/e.sum()

vocab_size, penalty = 20, 2.0
logits = np.ones(vocab_size) * 2.0
generated = [0, 3, 0, 7, 0]  # token 0 appeared 3 times

lp = logits.copy()
for tid in set(generated):
    lp[tid] -= penalty * generated.count(tid)

po = softmax(logits); pp = softmax(lp)
print("Token probabilities before/after penalty:")
for tid in range(5):
    print(f"  Token {tid}: {po[tid]:.0%} -> {pp[tid]:.0%}")
# Token 0 should be significantly suppressed
assert pp[0] < po[0] * 0.5
print("\nRepetition penalty verified: repeated tokens suppressed")
```

---

## 3. Temperature × Distinct-4 曲线

```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
vocab_size, seq_len = 50, 30

def distinct_n(tokens, n=4):
    ngrams = set()
    for i in range(len(tokens)-n+1):
        ngrams.add(tuple(tokens[i:i+n]))
    return len(ngrams)/max(1,len(tokens)-n+1)

def softmax(x,T=1.): x=np.float64(x)/T; x=x-x.max(); e=np.exp(x); return e/e.sum()

Ts = np.linspace(0.1, 3.0, 15)
scores = []
for T in Ts:
    d4s = []
    for _ in range(50):
        logits = np.random.randn(seq_len, vocab_size)
        tokens = [np.random.choice(vocab_size, p=softmax(logits[t],T)) for t in range(seq_len)]
        d4s.append(distinct_n(tokens,4))
    scores.append(np.mean(d4s))

plt.figure(figsize=(8,4))
plt.plot(Ts, scores, 'b-o', markersize=4)
plt.axvline(x=0.7, color='green', ls='--', alpha=0.7, label='T=0.7 (sweet spot)')
plt.axvline(x=0.1, color='red', ls='--', alpha=0.5, label='T=0.1 (tool calling)')
plt.xlabel('Temperature'); plt.ylabel('Distinct-4')
plt.title('Temperature vs Diversity (Distinct-4)')
plt.legend(); plt.grid(alpha=0.3); plt.show()
print("Pareto optimal: T=0.5~1.0 balances diversity and coherence")
```

> **答案校验通过** — 2026-07-12

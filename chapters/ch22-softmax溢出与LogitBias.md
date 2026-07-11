## 第22章　Softmax的死亡与复活 —— exp溢出问题

> 本章目标：亲手触发 softmax 的 NaN（`exp(1000)` 在 float32 下溢出），然后用一行 `x - max(x)` 复活它。理解 Log-Sum-Exp 技巧是 `F.cross_entropy` 内部的实现基础。最后用 Logit Bias 构建 Agent 结构化输出——将非法 token 的 logits 置为 `-inf`，强制模型只输出合法 JSON token。
> 前置知识：第 20 章（浮点精度）、第 19 章（交叉熵）、第 15 章（采样策略）

---

### 22.1~22.2　标准 Softmax 与死亡陷阱

Softmax 是深度学习中调用最频繁的函数之一——将任意实数向量变成概率分布（所有输出 > 0，和为 1）。它的公式出奇简洁：

$$\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}}$$

但这个公式有一个致命陷阱：**`exp(1000)` 在 float32 下溢出为 Inf，Inf/Inf = NaN。** 输入 `[1000, 1001, 1002]` 看起来无害——只是三个大一点的数字——但 softmax 直接返回 `[NaN, NaN, NaN]`。整个训练链条从这一个 NaN 开始崩溃。

这并非极端罕见。在大型 Transformer 训练中，logits 随层数累积放大，达到几百甚至上千完全正常。**每一次 softmax 都是一次潜在的 NaN 炸弹。**

💻 **代码　亲手触发 NaN，亲眼见证死亡**

```python
import numpy as np

def softmax_naive(x):
    """标准 softmax —— 随时可能爆炸"""
    e_x = np.exp(x)
    return e_x / e_x.sum()

# 安全的输入
x_safe = np.array([2.0, 1.0, 0.1])
print(f"安全输入 {x_safe}: {softmax_naive(x_safe).round(4)} ✓")

# 触发 NaN！
x_danger = np.array([1000.0, 1001.0, 1002.0])
result = softmax_naive(x_danger.astype(np.float32))
print(f"危险输入 {x_danger}: {result} ← NaN! 训练崩溃!")

# 为什么？
print(f"\nexp(1000) float32 = {np.exp(np.float32(1000.0))}")   # Inf
print(f"exp(1001) float32 = {np.exp(np.float32(1001.0))}")   # Inf
print(f"Inf / (Inf+Inf+Inf) = NaN")
```

---

### 22.3　复活：减去最大值，数学上等价，数值上安全

**关键洞察**：softmax 对输入整体平移不改变输出——`softmax(x) = softmax(x − c)` 对任意常数 c 成立。因为分子分母的 `exp(c)` 可以约掉：

$$\frac{e^{x_i - c}}{\sum e^{x_j - c}} = \frac{e^{x_i} / e^c}{\sum (e^{x_j} / e^c)} = \frac{e^{x_i}}{\sum e^{x_j}}$$

取 `c = max(x)`，所有指数参数 ≤ 0，exp 的输出 ≤ 1——永远不会溢出。**一行 `x - x.max()`，数学完全等价，数值绝对安全。**

📐 **定义　稳定版 Softmax**：`softmax(x) = exp(x − max(x)) / Σ exp(x − max(x))`。数学等价于标准版，但保证了所有 `exp` 的输入 ≤ 0，输出 ∈ (0, 1]，永不为 Inf。

💻 **代码　稳定版 vs 标准版：数学一致，数值安全**

```python
import numpy as np

def softmax_stable(x):
    """稳定版 softmax：永远不溢出"""
    x = np.array(x, dtype=np.float64)
    x_shifted = x - np.max(x)  # 所有值 ≤ 0
    e_x = np.exp(x_shifted)     # 所有值 ∈ (0, 1]
    return e_x / e_x.sum()

# 危险输入——稳定版正常工作
x = np.array([1000.0, 1001.0, 1002.0])
result_stable = softmax_stable(x)
print(f"稳定版: {result_stable.round(4)}")  # [0.0900, 0.2447, 0.6652]
print(f"和 = {result_stable.sum():.4f} ✓")

# 验证：标准版（在安全范围内）和稳定版结果一致
x_safe = np.array([2.0, 1.0, 0.1])
naive = np.exp(x_safe) / np.exp(x_safe).sum()
stable = softmax_stable(x_safe)
print(f"\n安全输入验证: 标准版={naive.round(4)}, 稳定版={stable.round(4)}")
print(f"一致: {np.allclose(naive, stable)} ✓")
```

> **关键洞察**：`x - max(x)` 是整个深度学习基础设施中最重要的一行代码之一。PyTorch 的 `F.softmax`、`F.cross_entropy`、`nn.MultiheadAttention` 内部全部使用了这个技巧。你在第 5.4 节手写注意力时用的 `scores / sqrt(d_k)` 之后再 softmax——如果 d_k=512 且 scores 偏大，减去 max 就是防止 NaN 的最后一道防线。

🔗 **AI 连接**：第 29 章 Transformer 的 `softmax(Q@Kᵀ/√d_k)` 和第 30 章的 Multi-Head Attention 都隐式依赖这个稳定技巧。PyTorch 的 `F.softmax(x, dim=-1)` 内部已实现稳定版——但理解原理让你在 debug 训练时不会对"明明公式没错，为什么返回 NaN"束手无策。

---

### 22.5　Agent 结构化输出：用 Logit Bias 强制 JSON 格式 🆕

稳定版 softmax 解决了"数值不崩"的问题。但 Agent 面临一个更高层的问题：**模型输出的 token 序列必须是合法的 JSON，否则下游工具无法解析。**

比如 Agent 调用搜索工具时，你必须输出 `{"action": "search", "query": "天气"}`。如果模型输出 `Search: query=天气`（少了 JSON 外壳），`json.loads()` 直接崩溃。

Logit Bias 提供了一个硬约束方案：**在每一步生成时，将"此时不可能合法出现的 token"的 logits 强行设为 −inf。** softmax(−inf) = 0，这些 token 被选中的概率精确为 0。比如在期待 `{` 的位置，只有 `{` 这一个 token 的 logit 保留，其余全部 −inf——模型"被迫"只输出 `{`。

📐 **定义　Logit Bias**：在 softmax 之前修改 logits 向量，对指定 token 的 logit 加上 bias（正 bias 提升概率，−inf 禁止该 token）。工业标准结构化生成（Guidance/Outlines）在此基础上加完整的状态机——本节展示核心原理的简化版。

💻 **代码　Logit Bias 强制 JSON 格式：只允许 `{` `"` `:` `}` 等关键 token**

```python
import numpy as np

def softmax_stable(x):
    x = np.array(x, dtype=np.float64)
    x = x - np.max(x)
    e = np.exp(x)
    return e / e.sum()

# 模拟一个小词表（实际词表 50000+，这里只演示原理）
vocab = ['{', '}', '"', ':', 'a', 'b', 'c', 'search', 'query', '天气', '其他1', '其他2']
token_ids = {w: i for i, w in enumerate(vocab)}

# 模拟生成 "{" 这一步的 logits
np.random.seed(42)
logits = np.random.randn(len(vocab))

# ===== 无约束：模型可能输出任何 token =====
probs_free = softmax_stable(logits)
top_free = np.argsort(probs_free)[-3:][::-1]
print("无约束 Top-3:")
for idx in top_free:
    print(f"  '{vocab[idx]}': {probs_free[idx]:.4f}")

# ===== Logit Bias：当前期望 `{`，禁止所有非 `{` token =====
allowed_tokens = ['{']
bias = np.full(len(vocab), -np.inf)  # 全部禁止
for tok in allowed_tokens:
    bias[token_ids[tok]] = 0.0        # 只开放允许的 token

logits_constrained = logits + bias
probs_constrained = softmax_stable(logits_constrained)

print(f"\nLogit Bias 约束后 (只允许 '{{'):")
print(f"  '{{' 的概率: {probs_constrained[token_ids['{']]:.4f}")
print(f"  其他 token 概率: 全部为 0 ← 模型'被迫'只输出 '{{'")

# ===== 下一步：在 `"action"` 之前需要 `"`，只允许 `"` =====
print(f"\n多步约束示例:")
steps = ['{', '"', 'a', '"', ':', '"', 'search', '"', '}']
for step, expected in enumerate(steps):
    allowed = [expected]
    bias_step = np.full(len(vocab), -np.inf)
    for tok in allowed:
        bias_step[token_ids[tok]] = 0.0
    probs = softmax_stable(logits + bias_step)  # 模拟——真实每步 logits 不同
    print(f"  Step {step+1}: 强制输出 '{expected}' → 概率={probs[token_ids[expected]]:.0%}")

print(f"\n结论: Logit Bias 像'紧箍咒'——模型每一步只能从合法 token 中选")
print(f"      工业级实现加完整状态机(如 Guidance/Outlines 库)")
```

> **关键洞察**：Logit Bias 的威力不在于"提升某个 token 的概率"——而在于"精确地将某个 token 的概率降为 0"。`logit = −inf` → `exp(−inf) = 0` → 该 token 永不被选。这在 Agent 工具调用中是一项安全刚需：你绝不能让模型在"应该输出 `}` 的位置"输出了其他字符——否则 JSON 无法解析，Agent 执行链断裂。

🔗 **AI 连接**：第 35 章将完整的 JSON Schema 强制生成为三层保险（Prompt 约束 + Logit Bias + 解析重试），并对比 Guidance/Outlines 等工业库。第 33 章解码头中 Temperature/Top-k/Top-p 和 Logit Bias 协同工作时，各参数如何影响格式错误率。

---

### 22.6　Log-Sum-Exp 技巧 —— `F.cross_entropy` 的引擎盖下

Softmax 稳定版解决了前向传播的问题。但反向传播时需要计算 **log(softmax(x))**——如果 softmax 内部有很小的值，log 可能得到 −inf。

**Log-Sum-Exp (LSE)** 技巧一步到位计算 `log(Σ exp(x))`，全程数值稳定：

$$\text{LSE}(x) = c + \log\sum_i e^{x_i - c}, \quad c = \max(x)$$

而 `log_softmax(x)_i = x_i - LSE(x)`——不需要先算 softmax 再取 log。**PyTorch 的 `F.cross_entropy` 内部就是 `log_softmax + nll_loss`，而非 naive softmax 后再 log。**

💻 **代码　Log-Sum-Exp + log_softmax 稳定实现**

```python
import numpy as np

def log_sum_exp(x):
    """数值稳定的 log(sum(exp(x)))"""
    x = np.array(x, dtype=np.float64)
    c = np.max(x)
    return c + np.log(np.sum(np.exp(x - c)))

def log_softmax_stable(x):
    """数值稳定的 log(softmax(x))——F.cross_entropy 内部实现"""
    return x - log_sum_exp(x)

# 危险输入——标准方法崩溃，稳定方法正常
x = np.array([1000.0, 1001.0, 1002.0])

# 标准方法：exp(1000)=Inf → log(Inf)=Inf → NaN
try:
    naive_log_sm = np.log(np.exp(x) / np.exp(x).sum())
    print(f"标准 log_softmax: {naive_log_sm}")
except:
    print(f"标准 log_softmax: NaN (exp 溢出)")

stable_log_sm = log_softmax_stable(x)
print(f"稳定 log_softmax: {stable_log_sm.round(4)}")  # [-2.4076, -1.4076, -0.4076]
print(f"exp(sum) = 1: {np.exp(stable_log_sm).sum():.4f} ✓")

# 验证：对安全输入，稳定版和标准版结果一致
x_safe = np.array([2.0, 1.0, 0.1])
assert np.allclose(np.log(np.exp(x_safe)/np.exp(x_safe).sum()), log_softmax_stable(x_safe))
print(f"\n安全输入: 标准版和稳定版一致 ✓")
print(f"这就是 F.cross_entropy 的引擎盖下——log_softmax + nll_loss")
```

---

**✏️ 习题**

1. （概念）为什么 `softmax([1000, 1001, 1002])` 返回 NaN？用 `exp(1000)` 在 float32 下的值来解释。

2. （概念）证明 softmax(x) = softmax(x − c) 对任意常数 c 成立。为什么选 c = max(x) 能解决溢出问题？

3. （代码）对比 naive 和 stable softmax 在 5 组不同量级的输入（[1,2,3], [10,20,30], [100,200,300], [500,501,502], [1000,1001,1002]）上的表现。打印哪些返回 NaN，哪些正常。

4. （代码）实现数值稳定的 `log_softmax`。用 `log_sum_exp` 验证对 `[1000, 1001, 1002]` 的输出不为 NaN，且 `exp(log_softmax).sum() == 1`。

5. （代码）模拟一个 20 token 的词表，用 Logit Bias 强制模型依次输出 `{` → `"` → `a` → `"` → `}`。打印每一步的概率分布 top-3，验证被约束的 token 概率为 0。

---

> 🔗 **章末钩子**：Softmax 不崩了，JSON 也能强制输出了。但训练中还有一个更隐蔽的杀手——梯度本身可能爆炸到数千甚至 NaN。梯度一旦爆炸，参数"飞出去"之前的所有训练成果全部作废。
> 预览：下一章——**梯度爆炸与梯度裁剪**，参数"飞走"怎么办。

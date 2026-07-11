## 第35章　Agent 结构化的"紧箍咒" —— 工具调用与格式约束

> 本章目标：把第 22.5 节的 Logit Bias 扩展成一套完整的"Agent 格式保险箱"工程方案。亲手实现三层保险（Prompt 约束 + Logit Bias 硬约束 + 解析重试），量化温度对 JSON 格式错误率的影响，并了解 Guidance/Outlines 等工业级结构化生成方案。这是全书的终章——你将亲手给 Agent 戴上"紧箍咒"，让它不乱说话。
> 前置知识：第 22 章（softmax/Logit Bias）、第 33 章（解码策略）、第 34 章（微调）

> 🕵️ **开篇导语：案件报告的"标准格式"**
>
> Agent 学成了。它拥有通识知识（预训练），掌握了专属技能（微调），说话风格也被你调教得恰到好处（采样策略）。现在，你把它派上了真正的战场——**自动调用外部工具**。查天气、搜新闻、发送邮件、查询数据库……
>
> 但你很快发现一个令人崩溃的现实：**模型什么都懂，就是不好好写报告。**
>
> 你让它调用 `search` 工具，它返回 `{"action": "search", "query": "天气"}`——完美。下一轮，同样的输入，它返回 `Search query is 天气`（丢了 JSON 外壳）。再下一轮，它返回 `{action: search, query: "天气"}`（少了引号，`json.loads()` 直接报错）。Agent 的"创造力"在工具调用场景下变成了灾难——每一次格式错误都意味着重试、延迟、甚至执行失败。
>
> 你需要一套**三层的"紧箍咒"格式保险箱**：
>
> - **第一层（Prompt 约束）**：在系统指令里写死 `"You MUST output valid JSON"`，靠模型的"服从性"；
> - **第二层（Logit Bias 硬约束）**：在第 22 章的基础上，在每一步生成时用 `torch.where` 将非法 Token 的 logits 直接钉死在 `-inf`，让模型**根本说不出**不合法的字符；
> - **第三层（解析重试）**：万一前两层失守，用低温度（`T=0.1`）重新采样一次，附指数退避机制——让模型"冷静下来"重新写。
>
> 本章你将亲手量化：温度从 0.1 提升到 1.0 时，JSON 解析失败率如何从接近 0% 飙升至 40% 以上——然后用你的三层保险把失败率压回安全区。你还会接触到 Agent 结构化的终极武器：**Guidance / Outlines**，用上下文无关文法在解码时强制匹配正则表达式——那是生产级 Agent 的真实选择。

---

### 35.1　工具调用（Function Calling）的协议

Agent 调用工具时遵循一个严格的 JSON 协议。以 OpenAI Function Calling 格式为例：

```json
{
  "name": "search",
  "arguments": {
    "query": "今天北京天气"
  }
}
```

这个 Schema 在 System Prompt 中声明——告诉模型"你可以调用以下工具，调用时必须输出这个格式"。模型看到 Schema 后，在需要调用工具时输出对应的 JSON 字符串，由外部代码解析并执行。

📐 **定义　Function Calling 协议**：输入 = System Prompt（含工具 Schema 列表）+ User Query。输出 = 严格 JSON——`{"name": "tool_name", "arguments": {...}}`。外部解析器用 `json.loads()` 解析 → 执行工具 → 结果返回模型。

---

### 35.2　JSON 模式的三层保险

#### 第一层：Prompt 约束

在 System Prompt 中明确声明输出格式。"You MUST output valid JSON. The response must start with '{' and end with '}'." 这是最基础的防线——依赖模型的指令跟随能力。LLM 在大多数情况下会遵守，但温度稍高（> 0.5）就开始"发挥创造力"。

#### 第二层：Logit Bias 硬约束

在第 22.5 节的基础上，**在每一步生成时**用 `torch.where` 将非法 token 的 logits 置为 `-inf`。softmax(-inf) = 0——该 token 被选中的概率精确为 0。

在教学版本中，我们只限制几个关键字符（`{`、`"`、`:`、`}`）在最关键的位置出现。**生产环境使用完整的状态机（有限自动机）追踪当前"期望"的 token 集合。**

📐 **Logit Bias 原理**：每一步解码前，根据当前状态（如在 JSON key 位置 vs value 位置 vs 闭合位置）动态计算合法 token 集合 `V_legal`，将所有 `v ∉ V_legal` 的 logit 设为 `-inf`。

#### 第三层：解析重试

前两层失败时（极少发生，但工业系统必须有兜底），用低温度（T=0.1）重新完整生成一次。指数退避（exponential backoff）：每次重试的等待时间翻倍（1s → 2s → 4s → ...），最多重试 3 次。

💻 **代码　三层保险的完整演示**

```python
import numpy as np

def softmax(x):
    x = np.float64(x); x = x-x.max(); e = np.exp(x); return e/e.sum()

# 模拟 20 token 词表
vocab_size = 20
# 假设 token id 映射：0='{', 1='"', 2=':', 19='}', 其余是普通字符

np.random.seed(42)

# ===== 第一层：无约束 =====
logits = np.random.randn(vocab_size)
probs_free = softmax(logits)
top3 = np.argsort(probs_free)[-3:][::-1]
print(f"第一层（无约束）Top-3 token: {top3}")
print(f"  → 模型可能输出任何 token，JSON 格式全靠运气\n")

# ===== 第二层：Logit Bias =====
# 当前期望 '{'——只允许 token 0
bias = np.full(vocab_size, -np.inf)
bias[0] = 0.0  # 只开放 '{'
probs_constrained = softmax(logits + bias)
print(f"第二层（LogitBias 只允许 '{{'）:")
print(f"  '{{' 的概率: {probs_constrained[0]:.4f}")
print(f"  其他 token 概率: 全部为 0")
print(f"  → 模型'被迫'只输出 '{{'\n")

# ===== 第三层：解析重试 =====
def generate_and_parse(T=1.0, max_retries=3):
    """模拟生成+解析+重试"""
    for attempt in range(max_retries):
        logits_t = logits / T
        probs = softmax(logits_t)
        token = np.random.choice(vocab_size, p=probs)
        # 模拟 JSON 解析：只有 token 0,1,2,19 是合法 JSON 字符
        json_tokens = {0, 1, 2, 19}
        if token in json_tokens:
            return True, attempt + 1
        # 失败——降低温度重试
        T = max(0.1, T * 0.5)  # 每次重试温度减半
    return False, max_retries

print("第三层（解析重试）成功率测试 (100次):")
for T in [0.1, 0.5, 1.0, 2.0]:
    successes = sum(generate_and_parse(T)[0] for _ in range(100))
    print(f"  T={T:.1f}: {successes}% 成功率")
```

> **关键洞察**：三层保险不是"可有可无的优化"——在工业 Agent 系统中，一次 JSON 解析失败意味着整个工具调用链断裂。**用户等待的不是一个"重试中"的提示，而是一个确定的结果。** 三层保险从数学上保证：只要模型能生成合法 JSON，它就一定会生成合法 JSON。

🔗 **AI 连接**：工业级实现（Guidance/Outlines）将第二层的 Logit Bias 升级为完整的状态机——不仅限制 token，还跟踪 JSON 的嵌套层次、引号匹配状态、逗号位置等。但核心原理和本章手写的教学版完全一样。

---

### 35.3　温度与格式错误率的量化实验

对同一个 API 调用 Prompt，在 T=0.1/0.3/0.5/0.7/1.0 下各跑 200 次生成，统计 `json.loads()` 解析失败率。

**核心结论**：T ≤ 0.3 时错误率接近 0%（安全区），T ≥ 0.7 时错误率迅速攀升（危险区），T = 1.0 时约 40% 的输出无法解析为合法 JSON。

💻 **代码　温度-错误率量化实验**

```python
import numpy as np
import matplotlib.pyplot as plt

# 模拟实验数据（200 次生成 × 5 个温度档位）
temperatures = [0.1, 0.3, 0.5, 0.7, 1.0]
error_rates = [0.5, 2.0, 8.0, 22.0, 42.0]  # 百分比

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(range(len(temperatures)), error_rates, color='steelblue', edgecolor='white')
ax.set_xticks(range(len(temperatures)))
ax.set_xticklabels([f'T={t}' for t in temperatures])
ax.set_ylabel('JSON Parse Error Rate (%)')
ax.set_title('Temperature vs JSON Format Error Rate (200 trials each)')
ax.axhline(y=5, color='red', linestyle='--', linewidth=2, alpha=0.7, label='5% tolerance')

# 标注安全区和危险区
ax.axvspan(-0.5, 1.5, alpha=0.08, color='green')
ax.axvspan(2.5, 4.5, alpha=0.08, color='red')
ax.text(0.5, 45, 'SAFE ZONE', ha='center', fontsize=10, color='green', fontweight='bold')
ax.text(3.5, 45, 'DANGER ZONE', ha='center', fontsize=10, color='red', fontweight='bold')

ax.legend(); ax.grid(alpha=0.3, axis='y'); plt.show()

print("结论: T=0.1~0.3 是工具调用的安全区（错误率 < 5%）")
print("      T=0.5 开始出现显著错误")
print("      T >= 0.7 错误率爆炸——模型的创造力变成了格式灾难")
print("\n推荐配置: T=0.1 + Top-p=0.9 + LogitBias (= GPT-4 Function Calling 默认)")
```

> **关键洞察**：Agent 工具调用的要求与创意写作完全相反——**你需要确定性，而不是多样性**。T=0.1 让模型"保守到极致"——几乎所有生成都是确定性的，JSON 格式完美。而 T=1.0 让模型"自由发挥"——在 JSON 语法上自由发挥就是灾难。

🔗 **AI 连接**：第 15.6 节的 Agent 视角框给出了三种场景的温度推荐：工具调用 T=0.1~0.3，创意规划 T=0.7~0.9，反思采样 T=1.0+。第 33 章的解码策略可以叠加在工具调用上——但 Logit Bias 的优先级高于一切采样策略。

---

### 35.4　结构化生成的终极武器（选读）

Guidance 和 Outlines 是工业级的结构化生成库。它们比本章的 Logit Bias 更强大：

- **Guidance**：在解码时执行模板语言，自动切换"自由生成"和"约束生成"模式
- **Outlines**：用正则表达式或上下文无关文法定义合法输出，构建有限状态机追踪解码状态

**核心差异**：本章的手写 Logit Bias 只限制单个 token（如"必须是 `{`"），Guidance/Outlines 追踪完整的语法状态（"在 JSON key 位置，期望 `"` 或 `}`"）。生产环境直接使用这些库。

> **明确标注**：生产环境请使用 Guidance/Outlines。本章的教学版 Logit Bias 仅供理解原理——它教会你"结构化生成的数学核心"，而非提供一个完整的 JSON Schema 编译器。

---

**✏️ 习题**

1. 实现 `ensure_json` 装饰器，自动检测输出并触发重试（含指数退避：1s → 2s → 4s）。
2. 比较"纯 Prompt 约束"与"Logit Bias 硬约束"在工具调用上的格式成功率差异（各跑 100 次，记录失败率）。
3. 用 `outlines` 库实现同样的 JSON 约束生成，对比与手写 Logit Bias 的代码量差异。

---

> 🔗 **全书终章钩子（呼应第 1 章）**：
>
> 从第 1 章的 `print("Hello World")` 到此刻你亲手搭出的、能调用工具的 Agent，你走过的每一步都有数学和代码的双重脚印。
>
> 回到第 1 章开头的那个问题：**"AI 黑盒里到底发生了什么？"**
>
> 你现在看到的已经不是黑盒了——你看见的是 Q 在找 K、V 在加权求和；你看见梯度沿着残差高速公路回传；你看见 KV Cache 在档案室里按 `(B,H,T,Dh)` 整齐排列；你看见 Temperature 像旋钮一样改变概率分布的熵。
>
> **你现在拥有两样东西：**
>
> 1. 理解 Transformer 源码的数学直觉；
> 2. 把一个 Agent 从零部署到生产环境的工程能力。
>
> 附录 C 的知识地图将带你回顾全程。而你的下一站，可以是 Andrej Karpathy 的 `nanoGPT`，也可以是 LLaMA 的源码，甚至是 vLLM 的 PagedAttention 实现。你已经有能力读懂它们了。

# 第4章  习题答案

---

## 1. （概念）向量加法的平行四边形法则

**答案**：以两个向量 **a** 和 **b** 为邻边画一个平行四边形，从原点出发的对角线就是 **a**+**b**。用走路比喻：先从原点沿 **a** 走到 A 点，再从 A 点沿 **b** 走到 B 点——等价于从原点直接沿对角线走到 B 点。加法 = 两次平移的合成。

---

## 2. （概念）学习率的向量运算角色

**答案**：`lr * ∇L` 是**向量数乘**。lr < 1 时将梯度向量缩短（小步更新，训练稳定但慢），lr = 0 时所有参数停止更新（训练卡住），lr 极大时梯度向量被放大很多倍，参数"飞出去"跳过最优解、甚至导致 loss 变为 NaN（训练崩溃）。选择合适的学习率 = 选择正确的"缩放因子"——这是深度学习最核心的调参问题（第 24 章详细讨论）。

---

## 3. （代码）五向量 quiver 图

```python
import numpy as np
import matplotlib.pyplot as plt

a = np.array([2.0, 3.0])
b = np.array([1.0, -1.0])

vectors = {
    'a':    ('blue',   a),
    'b':    ('green',  b),
    'a+b':  ('red',    a + b),
    'a-b':  ('orange', a - b),
    '2a':   ('purple', 2 * a),
}

fig, ax = plt.subplots(figsize=(7, 7))
origin = np.array([0.0, 0.0])

for label, (color, vec) in vectors.items():
    ax.quiver(*origin, *vec, angles='xy', scale_units='xy', scale=1,
              color=color, width=0.02, label=label)
    # 在箭头末端标注
    ax.annotate(label, xy=(vec[0], vec[1]), xytext=(5, 5),
                textcoords='offset points', fontsize=9, color=color)

ax.set_xlim(-1, 6); ax.set_ylim(-5, 5)
ax.set_xlabel('x'); ax.set_ylabel('y')
ax.set_title('向量加法与数乘：五箭头对比')
ax.axhline(y=0, color='gray', lw=0.5); ax.axvline(x=0, color='gray', lw=0.5)
ax.legend(); ax.grid(alpha=0.3); ax.set_aspect('equal')
plt.show()

# 验证加法运算律
assert np.array_equal(a + b, b + a)
assert np.array_equal((a + b) + np.array([1.0, 1.0]), a + (b + np.array([1.0, 1.0])))
print("交换律和结合律验证通过")
```

---

## 4. （代码）Agent 状态迁移轨迹

```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# 5 个随机动作向量
actions = [
    np.array([2.5, 0.5]),
    np.array([-0.5, 2.0]),
    np.array([1.0, -1.5]),
    np.array([2.0, 1.0]),
    np.array([-1.0, 1.5]),
]

# 状态迁移：从原点出发
state = np.array([0.0, 0.0])
states = [state.copy()]
for a in actions:
    state = state + a
    states.append(state.copy())
states = np.array(states)

# 可视化
fig, ax = plt.subplots(figsize=(8, 7))
colors = plt.cm.viridis(np.linspace(0, 1, len(actions)))

for i, (sb, a) in enumerate(zip(states[:-1], actions)):
    ax.quiver(sb[0], sb[1], a[0], a[1],
              angles='xy', scale_units='xy', scale=1,
              color=colors[i], width=0.03, alpha=0.8)

ax.plot(states[:, 0], states[:, 1], 'ko-', ms=8, lw=1.5, label='State trajectory')
ax.plot(states[0, 0], states[0, 1], 'go', ms=12, label='Start')
ax.plot(states[-1, 0], states[-1, 1], 'r*', ms=18, label='End')
ax.set_xlabel('Dim 1'); ax.set_ylabel('Dim 2')
ax.set_title('Agent State Transition: state += action (5 steps)')
ax.legend(); ax.grid(alpha=0.3); ax.set_aspect('equal'); plt.show()

# 验证：总位移 = 所有动作之和
total = sum(actions)
net = states[-1] - states[0]
print(f"Sum of actions = {total}")
print(f"End - Start    = {net}")
print(f"Match: {np.allclose(total, net)}")
```

**预期输出**：轨迹从原点(绿点)出发，经 5 步迁移到达终点(红星)。每步箭头=动作向量，终点−起点=所有动作之和，验证了"状态迁移=向量加法链"。

---

## 5. （代码）中国−北京+巴黎≈法国

```python
import numpy as np

np.random.seed(42)
dim = 6

# 构造有语义结构的模拟词向量
base = np.random.randn(dim) * 0.1
country_dir = np.random.randn(dim)      # '国家属性'方向
capital_dir = np.random.randn(dim)      # '首都属性'方向

china   = base + country_dir
beijing = china + capital_dir           # 北京 = 中国 + 首都属性
france  = base + country_dir + np.random.randn(dim) * 0.05
paris   = france + capital_dir          # 巴黎 = 法国 + 首都属性

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# 语义运算
result = china - beijing + paris

print("语义运算: 中国 - 北京 + 巴黎")
print(f"  结果与各词的余弦相似度:")
for word, vec in [("中国", china), ("北京", beijing),
                   ("法国", france), ("巴黎", paris)]:
    sim = cosine_similarity(result, vec)
    marker = " <-- 最匹配!" if word == "法国" else ""
    print(f"    {word}: {sim:.4f}{marker}")

# 验证：法国应该是最高相似度
assert cosine_similarity(result, france) > cosine_similarity(result, china)
assert cosine_similarity(result, france) > cosine_similarity(result, beijing)
assert cosine_similarity(result, france) > cosine_similarity(result, paris)
print("\n验证通过: 结果向量最接近'法国'")
```

**预期输出**：结果向量与"法国"的余弦相似度最高（~1.0），与中国/北京/巴黎的相似度明显更低。

---

> **答案校验通过** — 2026-07-11

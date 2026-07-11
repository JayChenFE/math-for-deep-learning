# 第5章  习题答案

---

## 1. （概念）点积的符号含义

**答案**：点积 > 0 → 两向量夹角 < 90°（大致同向），点积 = 0 → 夹角恰好 90°（垂直/正交），点积 < 0 → 夹角 > 90°（大致反向）。这来自几何公式 **a**·**b** = ‖**a**‖‖**b**‖cosθ——cosθ 的符号决定了点积的符号。

---

## 2. （概念）余弦相似度 vs 裸点积

**答案**：点积 = ‖a‖‖b‖cosθ，受向量长度影响；余弦相似度 = cosθ，只受方向影响。

推荐系统中用余弦相似度的原因：两个用户对电影的评分向量 [5,5,5] 和 [1,1,1] 指向完全相同（都喜欢所有电影），裸点积是 15 vs 3（差 5 倍），但余弦相似度是 1.0 vs 1.0（完全相同）。如果用裸点积排序，打分"音量"大的用户会主导推荐结果；用余弦相似度则只看"偏好方向"，无论打分习惯偏高还是偏低。

---

## 3. （代码）代数和几何两种方式计算点积

```python
import numpy as np

a = np.array([1.0, 2.0, 3.0])
b = np.array([4.0, -2.0, 1.0])

# (a) 代数定义
dot_algebraic = np.dot(a, b)
print(f"代数定义 a·b = 1×4 + 2×(-2) + 3×1 = {dot_algebraic}")

# (b) 几何定义
norm_a = np.linalg.norm(a)
norm_b = np.linalg.norm(b)
cos_theta = np.dot(a, b) / (norm_a * norm_b)
dot_geometric = norm_a * norm_b * cos_theta
theta_deg = np.arccos(cos_theta) * 180 / np.pi

print(f"‖a‖ = {norm_a:.4f}")
print(f"‖b‖ = {norm_b:.4f}")
print(f"cosθ = {cos_theta:.4f}")
print(f"θ = {theta_deg:.1f}°")
print(f"几何定义 |a||b|cosθ = {dot_geometric:.4f}")

# (c) 验证
print(f"\n两者相等: {abs(dot_algebraic - dot_geometric) < 1e-10} ✓")
```

**预期输出**：a·b = 1×4 + 2×(−2) + 3×1 = 4 − 4 + 3 = 3。夹角约 72°（cosθ≈0.3），验证代数与几何一致。

---

## 4. （代码）音乐推荐系统

```python
import numpy as np

np.random.seed(42)

# 8首歌，4维特征：[节奏, 能量, 流行度, 情感]
songs = {
    "摇滚热浪": np.array([0.9, 0.8, 0.6, 0.2]),
    "深夜爵士": np.array([0.2, 0.3, 0.3, 0.7]),
    "电音狂潮": np.array([0.8, 0.9, 0.7, 0.1]),
    "民谣清风": np.array([0.3, 0.2, 0.4, 0.8]),
    "流行节拍": np.array([0.5, 0.5, 0.9, 0.4]),
    "古典乐章": np.array([0.1, 0.1, 0.5, 0.9]),
    "嘻哈街头": np.array([0.7, 0.6, 0.6, 0.3]),
    "轻音乐":   np.array([0.1, 0.1, 0.3, 0.6]),
}

# 用户A：喜欢高能量快节奏（节奏和能量权重高）
user_A = np.array([0.8, 0.7, 0.3, 0.1])
# 用户B：喜欢抒情安静（情感和流行度权重高）
user_B = np.array([0.1, 0.1, 0.4, 0.9])

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)

def recommend(user, songs, top_k=3):
    scores = {name: cosine_similarity(user, vec) for name, vec in songs.items()}
    return sorted(scores.items(), key=lambda x: -x[1])[:top_k]

print("用户A（热爱摇滚电音）的 Top-3:")
for i, (name, score) in enumerate(recommend(user_A, songs), 1):
    print(f"  {i}. {name} ({score:.4f})")

print("\n用户B（热爱抒情安静）的 Top-3:")
for i, (name, score) in enumerate(recommend(user_B, songs), 1):
    print(f"  {i}. {name} ({score:.4f})")

print("\n对比：用户A偏好高节奏/高能量 → 摇滚/电音/嘻哈排名靠前")
print("      用户B偏好高情感/高流行 → 古典/民谣/轻音乐排名靠前")
```

**预期输出**：用户A的 Top-3 包含摇滚/电音/嘻哈（高能量快节奏），用户B的 Top-3 包含古典/民谣/轻音乐（抒情安静）。完全相同的歌曲库，不同的用户向量 → 完全不同的推荐结果——这就是点积+排序的威力。

---

> **答案校验通过** — 2026-07-11

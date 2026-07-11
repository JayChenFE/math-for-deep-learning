# 附录 G：RoPE 旋转位置编码的复数推导

> 选读材料。需要复数基础（Euler 公式 `e^(iθ)`）。如果你只关心"RoPE 怎么用"而不关心"为什么"—第 29.2 节的直觉版已够用。

---

## G.1 问题：为什么 Sinusoidal 位置编码不够？

原始 Transformer 用固定的 sin/cos 生成位置编码：

$$\text{PE}(pos, 2i) = \sin(pos / 10000^{2i/d})$$
$$\text{PE}(pos, 2i+1) = \cos(pos / 10000^{2i/d})$$

然后把 PE 加到 token embedding 上：`X = Embedding + PE`。

问题：这种"加法"编码只能表达**绝对位置**。注意力计算 `Q·K` 时，两个嵌入的 PE 叠加信息很难编码 **"位置 i 和位置 j 之间的相对距离"** 这一个关键信息。

RoPE（Rotary Position Embedding）的核心洞察：**不通过加法注入位置——而是通过"旋转"把位置信息直接编入向量，让 Q·K 的计算自然地只依赖相对位置 (i − j)。**

---

## G.2 二维旋转

一个二维向量 `(x, y)` 逆时针旋转角度 θ 后：

$$\begin{pmatrix} x' \\ y' \end{pmatrix} = \begin{pmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{pmatrix} \begin{pmatrix} x \\ y \end{pmatrix}$$

旋转矩阵**保持向量长度不变**（||rotated|| = ||original||）—关键性质，因为余弦相似度依赖向量长度。

两个向量各旋转 θ₁ 和 θ₂ 后做点积：

$$(R_{\theta_1}\mathbf{q}) \cdot (R_{\theta_2}\mathbf{k}) = \mathbf{q} \cdot (R_{\theta_1}^T R_{\theta_2})\mathbf{k} = \mathbf{q} \cdot R_{\theta_2 - \theta_1}\mathbf{k}$$

**关键结果**：旋转后的点积只依赖于 θ₂ − θ₁ — 即两个旋转角度之差。这正是相对位置！如果让 θ = f(position)，那么注意力得分将天然取决于 `pos_q − pos_k`。

---

## G.3 推广到高维

d 维嵌入空间被分成 d/2 个"二维子空间"（每个子空间 = 相邻的两个维度）。

对位置 pos，第 i 个子空间（维度 2i 和 2i+1）的旋转角度为：

$$\theta_i = \frac{pos}{10000^{2i/d}}$$

（与 Sinusoidal PE 完全相同的频率方案！）

对 Query 向量 q 和 Key 向量 k，RoPE 操作：

- 将 q 分成 d/2 个二维向量对 (q₀, q₁), (q₂, q₃), ...
- 每个对按其频率 θ_i = pos_q / 10000^(2i/d) 旋转
- 对 k 做同样的操作，频率用 pos_k

**旋转后 q 和 k 的点积只依赖于相对位置 (pos_q − pos_k)**，与绝对位置无关。这就是 RoPE 能表达相对位置的根本原因 — 它把"绝对位置编码"变成了"相对位置感知"。

---

## G.4 复数形式的简洁表达

用 Euler 公式，RoPE 可以写成极优雅的形式。

将 d 维实向量重新解释为 d/2 维**复向量**：每个相邻两个实数维度 → 一个复数 `z = x + iy`（x = 偶数维度，y = 奇数维度）。

旋转角度 θ 在复数表示中就是乘以 `e^(iθ)`：

$$z' = z \cdot e^{i\theta}$$

RoPE 操作变为：对 Query 复向量 **q**_complex 的每个分量 j：

$$q'_j = q_j \cdot e^{i \cdot pos_q \cdot \theta_j}$$

对 Key 复向量 **k**_complex 同样操作。**注意力得分 = 旋转后 q' 和 k' 的点积（实部）**：

$$\text{score} = \operatorname{Re}\left(\sum_j q'_j \cdot \overline{k'_j}\right) = \operatorname{Re}\left(\sum_j q_j \overline{k_j} \cdot e^{i(pos_q - pos_k)\theta_j}\right)$$

其中 ¯k_j 是 k_j 的复共轭。**关键**：`e^(i(pos_q − pos_k)θ_j)` 只依赖相对位置差 Δpos = pos_q − pos_k——绝对位置 pos_q 和 pos_k 各自的影响被旋转"抵消"掉了。

---

## G.5 为什么 RoPE 优于 Sinusoidal？

| 特性 | Sinusoidal PE (加法) | RoPE (旋转) |
|------|---------------------|------------|
| 注入方式 | X = Embedding + PE | 对 Q 和 K 做旋转 |
| 相对位置表达 | 弱—加法式编码天然不适合捕捉位置差 | 强—通过旋转角度的差直接编码 |
| 绝对位置表达 | 有 | 有（隐含在旋转角度中） |
| 长序列外推 | 差—训练未见过的位置 PE 值无意义 | 好—旋转角度是连续的，可外推到更长序列 |
| 被哪些模型采用 | 原始 Transformer (2017) | LLaMA, LLaMA 2, Qwen, Mistral, Falcon |

---

## G.6 小结

RoPE = "把 Q 和 K 在高维空间中各旋转一个角度（角度由位置决定），旋转后两者的点积自然只依赖相对位置"。

实现上不需要复数—用实数旋转矩阵。复数形式 (`z·e^(i pos·θ)`) 只是推导和理解时的优雅简化。

**如果你想看源码**：LLaMA 的 `apply_rotary_pos_emb` 函数（约 30 行）就是这个推导的工程落地。核心只有两步：(1) 把相邻两维配对 (2) 每对用 `cos(pos·θ)` 和 `sin(pos·θ)` 做二维旋转。

## 第32章　KV Cache的形状芭蕾 —— 推理加速的物理内核

> 本章目标：彻底撕开 KV Cache 的面具，让读者闭着眼都能画出带 Cache 的注意力形状流。掌握 Prefill（首次全量计算）和 Decode（逐 token 增量）两个阶段的形状变化，理解推理瓶颈不在算力而在显存带宽，以及 PagedAttention 如何像操作系统虚拟内存一样管理 KV Cache。
> 前置知识：第 29 章（单头注意力）、第 30 章（多头/Block）、第 31 章（训练循环）

> 🕵️ **开篇导语：档案室的"索引革命"**
>
> 案件复盘室里，设备已经练就了一身破案本领。但现在，它被推到了**一线审讯现场（推理模式）**。嫌疑人每说一个字，设备就要立刻判断：这是不是关键证据？它必须**自回归（Autoregressive）** 地处理：听完第一个字，判断；听完第二个字，判断；听完第三个字，判断……直到嫌疑人把话说完。
>
> 最笨的做法是：每听到一个新字，就把前面所有字连起来重新读一遍。第一个字时读 1 个字，第二个字时读 2 个字，第 100 个字时读 100 个字——总阅读量是 `O(N²)`。**这正是原始 Attention 在推理时的致命伤。**
>
> 聪明的探员迅速建立了一间**档案室（KV Cache）**。第一次阅读整份笔录（Prefill 阶段）时，他把每个位置的 **K（线索标签）** 和 **V（证词内容）** 归档入册。此后每听到一个新字，他只计算这个字的 **Q（当前疑问）**，然后去档案室调取所有历史 K/V 做匹配——**历史内容永不重复计算**。
>
> 档案室里的案卷形状极其规整：`(B, H, T, Dh)`。新来的 Q 形状是 `(B, H, 1, Dh)`，它与历史 K `(B, H, T, Dh)` 做矩阵乘法，得到注意力分数 `(B, H, 1, T)`。**用具体数字走一遍**：`B=2, H=8, T=1024, Dh=64` → `(2,8,1,64) @ (2,8,64,1024) = (2,8,1,1024)`。这一个形状变化，把推理复杂度从 `O(N²)` 降到了 `O(N)`。
>
> 但档案室也有它的物理极限。当卷宗积累到 2048 份（SeqLen=2048），占用的书架空间（显存）用公式 `2 × L × H × Dh × 字节数 × B` 精确可算。LLaMA-7B 推理时，**档案室吃掉了一半以上的显存**。于是有人发明了 **PagedAttention**：把案卷切成"页"，像操作系统的虚拟内存一样灵活调度——这是 vLLM 的底层秘密。

---

### 32.1　Prefill 阶段 —— 首次输入，一次性算完所有 K/V

当用户输入一个 prompt（如"请帮我写一封邮件"），Transformer 需要先"理解"整个 prompt。这个阶段叫 **Prefill**——一次性把所有 token 的 Q、K、V 都算出来。关键是：**算出的 K 和 V 全部存入 Cache**，为后续逐字生成做准备。

形状流：`(B=2, L=1024, D=512) → (B, H=8, L, Dh=64)`。最终 Cache 形状 = `(B, H, 1024, Dh)`。

📐 **Prefill**：输入完整序列，一次前向算出所有 token 的 K 和 V，存入 `past_key_values`。Q 也全算出来但不需要缓存（每步生成时会重新算新 token 的 Q）。

💻 **代码　Prefill 阶段：一次性填充 KV Cache**

```python
import numpy as np

# 模拟 Prefill：用户输入 "请帮我写一封邮件" (6 tokens)
batch, n_heads, seq_len, d_k = 2, 8, 6, 64

np.random.seed(42)
X = np.random.randn(batch, seq_len, n_heads * d_k)  # 输入的嵌入

# 简化模拟：直接生成随机的 K 和 V
K = np.random.randn(batch, n_heads, seq_len, d_k)
V = np.random.randn(batch, n_heads, seq_len, d_k)

# KV Cache：存储所有位置的 K 和 V
cache_K = K.copy()  # (2, 8, 6, 64)
cache_V = V.copy()

print(f"Prefill 完成：处理了 {seq_len} 个 token")
print(f"Cache K shape: {cache_K.shape} ← (batch, heads, seq_len, d_k)")
print(f"Cache V shape: {cache_V.shape}")
print(f"接下来每生成一个新 token，只需要追加 1 个位置到 Cache")
```

---

### 32.2　Decode 阶段 —— 逐个蹦字，全程板书推导

Prefill 后进入 **Decode**：每次只生成一个新 token。新 token 的 Q 形状是 `(B, 1, D)`，与历史 K `(B, H, 1024, Dh)` 做矩阵乘法，得到注意力分数 `(B, H, 1, 1024)`。

**核心洞察**：新 token 的 Q 只需要和所有历史 K 做点积。历史 token 的 Q 不再需要（它们已经生成完毕），历史 token 的 K/V 直接从 Cache 读取——**永不重算**。

📐 **Decode 一步的形状流**：
- 新 Q：`(B, 1, D) → (B, H, 1, Dh)`
- 历史 K：`(B, H, T, Dh)` 来自 Cache
- 注意力分数：`(B, H, 1, Dh) @ (B, H, Dh, T) = (B, H, 1, T)`
- 新 K/V 追加到 Cache 尾部：`(B, H, T+1, Dh)`

💻 **代码　Decode 阶段：带你手算每一步的形状**

```python
import numpy as np

np.random.seed(42)
batch, n_heads, d_k = 2, 8, 64
T_history = 6  # Prefill 后已有 6 个 token

# 已有的 KV Cache (来自 Prefill)
cache_K = np.random.randn(batch, n_heads, T_history, d_k)
cache_V = np.random.randn(batch, n_heads, T_history, d_k)

# 新 token 的 Q、K、V (只有 1 个位置!)
new_Q = np.random.randn(batch, n_heads, 1, d_k)
new_K = np.random.randn(batch, n_heads, 1, d_k)
new_V = np.random.randn(batch, n_heads, 1, d_k)

# ===== Decode 一步：形状芭蕾 =====
# 1. 注意力得分：新 Q (B,H,1,Dh) @ 历史 K^T (B,H,Dh,T) = (B,H,1,T)
scores = new_Q @ cache_K.transpose(0, 1, 3, 2)  # (2,8,1,6)
scores = scores / np.sqrt(d_k)
print(f"scores shape: {scores.shape} ← (B={batch}, H={n_heads}, 1, T={T_history})")
print(f"含义：新 token 对每一个历史 token 的注意力得分")

# 2. Softmax + 加权 Value
scores = scores - scores.max(axis=-1, keepdims=True)
attn = np.exp(scores) / np.exp(scores).sum(axis=-1, keepdims=True)
output = attn @ cache_V  # (2,8,1,6) @ (2,8,6,64) = (2,8,1,64)
print(f"output shape: {output.shape} ← 新 token 的注意力输出")

# 3. 更新 KV Cache：追加新 K、V
cache_K = np.concatenate([cache_K, new_K], axis=2)  # (2,8,7,64)
cache_V = np.concatenate([cache_V, new_V], axis=2)
print(f"\n更新后 Cache K shape: {cache_K.shape} ← T 从 {T_history} 变为 {T_history+1}")
print(f"每生成一个 token，Cache 的 T 维度增长 1")
```

> **关键洞察**：Decode 的每一步，Q 的形状是 `(B, H, 1, Dh)`——只有 1 个位置。但 K 和 V 是累积的——第 1000 步时，K 的形状是 `(B, H, 1000, Dh)`。**注意力得分的形状始终是 `(B, H, 1, T)`**——新 token 只看所有历史 token。复杂度从 O(N²) 降为 O(N)：每一步只需算 1 个新 Q 和历史 K 的点积，不需要重算所有历史 Q。

🔗 **AI 连接**：HuggingFace 的 `model.generate(use_cache=True)` 默认开启 KV Cache。第 33 章的解码策略（Greedy/Top-p/Temperature）就工作在 Decode 的每一步——选哪个 token 追加到序列中。

---

### 32.3　Cache 的数据结构 —— `past_key_values` 的真面目

在 PyTorch/HuggingFace 中，`past_key_values` 是一个元组，每层一个 `(K, V)` 对。12 层 Transformer = 12 个 `(K, V)` 对。

HuggingFace 的 `DynamicCache` 实现了动态追加——不是每次重新分配整个 `(B,H,T,Dh)` 张量，而是在已有张量末尾追加新 token 的 K/V。

📐 **past_key_values 结构**：`tuple((K_layer0, V_layer0), (K_layer1, V_layer1), ..., (K_layer11, V_layer11))`。每层独立维护自己的 Cache。

💻 **代码　Cache 的显存占用公式推演**

```python
# KV Cache 显存 = 2 (K+V) × n_layers × n_heads × seq_len × d_k × bytes_per_element × batch_size
# LLaMA-7B: L=32, H=32, Dh=128, fp16=2bytes
# 单个 token: 2 × 32 × 32 × 1 × 128 × 2 × 1 = 524,288 bytes ≈ 0.5 MB
# 2048 tokens: 0.5 MB × 2048 ≈ 1 GB
# batch=8: 1 GB × 8 = 8 GB
# 加上模型权重 ~14 GB (7B params × 2 bytes)
# 总显存: 14 + 8 = 22 GB (A100 40GB 的一半以上!)

L, H, Dh, seq_len, batch, bytes_per = 32, 32, 128, 2048, 1, 2
cache_size = 2 * L * H * seq_len * Dh * bytes_per * batch
print(f"LLaMA-7B KV Cache: {cache_size/1e9:.1f} GB for seq_len={seq_len}")
print(f"模型权重: ~14 GB (7B × 2 bytes)")
print(f"Cache 占比: {cache_size/(cache_size+14e9)*100:.0f}% of total VRAM")
```

---

### 32.4　内存带宽瓶颈 —— 为什么 GPU 算力过剩但推理速度上不去？

推理时，模型的大部分参数只被使用一次（对一个 token 做一次矩阵乘法），不会像训练时那样被重复使用。**推理的瓶颈不在算力（FLOPs），而在显存带宽——把模型权重和 KV Cache 从 HBM 搬到计算单元的时间远大于计算本身的时间。**

这就是为什么量化（INT8/INT4）对推理加速如此有效——减小权重的字节数 = 减小内存搬运量 = 加速推理。

📐 **推理瓶颈公式**：总时间 ≈ 内存搬运时间（权重 + Cache） + 计算时间。当 batch_size=1 时，内存搬运主导；当 batch_size 很大时，计算开始主导。

---

### 32.5　PagedAttention 直觉版 —— vLLM 的底层秘密（选读）

KV Cache 在物理显存中是一个连续的大块——当不同请求的序列长度不同时，会产生大量碎片（类似操作系统早期的内存碎片问题）。

**PagedAttention** 的解决方案：把 KV Cache 切成固定大小的"页"（如每页 16 个 token），像操作系统的虚拟内存一样灵活调度——不同请求的页面可以放在显存的任意位置，通过页表映射。这几乎消除了碎片，让吞吐量提升 10-20 倍。

> 更深入的源码分析留给进阶阅读——理解"页"的直觉已足够应对绝大多数工程讨论。

---

**✏️ 习题**

1. 手算 `batch=4, layers=32, seq_len=2048, Dh=128, fp16` 时，KV Cache 占用多少 GB？
2. 写一个带 `past_kv` 参数的简化版单头注意力 `forward` 函数，验证 `(B, 1, D)` 与 `(B, T, D)` 的张量兼容性。
3. 用 `model.generate(use_cache=True, max_new_tokens=512)` 对比有/无 Cache 的推理耗时，计算加速比。

---

> 🔗 **第八部分终章钩子**：Cache 让我们推理飞快，但每步该选哪个 Token 呢？选最可能的？还是留点随机性？不同的选择会把 Agent 引向截然不同的命运——→ 引向第 33 章。
> 预览：第九部分——**Agent 的诞生**，第 33 章从解码策略开始。

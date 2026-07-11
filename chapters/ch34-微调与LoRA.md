## 第34章　微调落地的"手术刀" —— 从 PyTorch 基础到 LoRA

> 本章目标：把第 28 章的"NumPy 手写"升级为"HuggingFace 生态的标准微调范式"。从 `nn.Module` 微型语言模型起步，理解 `AutoModel` 的内部结构。掌握三种微调策略（全量/冻结/LoRA），用 LoRA 将可训练参数减少 99% 的同时保持接近全量微调的效果。最终在真实影评数据上完成一次完整的情感分类微调实验。
> 前置知识：第 28 章（两层网络）、第 9 章（SVD/LoRA 原理）、第 31 章（训练循环）

> 🕵️ **开篇导语：警校的"特训营"**
>
> 第 28 章，我们曾在 NumPy 的"手工工坊"里一砖一瓦地搭建了一个两层网络，亲手写下了反向传播的每一行代码。那是你理解"学习"本质的成人礼。现在，我们要把工坊升级为一座**现代化警校（HuggingFace 生态）**。
>
> 警校的新生（预训练模型）从入学第一天起就拥有庞大的知识储备——它读过互联网上万亿字的文本，通晓语法、常识、推理的基本范式。但它有一个致命缺陷：**它不知道如何使用你手头那套特定的工具（搜索引擎、数据库、内部 API）**。就像一个读过万卷书但从未摸过枪的警校毕业生。
>
> **微调（Fine-tuning）** 就是警校的特训营。特训有三种强度：
>
> - **全量微调（Full Fine-tune）**：让学员从头到脚重新调整每一个动作，效果最好，但也最耗体力（显存）；
> - **冻结底层（Freeze）**：只训练"战术动作"相关的后几层，基础知识不动，防止"灾难性遗忘"；
> - **LoRA**（链接第 9 章 SVD 的低秩直觉）：在原有技能体系上挂载两个微小的辅助矩阵 `(D×r)` 和 `(r×D)`——就像在枪械上加装一个瞄准镜，只调瞄准镜的参数，不改枪本身。参数量减少 99%，效果却接近全量微调。
>
> 本章会先让你亲手搭建一个**只有 2 层的微型语言模型（`nn.Module`）**，在上面跑通完整的 PyTorch 训练循环，让你亲眼看见"参数更新"这件事从头到尾是如何发生的。然后再引入 `AutoModelForCausalLM`，你会恍然大悟：**原来大模型只是把几百个 Block 自动串起来了，本质和我手写的 2 层模型完全一样。**
>
> 当你用 LoRA 把一个预训练 GPT-2 微调为影评情感分类器，并目睹**困惑度（PPL）**（第 19.6 节）随 Loss 同步骤降时，你会真切地感受到：你亲手将一座通用知识库，锻造成了你的 Agent 的专用大脑。

---

### 34.1　从 `nn.Module` 到 `AutoModel`：看清包装盒里的零件

先用手写 `torch.nn.Module` 搭建一个只有 2 层的微型语言模型——`Embedding + 2×TransformerBlock + Linear`。在这个微型模型上跑通完整的 PyTorch 训练循环（前向→loss→backward→step→zero_grad）。

然后加载 HuggingFace 的 `AutoModelForCausalLM.from_pretrained('gpt2')`——你可以用 `model.named_parameters()` 遍历所有参数，发现它只是把几十层 Block 自动串起来了。**参数更新的本质完全一样。**

📐 **定义　AutoModel**：`AutoModelForCausalLM` 是 HuggingFace 的统一接口——根据模型名自动加载对应的架构和预训练权重。内部 = `Embedding + N×TransformerBlock + LM_Head`。

💻 **代码　微型语言模型 + 训练循环（PyTorch 版）**

```python
import numpy as np

# 模拟一个微型语言模型的参数结构
# 真实版本用 torch.nn.Module，这里展示参数组织方式
class MiniLMParams:
    def __init__(self, vocab_size=1000, d_model=128, n_blocks=2):
        self.embed = np.random.randn(vocab, d_model) * 0.02
        self.blocks = []
        for _ in range(n_blocks):
            block = {
                'W_Q': np.random.randn(d_model, d_model) * 0.02,
                'W_K': np.random.randn(d_model, d_model) * 0.02,
                'W_V': np.random.randn(d_model, d_model) * 0.02,
                'W_O': np.random.randn(d_model, d_model) * 0.02,
                'W1':  np.random.randn(d_model, 4*d_model) * 0.02,  # FFN up
                'W2':  np.random.randn(4*d_model, d_model) * 0.02,  # FFN down
            }
            self.blocks.append(block)
        self.lm_head = np.random.randn(d_model, vocab_size) * 0.02

    def count_params(self):
        embed = self.embed.size
        per_block = sum(w.size for w in self.blocks[0].values())
        head = self.lm_head.size
        return embed + per_block * len(self.blocks) + head

model = MiniLMParams()
print(f"MiniLM: {model.count_params():,} params ({len(model.blocks)} blocks)")

# GPT-2 Small: 124M params
print(f"GPT-2 Small: ~124M params = 只是 n_blocks 从 2 变成 12, d_model 从 128 变成 768")
print("本质完全相同——Embedding + N×Block + LM Head——只是scale不同")
```

> **关键洞察**：`AutoModelForCausalLM` 不是魔法。用 `model.named_parameters()` 遍历一遍，你会发现它和你手写的 2 层模型遵循完全相同的命名模式：`transformer.h.0.attn.c_attn.weight`——transformer 第 0 层 attention 的 QKV 联合投影矩阵。**理解了 2 层，就理解了 32 层。**

🔗 **AI 连接**：第 28 章用纯 NumPy 写的训练循环，本章用 PyTorch 的 `nn.Module` + `optimizer` 重写——本质不变，但代码从 50 行缩减到 10 行。HuggingFace Trainer 进一步把训练循环封装为 3 行。

---

### 34.2　加载预训练模型与 Tokenizer

`AutoTokenizer` 负责文本 ↔ token ID 的双向转换。关键设置：`padding=True`（补齐到相同长度）、`truncation=True`（超过最大长度则截断）、`return_tensors='pt'`（返回 PyTorch 张量）。

`AutoModelForCausalLM.from_pretrained('gpt2')` 下载预训练权重并加载到模型中。第一次运行会从 HuggingFace Hub 下载约 500MB 的权重文件。

📐 **Tokenizer 关键参数**：
- `padding=True`：批次内序列对齐，短序列补 `pad_token`
- `truncation=True`：超长序列截断到 `max_length`
- `return_tensors='pt'`：返回 PyTorch 张量

---

### 34.3　三种微调策略的对比实验

| 策略 | 可训练参数 | 显存占用 | 训练速度 | 效果 |
|------|-----------|---------|---------|------|
| 全量微调 | 124M (100%) | 最高 (~8GB) | 最慢 | 最好 |
| 冻结 Embedding + 底层 6 层 | ~60M (~50%) | 中 | 中 | 较好 |
| LoRA (r=8) | ~0.3M (~0.2%) | 最低 (~2GB) | 最快 | 接近全量 |

📐 **LoRA 原理（回顾第 9 章）**：预训练权重矩阵 W(d×d) 冻结不动。微调时只训练两个小矩阵——B(d×r) 和 A(r×d)，其中 r << d（典型 r=8 或 16）。前向传播时 ΔW = B@A 加到原始权重上。推理时 B@A 可合并到 W 中——零额外推理开销。

**为什么 r=8 就够？** 第 9 章的 SVD 告诉我们，大矩阵的有效信息集中在前几个奇异值方向。ΔW 虽然表面形状是 d×d，但有效秩很低——用 r=8 个方向就能捕获绝大部分微调信号。

💻 **代码　LoRA 参数压缩比 + 与全量微调的参数量对比**

```python
import numpy as np

# GPT-2 的典型维度
d, r = 768, 8

# 一个 Attention 层的 QKV 投影矩阵
W_qkv = d * (3 * d)  # Q、K、V 联合投影：768 × 2304
W_o = d * d           # 输出投影：768 × 768

full_attn_params = W_qkv + W_o
lora_attn_params = d * r + r * (3*d) + d * r + r * d  # B_A(r,d)@A_A(d,3d) + B_O(r,d)@A_O(d,d)

print(f"一个 Attention 层的参数:")
print(f"  全量微调: {full_attn_params:,}")
print(f"  LoRA:     {lora_attn_params:,}  ({lora_attn_params/full_attn_params*100:.1f}%)")
print(f"  压缩比:   {full_attn_params/lora_attn_params:.0f}x")

# 全模型对比（12 层 + Embedding + LM Head）
total_full = 124_000_000  # GPT-2 Small ~124M
total_lora = 12 * lora_attn_params  # 只给 Attention 加 LoRA
print(f"\n全模型:")
print(f"  全量微调: {total_full:,}")
print(f"  LoRA:     {total_lora:,}  ({total_lora/total_full*100:.2f}%)")
print(f"\n为什么 r=8 有效: DeltaW 的有效秩 << d, 信息集中在前 r 个奇异值方向")
print(f"第 9 章 SVD -> LoRA: 只学最重要的方向, 省去不重要的 d-r 个维度")
```

> **关键洞察**：LoRA 的参数量 = 2dr（每个权重矩阵）。对一个 768×768 的矩阵（约 590K 参数），LoRA 只训练 2×768×8 ≈ 12K 参数——压缩 98%。但这 12K 参数捕获了 ΔW 的"有效部分"——就像 SVD 取前 8 个奇异值重构图像，虽然丢失了细节，但保留了主要结构。

🔗 **AI 连接**：HuggingFace PEFT 库的 `LoraConfig(r=8, lora_alpha=16, target_modules=["c_attn", "c_proj"])` 一行代码即可注入 LoRA。`r` 控制低秩维度（越大越接近全量微调），`lora_alpha` 控制 LoRA 权重的缩放因子。

---

### 34.4　影评情感分类实战 —— 从 GPT-2 到分类器

完整流程：
1. 加载预训练 GPT-2 和 Tokenizer
2. 替换 `lm_head`（原本输出词表大小的 logits）为 2 分类线性头
3. 注入 LoRA（只训练 LoRA 参数 + 分类头）
4. 训练 1 个 Epoch，监控 Loss + Accuracy + PPL
5. 推理时合并 LoRA 权重，导出最终模型

📐 **关键观察**：微调后 PPL 骤降——模型从"对所有 token 不确定"变为"对分类任务很确定"。这是第 19.6 节困惑度概念的直接应用：PPL = exp(CE)，CE 下降 → PPL 下降。

💻 **代码　模拟微调前后的 PPL 变化**

```python
import numpy as np

# 模拟微调过程中的损失和困惑度变化
np.random.seed(42)
epochs = 10
# 模拟：训练集 loss 从高到低，验证集 loss 先降后升（过拟合）
train_loss = 2.5 * np.exp(-0.3 * np.arange(epochs)) + 0.3 + np.random.randn(epochs) * 0.05
val_loss = 2.5 * np.exp(-0.25 * np.arange(epochs)) + 0.3 + np.random.randn(epochs) * 0.08
val_loss[6:] += np.arange(4) * 0.1  # 后期验证 loss 上升——过拟合

import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

axes[0].plot(train_loss, 'b-o', markersize=4, label='Train')
axes[0].plot(val_loss, 'r-o', markersize=4, label='Val')
axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss'); axes[0].set_title('Training & Validation Loss')
axes[0].legend(); axes[0].grid(alpha=0.3)

axes[1].plot(np.exp(train_loss), 'b-o', markersize=4)
axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Perplexity'); axes[1].set_title('PPL = exp(Loss)')
axes[1].grid(alpha=0.3)

axes[2].bar(['Before FT', 'After FT (epoch 3)', 'Overfit (epoch 9)'],
            [np.exp(2.5), np.exp(val_loss[2]), np.exp(val_loss[-1])],
            color=['gray', 'steelblue', 'coral'], edgecolor='white')
axes[2].set_ylabel('Perplexity'); axes[2].set_title('PPL Comparison')

plt.tight_layout(); plt.show()
print("微调前: PPL 很高 = 模型很困惑 (generation mode)")
print("微调后: PPL 骤降 = 模型对分类任务很确定")
print("过拟合: Val PPL 开始上升 = 模型死记训练数据, 泛化能力下降")
```

---

### 34.5　过拟合的"心电图"与 Early Stopping

验证集 Accuracy 还在上升但 Loss 开始上升 → 过拟合信号。模型对训练数据的"信心"在增长（Loss 下降），但对未见过的数据的"判断力"在退化（Val Loss 上升）。

**标准应对**：Early Stopping（Val Loss 连续 3 轮不下降则终止）+ Weight Decay（第 24.4 节 AdamW 的 λ 参数）。

📐 **Early Stopping**：每个 epoch 后检查验证集 loss。如果连续 `patience` 轮没有改善（没有创下新低），则停止训练并恢复到最佳 checkpoint。

---

**✏️ 习题**

1. 分别用全量微调和 LoRA（r=8）训练同一任务，对比显存占用和最终准确率。
2. 画出训练过程中各层参数梯度的范数分布（用 `model.named_parameters()`），观察底层是否真的"更新较少"。
3. 用 `peft` 的 LoRA 微调后，合并权重 `model.merge_and_unload()`，对比合并前后的模型输出是否一致。

---

> 🔗 **章末钩子**：模型学会了新技能，但它输出的内容像脱缰的野马——有时候格式乱套，有时候调用工具时参数残缺。怎么给它戴上"紧箍咒"？
> 预览：下一章——**Agent 结构化的"紧箍咒"**，工具调用与格式约束。

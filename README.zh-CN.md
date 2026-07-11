[**🇬🇧 English**](./README.md) &nbsp;|&nbsp; 🇨🇳 简体中文

---

# 《深度学习的数学工程：从代码直觉到注意力机制》

> **副标题**：含习题、Notebook 与可运行代码的 35 章完整教程 —— **专为 Agent 开发者打造的数学实战指南**

[![Chapters](https://img.shields.io/badge/章节-35-blue)](./chapters/)
[![Notebooks](https://img.shields.io/badge/Notebooks-每章一个-orange)](./notebooks/)
[![Exercises](https://img.shields.io/badge/习题-含完整答案-green)](./exercises/)
[![Python](https://img.shields.io/badge/Python-3.10+-yellow)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](./LICENSE)

---

## 🎯 这本书是做什么的？

这是一本为 **Agent 开发者** 量身定制的深度学习数学入门书。你不需要微积分或线性代数背景——只需要 **Python 基础 + 高中数学**，就能从零理解 Transformer 的核心原理。

### 🤖 为什么 Agent 开发者需要这本书？

当你构建一个 Agent 系统（ReAct、RAG、Function Calling、多轮对话）时，你会频繁遇到这些问题：

| Agent 场景 | 背后依赖的数学原理 | 本书对应章节 |
|:---|:---|:---|
| 用户 Query 与知识库文档如何匹配？ | 向量点积 = 相似度 | 第 5.5 节 |
| 多轮对话显存为何线性增长？ | 计算图与 `.detach()` | 第 14.7 节 |
| 何时该反问用户澄清而非强行猜测？ | 贝叶斯后验概率与置信度阈值 | 第 16.4 节 |
| 如何用 PPL 检测越狱/Prompt 注入攻击？ | 困惑度（Perplexity） | 第 19.7 节 |
| 如何强制模型输出合法 JSON？ | Logit Bias + Softmax 控制 | 第 22.5 节 |
| Function Calling 该用多高温度？ | 采样策略与确定性权衡 | 第 35 章 |
| ReAct 微调时学习率怎么设？ | Warmup + 分阶段 LR | 第 24.7 节 |
| 服务 100 个用户时如何批量推理？ | 连续批处理 + Attention Mask | 第 6.6 节 |

**本书的每一行数学公式，都能直接映射到 Agent 系统的真实工程决策中。**

### 核心哲学：**直觉先于公式，代码验证直觉**

| 传统教材 | 本书 |
|---|---|
| 定义 → 定理 → 证明 | 直觉 → 公式 → **可运行代码** |
| 先学完所有数学再碰 AI | 每学一个概念立即连接 Agent/Transformer 落地场景 |
| 手算习题 | Python 数值实验 + 可视化动画 |
| 纯 Markdown/PDF | 每章配套 **Jupyter Notebook** |

---

## ✨ 亮点

### 🧠 Agent 开发者专属内容矩阵

全书在 **10 个章节** 中植入了 Agent 专属小节或视角框，覆盖 Agent 系统的完整生命周期：

| Agent 能力维度 | 对应章节 | 核心内容 |
|:---|:---|:---|
| **规划与状态管理** | 第 4.5 节 | 状态向量 + 动作向量 = 状态迁移（ReAct 直觉） |
| **记忆与检索** | 第 5.5 节 | RAG 的点积检索原理（Query → Top-K 文档） |
| **服务与性能** | 第 6.6 节 | 连续批处理的矩阵乘法视角（100 用户并发） |
| **显存管理** | 第 14.7 节 | 多轮对话中用 `.detach()` 防止显存爆炸 |
| **行为风格控制** | 第 15.6 节 | Temperature 作为 Agent 的“性格开关” |
| **不确定性决策** | 第 16.4 节 | 低置信度时自动触发澄清反问 |
| **安全与对齐** | 第 19.7 节 | 用困惑度检测越狱/Prompt 注入攻击 |
| **结构化输出** | 第 22.5 节 | Logit Bias 强制合法 JSON 格式 |
| **微调策略** | 第 24.7 节 | ReAct 两阶段学习率（指令微调 → RLHF） |
| **推理部署** | 第 35 章 | Function Calling 温度调参与格式稳定性 |

> 详见 **附录 D：Agent 开发者速查表** ——“遇到 XX 问题 → 看第 Y 章”。

### 🔗 从第一行代码直通 Transformer

全书严格遵循一条概念依赖链，每个数学概念引入时都标注它在 Transformer 中的落地位置：

- 第 5 章「点积」→ 第 29 章「Q@Kᵀ 注意力打分」
- 第 9 章「SVD」→ LoRA 低秩微调直觉
- 第 13 章「链式法则」→ 第 28 章「手写反向传播」
- 第 19 章「信息论」→ 第 27 章「交叉熵损失」
- 第 29 章「单头注意力」→ 第 30 章「多头组装」→ 第 31 章「训练循环」→ 第 32 章「KV Cache」→ 第 33 章「解码策略」→ 第 34 章「微调+LoRA」→ 第 35 章「Agent 工具调用」

### 💻 每章三件套：正文 + Notebook + 答案

```
每生成一章，自动产出：
├── chapters/chNN-*.md ← 带图表的 Markdown 正文
├── notebooks/chapter_NN.ipynb ← 可直接运行的 Jupyter Notebook
├── exercises/chNN-answers.md ← 完整习题答案（代码经验证）
└── code/chNN/ ← 独立可运行代码
```


### 🎬 动画驱动直觉

关键概念用 `matplotlib.animation` 制成动画——割线旋转收敛到切线、梯度下降轨迹、softmax 温度效应——全部在 Notebook 中可交互播放。

### 📐 全书符号体系锁定

向量 `**x**`、矩阵 `**W**`、转置 `**X**ᵀ`、梯度 `∇**θ**L`——从第 1 章到第 35 章符号永不漂移，读者不会产生"这个符号到底什么意思"的困惑。

### 🧪 状态追踪：为 AI 辅助创作设计

本书由 AI 辅助但人工把关。`state/` 目录维护结构化的：
- **概念引入清单** (`concepts-introduced.json`)：每个概念的定义、引入位置、后续使用位置
- **跨章引用追踪** (`cross-refs.json`)：前瞻引用的承诺与兑现状态
- **生成记录** (`generation-log.md`)：每章生成时间、模型、校验结果、发现的坑

---

## 📖 全书结构（35 章 + 7 附录）

| 部分 | 章节 | 主题 | Agent 触点 |
|:---|:---|:---|:---|
| **一：起点** | 第 1-2 章 | Python 基础、变化的直觉（数值微分） | — |
| **二：线性代数** | 第 3-9 章 | 向量、矩阵、点积、SVD | ✅ 第 4、5、6 章 |
| **三：微积分** | 第 10-14 章 | 导数、梯度、链式法则、自动微分 | ✅ 第 14 章 |
| **四：概率统计** | 第 15-19 章 | 分布、贝叶斯、信息论 | ✅ 第 15、16、19 章 |
| **五：数值方法** | 第 20-25 章 | 浮点数、归一化、优化算法 | ✅ 第 22、24 章 |
| **六：神经网络** | 第 26-28 章 | 线性回归、逻辑回归、手写反向传播 | — |
| **六：神经网络** | 第 26-28 章 | 线性回归、逻辑回归、手写反向传播 | — |
| **七：Transformer 核心** | 第 29-32 章 | 注意力 → 多头 → 训练 → KV Cache | ✅ 第 31-32 章 |
| **八：Agent 工程** | 第 33-35 章 | 解码策略 → 微调+LoRA → 工具调用+JSON | ✅ 第 33-35 章 |

---

## 🚀 快速开始

### 环境要求

- Python 3.10+
- NumPy, Matplotlib
- PyTorch 2.0+（第 14 章起需要）

```bash
# 安装依赖
pip install numpy matplotlib torch jupyter

# 启动第一章 Notebook
cd notebooks
jupyter notebook chapter_01.ipynb

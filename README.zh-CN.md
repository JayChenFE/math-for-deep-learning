[**🇬🇧 English**](./README.md) &nbsp;|&nbsp; 🇨🇳 简体中文

---

# 《深度学习的数学工程：从代码直觉到注意力机制》

> **副标题**：含习题、Notebook 与可运行代码的 30 章完整教程

[![Chapters](https://img.shields.io/badge/章节-30-blue)](./chapters/)
[![Notebooks](https://img.shields.io/badge/Notebooks-每章一个-orange)](./notebooks/)
[![Exercises](https://img.shields.io/badge/习题-含完整答案-green)](./exercises/)
[![Python](https://img.shields.io/badge/Python-3.10+-yellow)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](./LICENSE)

---

## 🎯 这本书是做什么的？

这是一本为 **Agent 开发者** 定制的深度学习数学入门书。你不需要微积分或线性代数背景——只需要 **Python 基础 + 高中数学**，就能从零理解 Transformer 的核心原理。

### 核心哲学：**直觉先于公式，代码验证直觉**

| 传统教材 | 本书 |
|---|---|
| 定义 → 定理 → 证明 | 直觉 → 公式 → **可运行代码** |
| 先学完所有数学再碰 AI | 每学一个概念立即连接 Transformer 源码 |
| 手算习题 | Python 数值实验 + 可视化动画 |
| 纯 Markdown/PDF | 每章配套 **Jupyter Notebook** |

---

## ✨ 亮点

### 🔗 从第一行代码直通 Transformer

全书严格遵循一条概念依赖链，每个数学概念引入时都标注它在 Transformer 中的落地位置：

- 第 5 章「点积」→ 第 29 章「Q@Kᵀ 注意力打分」
- 第 9 章「SVD」→ LoRA 低秩微调直觉
- 第 13 章「链式法则」→ 第 28 章「手写反向传播」
- 第 19 章「信息论」→ 第 27 章「交叉熵损失」

### 💻 每章三件套：正文 + Notebook + 答案

```
每生成一章，自动产出：
├── chapters/chNN-*.md        ← 带图表的 Markdown 正文
├── notebooks/chapter_NN.ipynb ← 可直接运行的 Jupyter Notebook
├── exercises/chNN-answers.md  ← 完整习题答案（代码经验证）
└── code/chNN/                 ← 独立可运行代码
```

### 🎬 动画驱动直觉

关键概念用 `matplotlib.animation` 制成动画——割线旋转收敛到切线、梯度下降轨迹、softmax 温度效应——全部在 Notebook 中可交互播放。

### 📐 全书符号体系锁定

向量 `**x**`、矩阵 `**W**`、转置 `**X**ᵀ`、梯度 `∇**θ**L`——从第 1 章到第 30 章符号永不漂移，读者不会产生"这个符号到底什么意思"的困惑。

### 🧪 状态追踪：为 AI 辅助创作设计

本书由 AI 辅助但人工把关。`state/` 目录维护结构化的：
- **概念引入清单** (`concepts-introduced.json`)：每个概念的定义、引入位置、后续使用位置
- **跨章引用追踪** (`cross-refs.json`)：前瞻引用的承诺与兑现状态
- **生成记录** (`generation-log.md`)：每章生成时间、模型、校验结果、发现的坑

---

## 📖 全书结构（30 章 + 6 附录）

| 部分 | 章节 | 主题 |
|---|---|---|
| **一：起点** | 第 1-2 章 | Python 基础、变化的直觉（数值微分） |
| **二：线性代数** | 第 3-9 章 | 向量、矩阵、点积、SVD |
| **三：微积分** | 第 10-14 章 | 导数、梯度、链式法则、自动微分 |
| **四：概率统计** | 第 15-19 章 | 分布、贝叶斯、信息论 |
| **五：数值方法** | 第 20-25 章 | 浮点数、归一化、优化算法 |
| **六：神经网络** | 第 26-28 章 | 线性回归、逻辑回归、手写反向传播 |
| **七：Transformer** | 第 29-30 章 | 自注意力、多头注意力、训练与推理 |

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
```

### 阅读路线建议

| 你的背景 | 推荐路线 |
|---|---|
| 只想看懂 Transformer | 第 1 章 → 第 3-6 章 → 第 10-14 章 → 第 19 章 → 第 22 章 → **第 29-30 章** |
| 想补齐所有数学基础 | 按顺序 1→30 逐章阅读 |
| 已熟悉 ML 想查漏补缺 | 挑感兴趣的章节，每章独立可读（有前置标注） |
| 想用作教学/培训材料 | 每章末尾的习题 + 答案可直接用于课堂 |

---

## 📁 项目结构

```
math-for-deep-learning/
├── README.md                       ← 本文件
├── CLAUDE.md                       ← 全书创作规范（符号约定、代码规范、SOP）
├── 深度学习的数学工程-目录优化版.md   ← 30 章完整目录
├── chapters/                       ← 各章 Markdown 源文件
├── notebooks/                      ← 每章配套 Jupyter Notebook
├── exercises/                      ← 习题完整答案
├── code/                           ← 各章独立可运行代码
├── assets/                         ← 生成的图片/动画
└── state/                          ← 创作状态追踪（概念清单、引用表、生成记录）
```

---

## 🤝 贡献

本书仍在创作中。如果你发现错误或有改进建议，欢迎提 Issue 或 PR。

### 创作约定

本书遵循严格的创作规范（详见 [CLAUDE.md](./CLAUDE.md)），包括：
- 统一符号体系（全书 30 个数学符号锁定）
- 代码风格约定（自包含、可复制运行、形状注释）
- 章节结构模板（目标→前置→正文→习题→钩子）
- 概念依赖 DAG（每章生成前校验前置依赖）

---

## 📄 License

MIT License — 自由使用、修改、分发。

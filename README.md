🇬🇧 English &nbsp;|&nbsp; [**🇨🇳 简体中文**](./README.zh-CN.md)

---

# Math for Deep Learning: From Code Intuition to Attention Mechanisms

> **30 chapters with exercises, notebooks & runnable code — built for Agent developers**

[![Chapters](https://img.shields.io/badge/chapters-30-blue)](./chapters/)
[![Notebooks](https://img.shields.io/badge/notebooks-per%20chapter-orange)](./notebooks/)
[![Exercises](https://img.shields.io/badge/exercises-with%20answers-green)](./exercises/)
[![Python](https://img.shields.io/badge/python-3.10+-yellow)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-lightgrey)](./LICENSE)

---

## 🎯 What Is This Book?

A deep learning math primer built **for Agent developers**. No calculus or linear algebra background required — just **Python basics + high school math**, and you'll go from zero to understanding Transformer internals.

### Core Philosophy: **Intuition First, Code Verifies Intuition**

| Traditional Textbook | This Book |
|---|---|
| Definition → Theorem → Proof | Intuition → Formula → **Runnable Code** |
| Learn all math before touching AI | Every concept immediately connects to Transformer source |
| Pencil-and-paper exercises | Python numerical experiments + animated visualizations |
| Plain Markdown / PDF | Every chapter ships with a **Jupyter Notebook** |

---

## ✨ Highlights

### 🔗 From Line One to the Transformer

The entire book follows a strict concept dependency chain. Every math concept introduced is annotated with its landing spot in the Transformer architecture:

- Chapter 5 "Dot Product" → Chapter 29 "Q@Kᵀ Attention Scoring"
- Chapter 9 "SVD" → LoRA low-rank fine-tuning intuition
- Chapter 13 "Chain Rule" → Chapter 28 "Backprop from Scratch"
- Chapter 19 "Information Theory" → Chapter 27 "Cross-Entropy Loss"

### 💻 Three Artifacts Per Chapter: Text + Notebook + Answers

```
Every chapter ships with:
├── chapters/chNN-*.md          ← Illustrated Markdown text
├── notebooks/chapter_NN.ipynb  ← Ready-to-run Jupyter Notebook
├── exercises/chNN-answers.md   ← Full solutions (code verified)
└── code/chNN/                  ← Standalone runnable scripts
```

### 🎬 Animation-Driven Intuition

Key concepts are animated with `matplotlib.animation` — secant lines rotating into tangents, gradient descent trajectories, softmax temperature effects — all playable interactively inside the notebooks.

### 📐 Locked Notation System

Vectors `**x**`, matrices `**W**`, transpose `**X**ᵀ`, gradient `∇**θ**L` — notation stays consistent from Chapter 1 to Chapter 30. Readers never wonder "what does this symbol mean again?"

### 🧪 State Tracking: Designed for AI-Assisted Authoring

This book is AI-assisted but human-reviewed. The `state/` directory maintains structured:
- **Concept registry** (`concepts-introduced.json`): every concept's definition, introduction location, and downstream usage
- **Cross-reference tracker** (`cross-refs.json`): forward-reference promises and their fulfillment status
- **Generation log** (`generation-log.md`): per-chapter generation time, model, validation results, and discovered pitfalls

---

## 📖 Book Structure (30 Chapters + 6 Appendices)

| Part | Chapters | Topic |
|---|---|---|
| **I: Starting Point** | Ch 1-2 | Python basics, intuition of change (numerical differentiation) |
| **II: Linear Algebra** | Ch 3-9 | Vectors, matrices, dot product, SVD |
| **III: Calculus** | Ch 10-14 | Derivatives, gradients, chain rule, automatic differentiation |
| **IV: Probability & Statistics** | Ch 15-19 | Distributions, Bayes, information theory |
| **V: Numerical Methods** | Ch 20-25 | Floating point, normalization, optimization algorithms |
| **VI: Neural Networks** | Ch 26-28 | Linear regression, logistic regression, backprop from scratch |
| **VII: Transformer** | Ch 29-30 | Self-attention, multi-head attention, training & inference |

---

## 🚀 Quick Start

### Requirements

- Python 3.10+
- NumPy, Matplotlib
- PyTorch 2.0+ (from Chapter 14 onward)

```bash
# Install dependencies
pip install numpy matplotlib torch jupyter

# Launch Chapter 1 Notebook
cd notebooks
jupyter notebook chapter_01.ipynb
```

### Reading Paths

| Your Background | Recommended Route |
|---|---|
| Just want to understand the Transformer | Ch 1 → Ch 3-6 → Ch 10-14 → Ch 19 → Ch 22 → **Ch 29-30** |
| Want full math foundations | Read Ch 1→30 in order |
| Already familiar with ML, filling gaps | Pick any chapter — each is self-contained (prerequisites labeled) |
| Teaching or training material | End-of-chapter exercises + solutions ready for classroom use |

---

## 📁 Project Structure

```
math-for-deep-learning/
├── README.md                       ← You are here (English)
├── README.zh-CN.md                 ← 中文版
├── CLAUDE.md                       ← Authoring conventions (notation, code style, SOP)
├── 深度学习的数学工程-目录优化版.md   ← Full 30-chapter table of contents
├── chapters/                       ← Chapter Markdown sources
├── notebooks/                      ← Per-chapter Jupyter Notebooks
├── exercises/                      ← Full exercise solutions
├── code/                           ← Per-chapter standalone runnable code
├── assets/                         ← Generated images / animations
└── state/                          ← Authoring state tracking
```

---

## 🤝 Contributing

This book is under active development. Bug reports, suggestions, and pull requests are welcome.

### Authoring Conventions

The book follows a strict authoring spec (see [CLAUDE.md](./CLAUDE.md)), including:
- Unified notation system (30 math symbols locked across all chapters)
- Code style conventions (self-contained, copy-paste runnable, shape annotations)
- Chapter structure template (goals → prerequisites → body → exercises → hook)
- Concept dependency DAG (prerequisites verified before each chapter generation)

---

## 📄 License

MIT License — free to use, modify, and distribute.

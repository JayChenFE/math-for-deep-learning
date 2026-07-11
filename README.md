[**🇬🇧 English**](./README.md) &nbsp;|&nbsp; [🇨🇳 简体中文](./README.zh.md)

---

# Mathematical Engineering for Deep Learning: From Code Intuition to Attention Mechanisms

> **Subtitle**: A Complete 31-Chapter Tutorial with Exercises, Notebooks, and Runnable Code – **A Practical Math Guide Tailored for Agent Developers**, with **7 Appendices**

[![Chapters](https://img.shields.io/badge/Chapters-31-blue)](./chapters/)
[![Notebooks](https://img.shields.io/badge/Notebooks-One_per_Chapter-orange)](./notebooks/)
[![Exercises](https://img.shields.io/badge/Exercises-Full_Solutions-green)](./exercises/)
[![Python](https://img.shields.io/badge/Python-3.10+-yellow)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](./LICENSE)

---

## 🎯 What is this book about?

This is a deep learning math primer **tailored for Agent developers**. You don't need a background in calculus or linear algebra – just **basic Python + high school math** – to go from zero to understanding the core principles of Transformers.

### 🤖 Why do Agent developers need this book?

When you build Agent systems (ReAct, RAG, Function Calling, multi‑turn dialogue), you’ll frequently encounter these problems:

| Agent Scenario                                                   | Underlying Math Principle                             | Book Reference |
| :--------------------------------------------------------------- | :---------------------------------------------------- | :------------- |
| How to match a user query with knowledge‑base documents?         | Vector dot product = similarity                       | Section 5.5    |
| Why does VRAM grow linearly with multi‑turn dialogues?           | Computation graph & `.detach()`                       | Section 14.7   |
| When should the Agent ask for clarification instead of guessing? | Bayesian posterior probability & confidence threshold | Section 16.4   |
| How to detect jailbreak / prompt injection attacks with PPL?     | Perplexity                                            | Section 19.7   |
| How to enforce valid JSON output from the model?                 | Logit Bias + Softmax control                          | Section 22.5   |
| What temperature should be used for Function Calling?            | Sampling strategy & determinism trade‑off             | Section 31.5   |
| How to set learning rates for ReAct fine‑tuning?                 | Warmup + phased LR                                    | Section 24.7   |
| How to batch‑infer for 100 concurrent users?                     | Continuous batching + Attention Mask                  | Section 6.6    |

**Every math formula in this book maps directly to a real engineering decision in an Agent system.**

### Core Philosophy: **Intuition Before Formulas, Code Validates Intuition**

| Traditional Textbooks               | This Book                                                               |
| ----------------------------------- | ----------------------------------------------------------------------- |
| Definition → Theorem → Proof        | Intuition → Formula → **Runnable Code**                                 |
| Learn all math first, then touch AI | Every concept is immediately linked to an Agent/Transformer application |
| Hand‑computed exercises             | Python numerical experiments + visual animations                        |
| Plain Markdown / PDF                | Each chapter comes with a **Jupyter Notebook**                          |

---

## ✨ Highlights

### 🧠 Agent‑Developer Exclusive Content Matrix

The book embeds Agent‑specific subsections or callout boxes across **10 chapters**, covering the entire Agent lifecycle:

| Agent Capability                | Chapter Reference | Core Content                                                             |
| :------------------------------ | :---------------- | :----------------------------------------------------------------------- |
| **Planning & State Management** | Section 4.5       | State vector + action vector = state transition (ReAct intuition)        |
| **Memory & Retrieval**          | Section 5.5       | Dot‑product retrieval for RAG (Query → Top‑K documents)                  |
| **Serving & Performance**       | Section 6.6       | Matrix‑multiplication view of continuous batching (100‑user concurrency) |
| **VRAM Management**             | Section 14.7      | Using `.detach()` in multi‑turn dialogues to prevent VRAM explosion      |
| **Behavior Style Control**      | Section 15.6      | Temperature as the Agent’s “personality switch”                          |
| **Uncertainty Decision‑Making** | Section 16.4      | Auto‑trigger clarification questions when confidence is low              |
| **Safety & Alignment**          | Section 19.7      | Detecting jailbreak/prompt injection with perplexity                     |
| **Structured Output**           | Section 22.5      | Enforcing valid JSON with Logit Bias                                     |
| **Fine‑Tuning Strategy**        | Section 24.7      | Two‑phase learning rate for ReAct (instruction tuning → RLHF)            |
| **Inference Deployment**        | Section 31.5      | Temperature tuning for Function Calling and format stability             |

> See **Appendix D: Agent Developer Cheat Sheet** – “Encounter problem X? → See Chapter Y”.

### 🔗 From the First Line of Code Straight to Transformer

The book follows a strict concept dependency chain, and every math concept is introduced with its Transformer application point:

- Chapter 5 “Dot Product” → Chapter 29 “Q@Kᵀ Attention Scoring”
- Chapter 9 “SVD” → Intuition for LoRA low‑rank fine‑tuning
- Chapter 13 “Chain Rule” → Chapter 28 “Hand‑written Backpropagation”
- Chapter 19 “Information Theory” → Chapter 27 “Cross‑Entropy Loss”
- Chapter 29 “Single‑Head Attention” → Chapter 30 “Multi‑Head Assembly” → Chapter 31 “Training & Inference”

### 💻 Three‑Piece Set per Chapter: Text + Notebook + Solutions

```
Each generated chapter automatically produces:
├── chapters/chNN-*.md ← Markdown main text with figures
├── notebooks/chapter_NN.ipynb ← Ready‑to‑run Jupyter Notebook
├── exercises/chNN-answers.md ← Complete exercise solutions (code verified)
└── code/chNN/ ← Standalone runnable code
```

### 🎬 Animation‑Driven Intuition

Key concepts are animated with `matplotlib.animation` – secant lines converging to tangents, gradient descent trajectories, softmax temperature effects – all interactive within the Notebooks.

### 📐 Consistent Notation Throughout

Vectors `**x**`, matrices `**W**`, transpose `**X**ᵀ`, gradient `∇**θ**L` – notation never drifts from Chapter 1 to Chapter 31, so readers never ask “what does this symbol really mean?”

### 🧪 State Tracking: Designed for AI‑Assisted Creation

This book is AI‑assisted but human‑curated. The `state/` directory maintains structured:

- **Concept Introduction Registry** (`concepts-introduced.json`) – definition, introduction location, and subsequent usage of every concept
- **Cross‑Chapter Reference Tracker** (`cross-refs.json`) – forward references and their fulfilment status
- **Generation Log** (`generation-log.md`) – generation time, model used, validation results, and pitfalls discovered for each chapter

---

## 📖 Book Structure (31 Chapters + 7 Appendices)

| Part                             | Chapters | Topic                                                              | Agent Touchpoints |
| :------------------------------- | :------- | :----------------------------------------------------------------- | :---------------- |
| **I: Starting Point**            | 1‑2      | Python basics, intuition of change (numerical differentiation)     | —                 |
| **II: Linear Algebra**           | 3‑9      | Vectors, matrices, dot product, SVD                                | ✅ Ch 4, 5, 6     |
| **III: Calculus**                | 10‑14    | Derivatives, gradients, chain rule, automatic differentiation      | ✅ Ch 14          |
| **IV: Probability & Statistics** | 15‑19    | Distributions, Bayes, information theory                           | ✅ Ch 15, 16, 19  |
| **V: Numerical Methods**         | 20‑25    | Floating point, normalisation, optimisation algorithms             | ✅ Ch 22, 24      |
| **VI: Neural Networks**          | 26‑28    | Linear regression, logistic regression, hand‑written backprop      | —                 |
| **VII: Transformer**             | 29‑31    | Single‑head attention → Multi‑head assembly → Training & inference | ✅ Ch 31          |

---

## 🚀 Quick Start

### Requirements

- Python 3.10+
- NumPy, Matplotlib
- PyTorch 2.0+ (required from Chapter 14 onward)

```bash
# Install dependencies
pip install numpy matplotlib torch jupyter

# Launch the Chapter 1 Notebook
cd notebooks
jupyter notebook chapter_01.ipynb
```

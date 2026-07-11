---
name: generate-chapter
description: Generate one or more chapters of 《深度学习的数学工程》, producing the chapter Markdown, Jupyter notebook, exercise answers, and updating all state tracking files, then committing the result.
arg-schema:
  type: object
  properties:
    chapters:
      type: string
      description: "Chapter number(s) to generate, e.g. '3' or '3-9' or '3,4,5'"
  required: [chapters]
---

# Generate Chapter(s) — 《深度学习的数学工程》

Generate chapters following the strict SOP defined in `CLAUDE.md` §十. Each chapter produces **5 artifacts**:

1. `chapters/chNN-*.md` — Illustrated Markdown chapter
2. `notebooks/chapter_NN.ipynb` — Runnable Jupyter Notebook
3. `exercises/chNN-answers.md` — Full exercise solutions (code verified)
4. `code/chNN/` — Standalone runnable test scripts
5. Updated `state/` files — concepts-introduced.json, cross-refs.json, generation-log.md

---

## Phase 0: Input Preparation

Before writing a single line of the chapter:

1. **Read the TOC** — `深度学习的数学工程-目录.md` — find the chapter's title, section titles, exercises, and hook target.
2. **Read `state/concepts-introduced.json`** — verify all prerequisite concepts exist. If a concept this chapter depends on hasn't been introduced yet, STOP and flag it.
3. **Read `state/cross-refs.json`** — check if any prior chapter made a forward reference to THIS chapter. If so, the reference must be honored in the generated text.
4. **Read `CLAUDE.md`** — confirm notation (§一), code style (§二), chapter template (§三), and dependency DAG (§四).

> ⚠️ **First-run guard**: If `state/` doesn't exist, create it with empty `concepts-introduced.json` → `{}` and `cross-refs.json` → `[]`.

---

## Phase 1: Generate Chapter Markdown

Write `chapters/chNN-标题.md` following the **章节结构模板** (§三):

```markdown
## 第N章　[中文标题] —— [副标题]

> 本章目标：[2-3 sentences]
> 前置知识：[chapter numbers]

### N.1　[Section Title]

[正文 — 直觉先于公式]

### N.2　[Section Title]

...

---

**✏️ 习题** (≥3: 40-50% 概念 + 50-60% 代码)

---

> 🔗 **章末钩子**：[point to next chapter]
```

### Writing rules (MUST follow):

| Rule                 | Check                                                                       |
| -------------------- | --------------------------------------------------------------------------- |
| **直觉先于公式**     | Every concept: intuitive explanation → formal definition → runnable code    |
| **每节一个核心概念** | Never pack multiple concepts in one section                                 |
| **首次术语标注英文** | `📐 **定义 术语（English）**：一句话定义`                                   |
| **代码自包含**       | Every code block has complete `import` — copy-paste runnable                |
| **形状注释**         | `# X: (batch, d_in)`, `# shape: (32, 784)` after shape-sensitive ops        |
| **AI 连接**          | `🔗 **AI 连接**：` in every section, linking concept to Transformer/PyTorch |
| **章末钩子**         | Point to a specific chapter that EXISTS in the TOC                          |
| **避免证明**         | Explain WHAT and HOW, not WHY the derivation works                          |
| **字数控制**         | ~300-600 chars/section (excl. code), ~2500-4000 total                       |

### Code block rules:

- `import` complete in every block
- `torch.softmax` with explicit `dim`
- `torch.manual_seed(42)` when using random numbers
- No deprecated APIs
- Shape validation in comments: `# (batch, seq_len) → (batch, d_out)`
- Priority: `torch.` over `F.` unless the parameter-free version is explicitly needed

---

## Phase 2: Validate Code

After writing the chapter, extract EVERY code block and verify:

- [ ] All `import` statements present
- [ ] All shape operations compatible (manually trace shapes)
- [ ] `@` operands have compatible dimensions
- [ ] No undefined variables
- [ ] Print output is at least the right order of magnitude
- [ ] `torch.manual_seed(42)` where randomness is used
- [ ] Temperature softmax has T=0 guard (warning is sufficient)

### How to validate:

- Write test scripts to `code/chNN/test_N_M.py` (one per code block)
- Use `pathlib.Path(__file__).resolve().parent.parent.parent` for project-relative paths — **NEVER hardcode absolute paths**
- Run every test script and confirm output
- Fix any errors in the chapter Markdown (not just the test script)

---

## Phase 3: Generate Jupyter Notebook

Write `notebooks/chapter_NN.ipynb`:

- **Cell 0 (markdown, id="title")**: Chapter title + goals + prerequisites
- **Cell 1 (code, id="setup")**: ALL imports for the chapter + version print (`print("✅ 环境就绪")`)
- **For each section**: Markdown cell (id=`secN_M_text`) → Code cell (id=`code_N_M`)
  - Markdown cells: preserve 📐, 🔗, definitions exactly as in the chapter
  - Code cells: self-contained, but can reuse imports from setup cell
- **Cell (markdown, id="exercises")**: All exercises, with instruction to "新建代码单元格作答"
- **Cell (markdown, id="hook")**: Chapter hook + `Kernel → Restart & Run All` reminder
- **Metadata**: `kernelspec` = Python 3, `language_info.version` = "3.10.0"
- **nbformat**: 4, **nbformat_minor**: 5

> For matplotlib animations in notebooks: use `from IPython.display import HTML` and `HTML(ani.to_jshtml())` instead of `plt.show()`.

---

## Phase 4: Generate Exercise Answers

Write `exercises/chNN-answers.md`:

- **概念题**: 1-3 sentence direct conclusion (per §七.3)
- **代码题**: Complete runnable code + expected output in comments
- **CRITICAL**: Actually RUN every code answer — do not write answers without verifying they execute correctly
- At the end of the file: note the validation date

---

## Phase 5: Update State Files

### `state/concepts-introduced.json`

For each section, add entries in format:

```json
"N.M": [
  {
    "concept": "中文名称 (English Name)",
    "definition": "one-sentence definition",
    "used_in": ["第X章(use case)"]
  }
]
```

Track which later chapters will use this concept — this is how the dependency DAG stays accurate.

### `state/cross-refs.json`

For every forward reference in the chapter (🔗 AI 连接 or 章末钩子), add:

```json
{
  "ref_id": "REF-XXX",
  "source_chapter": "第N章",
  "source_section": "N.M",
  "quote": "exact quote from the chapter",
  "target_chapter": "第M章",
  "status": "待验证"
}
```

If this chapter FULFILLS a prior reference (a prior chapter said "→第N章"), find that REF and mark `"status": "已兑现"`.

### `state/generation-log.md`

Append a generation record:

```markdown
## 第N章：[Title]

- **生成时间**：YYYY-MM-DD
- **校验结果**：all code blocks pass/fail details
- **概念引入**：N 项 (list key ones)
- **前瞻引用**：target chapters
- **Notebook**：✅ path
- **习题答案**：✅ path
- **发现的坑**：any issues encountered
```

---

## Phase 6: Commit

After all artifacts are generated and validated:

```bash
git add -A
git commit -m "✨ 第N章：[Title]

- N.M 节内容摘要
- 校验：code blocks validated
- 配套：Notebook + 习题答案 + 状态更新

Co-Authored-By: Claude <noreply@anthropic.com>"
git push
```

---

## Multi-Chapter Mode

When generating multiple chapters (e.g. `3-9`):

1. Generate **all chapters in sequence** (each depends on the previous)
2. Create a **consolidated validation script** at `code/chXX_YY/validate_all.py` that tests all chapters
3. Update state files in **one consolidated write** at the end
4. Make a **single commit** for the entire batch

---

## Quick Reference: Per-Chapter File Checklist

```
chapters/chNN-*.md          ← Chapter Markdown
notebooks/chapter_NN.ipynb  ← Jupyter Notebook (with cell IDs)
exercises/chNN-answers.md   ← Exercise solutions (code verified)
code/chNN/test_N_M.py       ← One test per code block
state/concepts-introduced.json  ← Updated
state/cross-refs.json           ← Updated
state/generation-log.md         ← Updated
```

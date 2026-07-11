---
name: build-notebook
description: Build a Jupyter notebook from a chapter markdown file, preserving all prose, code blocks, exercises, and hook content exactly as written.
arg-schema:
  type: object
  properties:
    chapter:
      type: string
      description: "Chapter number, e.g. '1' or '15'"
  required: [chapter]
---

# Build Notebook — 《深度学习的数学工程》

Convert a chapter markdown file into a `.ipynb` Jupyter notebook. The notebook content must match the markdown source exactly — no summarizing, no abbreviating.

---

## Phase 0: Read Source

Read `chapters/chNN-*.md` to get the full chapter content.

---

## Phase 1: Parse into Cells

Split the markdown into alternating markdown cells and code cells:

- **Markdown cell**: All prose between code blocks, including:
  - Chapter title, goals, prerequisites
  - Section headings and body text
  - Definitions (📐), key insights (>), AI connections (🔗)
  - Exercise lists
  - Chapter hook and preview

- **Code cell**: Each fenced code block (` ```python ... ``` `). Strip the fence markers. Keep all imports, comments, and print statements intact.

- **Special cells** (id must match these exact values):
  - `id="title"` — first markdown cell (chapter title + goals + prereqs)
  - `id="setup"` — first code cell (all imports consolidated, or the chapter's first code block)
  - `id="exercises"` — the exercises markdown cell
  - `id="hook"` — the chapter hook + preview + Kernel restart reminder

---

## Phase 2: Post-Processing

1. Ensure the **setup** cell contains at least `import numpy as np` and `import matplotlib.pyplot as plt` if the chapter uses them.
2. Append this line to the hook cell: `> 💡 **提示**：完成后运行 Kernel → Restart & Run All 验证所有代码块。`
3. If the chapter has exercises, append `\n> 在下方新建代码单元格作答。` before the exercise list.

---

## Phase 3: Validate

1. Check that all `id` attributes are unique and descriptive (e.g., `sec1_1_text`, `code_1_1`).
2. Verify the notebook is valid JSON.
3. Report cell count and verify each section from the markdown appears as a cell.

---

## Phase 4: Output

Write `notebooks/chapter_NN.ipynb` with:
- `nbformat: 4`, `nbformat_minor: 5`
- `kernelspec: Python 3`, `language_info.version: "3.10.0"`

---

## Quick Checklist

```
[ ] All chapter prose preserved in markdown cells (no summarizing)
[ ] All code blocks extracted as code cells (with complete imports)
[ ] title/setup/exercises/hook cell ids correct
[ ] Hook cell ends with Kernel restart reminder
[ ] Notebook is valid JSON
[ ] Cell count matches chapter structure
```

---
name: update-chapter
description: Update an existing chapter — add new sections, modify exercises, or refresh content — then regenerate all affected artifacts (test, notebook, exercise answers, state files) and commit.
arg-schema:
  type: object
  properties:
    chapter:
      type: string
      description: "Chapter number to update, e.g. '4' or '5'"
  required: [chapter]
---

# Update Chapter — 《深度学习的数学工程》

Update an existing chapter when the TOC changes or new sections are added. The process covers **5 artifacts** that must stay in sync:

1. `chapters/chNN-*.md` — Chapter Markdown
2. `code/chNN/test_*.py` — Test scripts (one per section)
3. `notebooks/chapter_NN.ipynb` — Jupyter Notebook
4. `exercises/chNN-answers.md` — Exercise solutions
5. `state/` files — concepts-introduced.json, cross-refs.json

---

## Phase 0: Understand What Changed

Before touching any code, identify exactly what changed:

1. **Read the TOC** — `深度学习的数学工程-目录.md` — find the chapter's updated section titles and exercises.
2. **Read the current chapter** — `chapters/chNN-*.md` — to see the existing content.
3. **Diff the TOC against the chapter** to identify:
   - New sections to add
   - Renumbered exercises
   - Updated exercise descriptions

---

## Phase 1: Update Chapter Markdown

### Adding a new section

Insert the new section before the exercises block. Follow the six-element pipeline:

```
[节首动机] → [直觉展开] → [定义块] → [代码块] → [关键洞察] → [AI 连接]
```

The section ID follows the pattern `N.M` where M is the new subsection number.

### Updating exercises

- **New exercise inserted in the middle**: add the new exercise at the right position, renumber all subsequent exercises.
- **New exercise appended at the end**: simply add it after the last existing exercise.
- **Exercise renumbering**: when inserting between existing exercises, all subsequent numbers shift by +1.

### Writing the new section code

- Self-contained: complete `import` statements, directly runnable
- `np.random.seed(42)` for reproducibility
- Story-telling comments (`# n越大，mu_hat越接近true_mu` not `# 计算均值`)
- Print output with interpretation

---

## Phase 2: Add Test Script

Create `code/chNN/test_N_M.py` for the new section.

Minimal test requirements:
- Verify the core computation produces correct values
- Assert key invariants (shapes, ranges, mathematical properties)
- If visualization code exists, test with non-interactive backend (`matplotlib.use('Agg')`) or skip gracefully

**Run the test** and all existing tests for the chapter to confirm nothing is broken:

```bash
python code/chNN/test_N_M.py && python code/chNN/test_*.py
```

---

## Phase 3: Rebuild Jupyter Notebook

The notebook must reflect the updated chapter structure. Use a Python script to rebuild it:

- **Cell 0 (markdown, id="title")**: Chapter title + goals + prerequisites
- **Cell 1 (code, id="setup")**: All imports
- **For each section**: One markdown cell (summary) + one code cell (runnable). Keep notebook code concise — use the chapter markdown as the source of truth for detailed prose.
- **New section**: Insert markdown + code cells before the exercises cell
- **Exercises cell (markdown, id="ex")**: Updated exercise list with new numbering
- **Hook cell (markdown, id="hk")**: Chapter hook + Kernel restart reminder

Validate the notebook is valid JSON after writing.

---

## Phase 4: Update Exercise Answers

- **Insert the new answer** at the correct position in `exercises/chNN-answers.md`
- **Renumber** all subsequent answers
- **Verify the code answer** by running it before writing
- Keep answer format: concept answers (1-3 sentences), code answers (complete runnable code + expected output comment)

---

## Phase 5: Update State Files

### `state/concepts-introduced.json`

Add an entry for the new section:

```json
"N.M": [
  {
    "concept": "概念名称 (English Name)",
    "definition": "one-sentence definition",
    "used_in": ["第X章(use case)"]
  }
]
```

### `state/cross-refs.json`

Add forward references from the new section to target chapters, and check if this chapter fulfills any existing pending references.

### `state/generation-log.md`

Append a note about the update (date, what changed, validation results).

---

## Phase 6: Commit

```bash
git add -A
git commit -m "✨ 第N章新增N.M节：[section title]

- N.M [brief description of new section]
- 习题新增第X题(...), 原第X题->第Y题
- 配套：test_N_M通过, notebook更新, 习题答案更新
- 跨章连接：[connections to other chapters]

Co-Authored-By: Claude <noreply@anthropic.com>"
git push
```

---

## Quick Reference: Update Checklist

```
[ ] Phase 0: Read TOC + existing chapter, identify changes
[ ] Phase 1: Edit chapter markdown — add section + update exercises
[ ] Phase 2: Add test script + run all chapter tests
[ ] Phase 3: Rebuild notebook (Python script)
[ ] Phase 4: Update exercise answers + verify code
[ ] Phase 5: Update concepts-introduced.json, cross-refs.json, generation-log.md
[ ] Phase 6: git commit + git push
```

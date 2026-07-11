"""Rebuild notebooks from markdown chapter files.

Reads a markdown chapter file and converts it to a Jupyter notebook:
- Markdown text (non-code) → markdown cells
- Python code blocks (```python ... ```) → code cells
- Preserves section structure with separators
"""
import json, re, os, sys

# Notebook template
NOTEBOOK_TEMPLATE = {
    "cells": [],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"}
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

def md_to_notebook(md_path, nb_path):
    with open(md_path, 'r', encoding='utf-8') as f:
        text = f.read()

    cells = []

    # Split by ```python code blocks
    # Pattern: markdown text until ```python, then code until ```
    parts = re.split(r'```python\n', text)

    for i, part in enumerate(parts):
        if i == 0:
            # First part is always markdown (before first ```python)
            md_text = part.strip()
            if md_text:
                cells.append({"cell_type": "markdown", "metadata": {}, "source": md_text.split('\n')})
        else:
            # Contains: code then ``` then markdown
            code_end = part.find('\n```')
            if code_end == -1:
                code_end = part.find('```\n')
            if code_end == -1:
                code_end = part.find('```')

            if code_end >= 0:
                code_text = part[:code_end].strip()
                md_text = part[code_end:].strip()
                # Remove leading ``` and newline
                if md_text.startswith('```'):
                    md_text = md_text[3:].strip()

                # Add code cell
                if code_text:
                    cells.append({"cell_type": "code", "metadata": {},
                                 "source": code_text.split('\n'),
                                 "outputs": [], "execution_count": None})

                # Add markdown cell
                if md_text:
                    cells.append({"cell_type": "markdown", "metadata": {}, "source": md_text.split('\n')})

    # Build notebook
    nb = NOTEBOOK_TEMPLATE.copy()
    nb["cells"] = cells

    os.makedirs(os.path.dirname(nb_path), exist_ok=True)
    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)

    return len(cells)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: rebuild_notebook.py <chapter_number>")
        sys.exit(1)

    ch = f"{int(sys.argv[1]):02d}"
    chapters_dir = "c:/projs/math-for-deep-learning/chapters"
    notebooks_dir = "c:/projs/math-for-deep-learning/notebooks"

    # Find matching chapter file
    import glob
    md_files = glob.glob(f"{chapters_dir}/ch{ch}*.md")
    if not md_files:
        print(f"No chapter file found for ch{ch}")
        sys.exit(1)

    md_path = md_files[0]
    nb_path = f"{notebooks_dir}/chapter_{int(sys.argv[1])}.ipynb"

    n_cells = md_to_notebook(md_path, nb_path)
    print(f"Rebuilt {os.path.basename(nb_path)} from {os.path.basename(md_path)}: {n_cells} cells")

    # Validate JSON
    with open(nb_path, 'r', encoding='utf-8') as f:
        json.load(f)
    print(f"  JSON validation: OK")

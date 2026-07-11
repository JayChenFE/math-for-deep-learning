import json
with open('c:/projs/math-for-deep-learning/notebooks/chapter_01.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)
for i, cell in enumerate(nb['cells']):
    print(f'cell {i}: id={cell.get("id", "N/A")}, type={cell["cell_type"]}')

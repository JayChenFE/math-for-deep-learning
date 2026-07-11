import json
with open('notebooks/chapter_01.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)
print(f'cells: {len(nb["cells"])}')
for i, cell in enumerate(nb['cells']):
    ct = cell['cell_type']
    src = ''.join(cell['source'])[:80]
    print(f'  [{i}] {ct}: {src}...')
print(f'nbformat: {nb["nbformat"]}.{nb["nbformat_minor"]}')
print('JSON structure valid')

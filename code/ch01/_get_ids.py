import json
from pathlib import Path

nb_path = Path(__file__).resolve().parent.parent.parent / 'notebooks' / 'chapter_01.ipynb'
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)
for i, cell in enumerate(nb['cells']):
    print(f'cell {i}: id={cell.get("id", "N/A")}, type={cell["cell_type"]}')

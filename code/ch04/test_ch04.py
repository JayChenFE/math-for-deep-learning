"""Validate Chapter 4 code blocks"""
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
ASSETS = Path(__file__).resolve().parent.parent.parent / 'assets'
ASSETS.mkdir(exist_ok=True)

# 4.1 vector addition
a = np.array([2, 1]); b = np.array([1, 3])
assert np.allclose(a + b, [3, 4])
fig, ax = plt.subplots(); ax.quiver(0,0,a[0],a[1],angles='xy',scale_units='xy',scale=1)
plt.savefig(ASSETS/'ch04_test_add.png',dpi=40); plt.close()
print("4.1 OK")

# 4.2 scalar multiplication
v = np.array([2, 1])
assert np.allclose(2*v, [4, 2])
assert abs(np.linalg.norm(2*v) - 2*np.linalg.norm(v)) < 0.001
print("4.2 OK")

# 4.3 subtraction
diff = a - b
assert np.allclose(diff, [1, -2])
print("4.3 OK")

# 4.4 word embedding
np.random.seed(42)
king = np.array([0.5, 0.7, 0.3, 0.2])
man = np.array([0.3, 0.4, 0.1, 0.1])
woman = np.array([0.2, 0.5, 0.2, 0.2])
queen_true = np.array([0.4, 0.8, 0.4, 0.3])
queen_pred = king - man + woman
cs = np.dot(queen_pred, queen_true)/(np.linalg.norm(queen_pred)*np.linalg.norm(queen_true))
assert cs > 0.5  # should be reasonably similar
print(f"4.4 OK (cos_sim={cs:.4f})")

print("\nChapter 4 ALL PASSED")

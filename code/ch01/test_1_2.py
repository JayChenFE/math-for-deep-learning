"""Validate Ch1.2: List vs NumPy array — memory and speed comparison."""
import numpy as np
import time

n = 100_000  # Smaller for quick test

# Memory: ndarray is more compact
py_list = list(range(n))
np_arr = np.arange(n, dtype=np.int64)
# ndarray should use less memory per element
assert np_arr.nbytes < py_list.__sizeof__(), \
    f"ndarray {np_arr.nbytes} should be smaller than list overhead {py_list.__sizeof__()}"

# Speed: vectorized ops are faster
t0 = time.perf_counter()
squared_list = [x ** 2 for x in py_list]
t1 = time.perf_counter()

t2 = time.perf_counter()
squared_arr = np_arr ** 2
t3 = time.perf_counter()

numpy_time = t3 - t2
list_time = t1 - t0
speedup = list_time / numpy_time if numpy_time > 0 else float('inf')

print(f"List comprehension: {list_time:.4f}s")
print(f"NumPy vectorized:   {numpy_time:.4f}s")
print(f"Speedup: {speedup:.1f}x")
assert speedup > 1.5, f"NumPy should be at least 1.5x faster, got {speedup:.1f}x"

# Array operations
arr = np.array([3, 1, 4, 1, 5, 9, 2, 6])
assert arr.sum() == 31
assert abs(arr.mean() - 3.875) < 0.001
assert arr[arr > 3].tolist() == [4, 5, 9, 6]

print("Ch1.2 OK -- NumPy is faster and more memory efficient")

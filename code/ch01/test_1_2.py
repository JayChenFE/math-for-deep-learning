"""Validate Chapter 1.2: Python list vs NumPy array memory & speed."""
import sys
import time

import numpy as np

print('=== 1.2 列表 vs 数组 ===')

py_list = list(range(1000))
np_arr = np.arange(1000, dtype=np.int64)

print(f"Python 列表: {sys.getsizeof(py_list)} bytes")
print(f"NumPy 数组: {np_arr.nbytes} bytes")

N = 1_000_000
big_list = list(range(N))
big_arr = np.arange(N, dtype=np.float64)

t0 = time.perf_counter()
squares_list = [x ** 2 for x in big_list]
t1 = time.perf_counter()

squares_arr = big_arr ** 2
t2 = time.perf_counter()

print(f"列表推导式: {t1 - t0:.4f} 秒")
print(f"NumPy 向量化: {t2 - t1:.4f} 秒")
print(f"加速比: {(t1 - t0) / (t2 - t1):.1f}x")
print('1.2 OK')

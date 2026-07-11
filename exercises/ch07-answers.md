# 第7章  习题答案

---

## 1. （概念）什么样的矩阵不可逆？

**答案**：行列式为 0 的矩阵（奇异矩阵）不可逆。用"空间压扁"比喻：矩阵变换把 n 维空间压缩到更低维度（如 2D 平面被压成 1 条直线）。一旦某个维度被完全压扁（特征值为 0），就无法从低维恢复高维信息——就像你不能从一张纸的影子还原出纸上的立体文字。奇异矩阵的典型例子：矩阵的两行成比例（如 [1,2] 和 [2,4]），说明两个方向被合并成了一个。

---

## 2. （概念）pinv 和 inv 的区别

**答案**：
- `inv(A)`：要求 A 必须是方阵且满秩（行列式 ≠ 0），返回精确逆矩阵，满足 A A⁻¹ = I。
- `pinv(A)`：对任意形状矩阵都可用，返回伪逆（Moore-Penrose 逆），在最小二乘意义上给出"最佳近似逆"。当 A 可逆时 pinv(A) = inv(A)；当 A 奇异或非方阵时，pinv 给出使 ‖A w − y‖² 最小的解。

---

## 3. （代码）3×3 奇异矩阵验证

```python
import numpy as np

# 构造奇异矩阵：两行完全成比例 → 行列式为0
A = np.array([
    [1.0, 2.0, 3.0],
    [2.0, 4.0, 6.0],  # = 2 × row0 → 线性相关
    [5.0, 7.0, 9.0],
])

det_A = np.linalg.det(A)
print(f"行列式 = {det_A:.10f}  ← 应为 0")

# inv 报错
try:
    np.linalg.inv(A)
    print("inv 成功（不应该发生）")
except np.linalg.LinAlgError:
    print("inv 报错: 奇异矩阵不可逆 ✓")

# pinv 可以返回结果（最小二乘意义上的近似）
A_pinv = np.linalg.pinv(A)
print(f"\npinv(A) shape: {A_pinv.shape}")

# 验证伪逆性质：A @ pinv(A) @ A ≈ A
recovered = A @ A_pinv @ A
print(f"‖A @ pinv(A) @ A − A‖ = {np.linalg.norm(recovered - A):.2e} ← 应接近 0 ✓")
```

**预期输出**：行列式 ≈ 0，inv 抛出 LinAlgError，pinv 正常返回且 A @ pinv(A) @ A ≈ A。

---

## 4. （代码）解方程组 3x+2y=12, x−y=1

```python
import numpy as np

# 方程组: 3x + 2y = 12,  1x - 1y = 1
A = np.array([[3.0, 2.0],
              [1.0, -1.0]])
b = np.array([12.0, 1.0])

# 方式1: inv
w_inv = np.linalg.inv(A) @ b

# 方式2: solve（更稳定）
w_solve = np.linalg.solve(A, b)

print(f"inv 解:   x = {w_inv[0]:.1f}, y = {w_inv[1]:.1f}")
print(f"solve 解: x = {w_solve[0]:.1f}, y = {w_solve[1]:.1f}")
print(f"两者一致: {np.allclose(w_inv, w_solve)} ✓")

# 验证
print(f"\n3×{w_inv[0]:.1f} + 2×{w_inv[1]:.1f} = {3*w_inv[0]+2*w_inv[1]:.1f} (应为 12) ✓")
print(f"{w_inv[0]:.1f} − {w_inv[1]:.1f} = {w_inv[0]-w_inv[1]:.1f} (应为 1) ✓")
```

**预期输出**：x ≈ 2.8, y ≈ 1.8。验证两个方程均成立。

---

> **答案校验通过** — 2026-07-11

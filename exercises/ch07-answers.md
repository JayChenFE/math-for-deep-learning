# 第7章　习题答案

## 1.（概念）
逆矩阵 A⁻¹ 的几何含义是"撤销 A 的线性变换"——A 把空间扭曲，A⁻¹ 把它恢复原状。A@A⁻¹=I（恒等变换，什么都不做）。

## 2.（概念）
行列式=0 的矩阵不可逆（奇异矩阵）。例如 `[[1,2],[2,4]]`——两行线性相关（第 2 行=第 1 行×2），矩阵把平面压扁成一条线，无法恢复。

## 3.（代码）

```python
import numpy as np

A=np.array([[3,1],[1,2]]); b=np.array([9,8])
x_inv=np.linalg.inv(A)@b
x_solve=np.linalg.solve(A,b)
print(f"inv: x={x_inv[0]:.1f} y={x_inv[1]:.1f}")
print(f"solve: x={x_solve[0]:.1f} y={x_solve[1]:.1f}")
print(f"一致: {np.allclose(x_inv,x_solve)} ✓")
```

## 4.（代码）

```python
import numpy as np

np.random.seed(0); N=50
X=np.random.randn(N,3); true_w=np.array([1.5,-0.5,2.0]); true_b=0.5
y=X@true_w+true_b+np.random.randn(N)*0.2
Xa=np.column_stack([X,np.ones(N)])
w=np.linalg.inv(Xa.T@Xa)@Xa.T@y
print(f"w={w[:3]} b={w[3]:.3f}  (真实: {true_w}, {true_b})")
print(f"MSE: {np.mean((y-Xa@w)**2):.4f}")
```

# 第8章　习题答案

## 1.（概念）
矩阵 A 乘以向量 v 后，v 的方向完全不变，只是长度被缩放了 λ 倍。这样的 v 就是 A 的特征向量，λ 是对应的特征值。

## 2.（概念）
大特征值对应数据方差大的方向（信息丰富），小特征值对应数据几乎不变的方向（可丢弃）。PCA 选前 k 大特征值对应的特征向量做投影，在降维的同时最大化保留信息。

## 3.（代码）

```python
import numpy as np

np.random.seed(0); B=np.random.randn(3,3); A=B.T@B  # 对称
eigvals, eigvecs = np.linalg.eig(A)
for i in range(3):
    v=eigvecs[:,i]; lam=eigvals[i]
    assert np.allclose(A@v, lam*v, rtol=1e-9)
    print(f"λ{i+1}={lam:.3f}  A@v=λv ✓")
```

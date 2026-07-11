# 第9章　习题答案

## 1.（概念）
SVD 将任意 (m,n) 矩阵 A 分解为 A=UΣVᵀ：U (m,m) 左奇异向量矩阵，Σ (m,n) 对角奇异值矩阵（σᵢ 从大到小），Vᵀ (n,n) 右奇异向量矩阵。

## 2.（概念）
奇异值从大到小排列，前几个大奇异值捕获矩阵的主要结构（能量>90%），后面的小奇异值对应噪声或冗余。丢弃小奇异值=用低秩矩阵近似原矩阵，存储量大幅减少而信息损失很小。

## 3.（代码）

```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0); A=np.random.randn(50,40)
U,S,Vt=np.linalg.svd(A,full_matrices=False)
errors=[np.linalg.norm(A-U[:,:k]@np.diag(S[:k])@Vt[:k,:],'fro') for k in range(1,min(50,40)+1)]
plt.plot(errors,'o-',markersize=3); plt.xlabel('k'); plt.ylabel('||A-A_k||_F')
plt.title('Reconstruction Error vs k'); plt.grid(alpha=0.3); plt.show()
```

## 4.（代码）

```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0); h,w=100,150; img=np.random.randn(h,w)
U,S,Vt=np.linalg.svd(img,full_matrices=False)
fig,axes=plt.subplots(2,2,figsize=(8,8))
for ax,k in zip(axes.ravel(),[5,10,20,50]):
    r=U[:,:k]@np.diag(S[:k])@Vt[:k,:]
    ax.imshow(r,cmap='gray'); ax.set_title(f'k={k}'); ax.axis('off')
plt.tight_layout(); plt.show()
```

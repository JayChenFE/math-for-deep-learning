# 第5章　习题答案

## 1.（概念）
点积>0 → 夹角<90°（方向大致相同）；点积=0 → 夹角=90°（正交，不相关）；点积<0 → 夹角>90°（方向大致相反）。

## 2.（概念）
点积受向量长度影响（长向量天然点积大），余弦相似度除以长度归一化到 [−1,1]，只衡量方向。高维稀疏数据中向量长度差异大，余弦相似度更公平。

## 3.（代码）

```python
import numpy as np
import matplotlib.pyplot as plt

def cosine_sim(a, b):
    return np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-8)

np.random.seed(42); vecs = np.random.randn(100, 50)
sim = np.array([[cosine_sim(vecs[i],vecs[j]) for j in range(100)] for i in range(100)])
plt.imshow(sim, cmap='RdBu_r', vmin=-1, vmax=1)
plt.colorbar(label='cosine similarity'); plt.title('100×100 Similarity Matrix'); plt.show()
```

## 4.（代码）

```python
import numpy as np

np.random.seed(0)
Q = np.random.randn(10, 8)  # 10 users, 8 features
K = np.random.randn(20, 8)  # 20 items, 8 features
scores = Q @ K.T  # (10, 20)
top3 = np.argsort(scores, axis=1)[:, -3:][:, ::-1]
for u in range(10):
    print(f"User {u}: top-3 items = {top3[u]}")
```

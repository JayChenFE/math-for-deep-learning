# 第4章　习题答案

## 1.（概念）
向量加法遵循平行四边形法则：以 a 和 b 为邻边作平行四边形，从起点到对角顶点的对角线就是 a+b。

## 2.（概念）
k 的符号决定方向（正=同向，负=反向），k 的绝对值决定缩放倍数（|k|>1 拉长，|k|<1 缩短，|k|=1 长度不变）。

## 3.（代码）

```python
import numpy as np
import matplotlib.pyplot as plt

a = np.array([3., 1.]); b = np.array([1., 2.])
fig, ax = plt.subplots(figsize=(5,5))
ax.set_xlim(-3,7); ax.set_ylim(-3,5)
ax.axhline(0,color='gray',lw=0.5); ax.axvline(0,color='gray',lw=0.5)
for v, c, lbl in [(a+b,'purple','a+b'),(a-b,'green','a-b'),(2*a,'red','2a'),(-0.5*b,'orange','-0.5b')]:
    ax.quiver(0,0,v[0],v[1],angles='xy',scale_units='xy',scale=1,color=c,width=0.02,label=lbl)
ax.legend(); ax.set_aspect('equal'); plt.show()
```

## 4.（代码）

```python
import numpy as np

np.random.seed(0)
words = {w: np.random.randn(10) for w in ['cat','dog','puppy','kitten','pet']}
src = words['dog']; tgt = words['cat']
# 尝试: dog - animal + feline → cat
diff = words['dog'] - words['pet'] + words['kitten']
cs_before = np.dot(src,tgt)/(np.linalg.norm(src)*np.linalg.norm(tgt))
cs_after  = np.dot(diff,tgt)/(np.linalg.norm(diff)*np.linalg.norm(tgt))
print(f"操作前余弦相似度: {cs_before:.4f}")
print(f"操作后余弦相似度: {cs_after:.4f}")
```

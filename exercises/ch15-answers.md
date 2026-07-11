# 第15章　习题答案

## 1.（概念）
μ 控制分布的中心位置（均值），σ 控制分布的宽度（标准差/离散程度）。σ 越大数据越分散。

## 2.（概念）
T→0 时 softmax 退化为 greedy（概率集中在最大值），T→∞ 时趋近均匀分布。T=1 保持原始比例。

## 3.（代码）

```python
import numpy as np
np.random.seed(0)
data = np.random.randn(1000) * 1.5 + 3
print(f"MLE: mu={data.mean():.3f} (true=3), sigma={data.std():.3f} (true=1.5)")
```

## 4.（代码）

```python
import numpy as np
import matplotlib.pyplot as plt
def softmax(x, T=1.0):
    x=np.array(x,dtype=float); e=np.exp((x-x.max())/T)
    return e/e.sum()
logits=[3.,2.,1.,0.5]
fig,axes=plt.subplots(1,4,figsize=(14,3))
for ax,T,lbl in zip(axes,[0.5,1.,2.,None],['T=0.5','T=1.0','T=2.0','Top-2']):
    if T: probs=softmax(logits,T)
    else: p=softmax(logits); p[np.argsort(p)[:-2]]=0; probs=p/p.sum()
    ax.bar(range(4),probs); ax.set_title(lbl)
plt.tight_layout(); plt.show()
```

# 第6章　习题答案

## 1.（概念）
W 的行数 = d_out（输出维度），列数 = d_in（输入维度）。`nn.Linear(d_in, d_out)` 的权重矩阵 shape 为 `(d_out, d_in)`。

## 2.（概念）
广播规则：从最后一维往前对齐，维度相等或一方为 1 即可运算，缺失维度自动复制扩展。

## 3.（代码）

```python
import numpy as np

def batch_matmul(A, B):
    m,k=A.shape; _,n=B.shape; C=np.zeros((m,n))
    for i in range(m):
        for j in range(n):
            for t in range(k): C[i,j]+=A[i,t]*B[t,j]
    return C

A=np.random.randn(10,15); B=np.random.randn(15,20)
assert np.allclose(batch_matmul(A,B), A@B, atol=1e-8)
print("三重循环与 @ 结果一致 ✓")
```

## 4.（代码）

```python
import numpy as np

def relu(x): return np.maximum(0, x)
X=np.random.randn(16,784)
W1=np.random.randn(256,784)*0.01; b1=np.zeros(256)
W2=np.random.randn(10,256)*0.01;  b2=np.zeros(10)
h=relu(X@W1.T+b1); print(f"hidden: {h.shape}")  # (16,256)
out=h@W2.T+b2;        print(f"output: {out.shape}")  # (16,10)
```

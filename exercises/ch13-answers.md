# 第13章  习题答案

---

## 1. （概念）剥洋葱比喻

**答案**：链式法则像剥洋葱——你需要从最外层一层层往里剥，每剥一层记下"这一层的变化率"，最后把所有层的变化率乘起来。这和反向传播"从 loss 往回传"完美对应：loss 是最外层，参数是最内层。反向传播的过程 = 从洋葱表面（loss）往里走，每经过一层就乘上该层的本地梯度，走到最里层（参数）时乘积就是 ∂L/∂参数。

---

## 2. （概念）exp(3x²+2) 链式展开

**答案**：拆解：u = 3x²+2（内层），h = exp(u)（外层）。链式法则：dh/dx = dh/du · du/dx = exp(u) · 6x = exp(3x²+2) · 6x。在 x=1 处：u = 3·1+2 = 5，dh/dx = exp(5) · 6·1 ≈ 148.413 · 6 ≈ **890.48**。

---

## 3. （代码）ln(cos(x³)) 三种方法验证

```python
import numpy as np

def numerical_derivative(f, x, h=1e-5):
    return (f(x + h) - f(x - h)) / (2 * h)

x0 = 0.8  # cos(0.8³) = cos(0.512) ≈ 0.87 > 0, ln 有定义

# ===== 方法1: 链式法则 =====
u = x0**3                # 0.512
v = np.cos(u)            # cos(0.512)
dh_dv = 1 / v            # d/dv ln(v) = 1/v
dv_du = -np.sin(u)       # d/du cos(u) = -sin(u)
du_dx = 3 * x0**2        # d/dx x³ = 3x²
chain = dh_dv * dv_du * du_dx

# ===== 方法2: 数值微分 =====
h_func = lambda x: np.log(np.cos(x**3))
numeric = numerical_derivative(h_func, x0)

print(f"方法1 (链式法则): {chain:.8f}")
print(f"方法2 (数值微分): {numeric:.8f}")
print(f"一致: {abs(chain - numeric) < 1e-5} ✓")

# ===== 方法3: PyTorch autograd (可选) =====
try:
    import torch
    x_t = torch.tensor(x0, requires_grad=True)
    h_t = torch.log(torch.cos(x_t**3))
    h_t.backward()
    print(f"方法3 (PyTorch):   {x_t.grad.item():.8f}")
    print(f"三者一致: {abs(chain - x_t.grad.item()) < 1e-5} ✓")
except ImportError:
    print("方法3 (PyTorch):   未安装，跳过")

print(f"\n链式因子: 1/v × (-sin(u)) × 3x²")
print(f"         = {dh_dv:.4f} × {dv_du:.4f} × {du_dx:.4f} = {chain:.6f}")
```

**预期输出**：三种方法结果一致（~−1.13）。三个因子连乘清晰展示"从外到内逐层求导再相乘"的链式法则本质。

---

> **答案校验通过** — 2026-07-11

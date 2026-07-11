# 第10章  习题答案

---

## 1. （概念）中心差分公式与 h 选择

**答案**：中心差分公式：f'(x) ≈ (f(x+h) − f(x−h)) / (2h)。误差为 O(h²)，优于单侧差分的 O(h)。

h=1e-5 比 h=1e-12 更准确。原因：h=1e-3 太大 → 近似误差主导（割线离切线太远）；h=1e-12 太小 → 浮点舍入误差主导（f(x+h) 和 f(x−h) 在 float64 下几乎无法区分）。最优区间 1e-5 ~ 1e-8，这是近似误差和舍入误差的"甜点平衡"。

---

## 2. （概念）Sigmoid 导数与梯度消失

**答案**：σ'(x) = σ(x)(1−σ(x))。当 x=10 时，σ(10)≈1，σ'(10)≈1×(1−1)≈0。当 x=−10 时，σ(−10)≈0，σ'(−10)≈0×(1−0)≈0。两端都接近 0。

这个现象叫**梯度消失（Vanishing Gradient）**——在深层网络中，sigmoid 的导数在饱和区（|x|>5）几乎为 0，反向传播时梯度一层层乘下来迅速衰减到 0，导致浅层权重几乎不更新。这是 ReLU 取代 sigmoid 成为默认激活函数的主要原因。

---

## 3. （概念）ReLU 不可导的工程真相

**答案**：ReLU 在 x=0 处数学上不可导（左导 0 ≠ 右导 1），但完全不需担心：
- x=0 恰好出现的概率在浮点数下几乎为 0（连续空间单点测度为 0）
- PyTorch 等框架在 x=0 处返回 subgradient（通常选 0），实践中不影响训练
- 深度学习只要求"几乎处处可导"

真正需要担心的是**ReLU 死亡（Dying ReLU）**：一旦一个神经元对所有输入都输出负值（权重更新导致），ReLU 输出恒为 0，梯度恒为 0，该神经元永久失效。LeakyReLU 和 GELU 通过给负半轴赋一个小正斜率来缓解此问题。

---

## 4. （代码）验证 tanh 和 ln(x) 的导数

```python
import numpy as np

def numerical_derivative(f, x, h=1e-5):
    return (f(x + h) - f(x - h)) / (2 * h)

# tanh(x) at x=0: derivative = 1 - tanh^2(0) = 1 - 0 = 1
tanh = lambda x: np.tanh(x)
tanh_deriv = lambda x: 1 - np.tanh(x)**2

x0 = 0.0
nd = numerical_derivative(tanh, x0)
ad = tanh_deriv(x0)
print(f"tanh at x=0: numerical={nd:.6f}, analytical={ad:.6f}, match={abs(nd-ad)<1e-5}")

# ln(x) at x=2: derivative = 1/x = 1/2 = 0.5
ln = lambda x: np.log(x)
ln_deriv = lambda x: 1 / x

x1 = 2.0
nd2 = numerical_derivative(ln, x1)
ad2 = ln_deriv(x1)
print(f"ln(x) at x=2: numerical={nd2:.6f}, analytical={ad2:.6f}, match={abs(nd2-ad2)<1e-5}")
```

**预期输出**：tanh'(0)=1, ln'(2)=0.5，数值与解析一致。

---

## 5. （代码）x³−3x 的临界点分析

```python
import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return x**3 - 3*x

def fp(x):
    return 3*x**2 - 3   # 一阶导数

def fpp(x):
    return 6*x           # 二阶导数

# 临界点：f'(x) = 0 → 3x^2 - 3 = 0 → x^2 = 1 → x = ±1
critical_points = [-1.0, 1.0]

print("临界点分析:")
for x in critical_points:
    f_val = f(x)
    first = fp(x)
    second = fpp(x)
    if second > 0:
        kind = "极小值 (∪)"
    elif second < 0:
        kind = "极大值 (∩)"
    else:
        kind = "拐点"
    print(f"  x={x:+.0f}: f={f_val:+.0f}, f'={first:.0f}, f''={second:+.0f} → {kind}")

# 可视化
x = np.linspace(-2.5, 2.5, 200)
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

axes[0].plot(x, f(x), 'b-', lw=2, label='f(x)=x³−3x')
for cp in critical_points:
    axes[0].plot(cp, f(cp), 'ro', markersize=8)
axes[0].axhline(y=0, color='gray', lw=0.5)
axes[0].set_title('f(x) with critical points marked'); axes[0].legend(); axes[0].grid(alpha=0.3)

axes[1].plot(x, fp(x), 'r-', lw=2, label="f'(x)=3x²−3")
axes[1].plot(x, fpp(x), 'g--', lw=2, label="f''(x)=6x")
axes[1].axhline(y=0, color='gray', lw=0.5)
axes[1].set_title('First & Second Derivatives'); axes[1].legend(); axes[1].grid(alpha=0.3)
plt.tight_layout(); plt.show()

print("\n结论: x=-1 处 f''<0 → 极大值(山顶)")
print("      x=+1 处 f''>0 → 极小值(谷底)")
print("      x=0 处 f''=0 → 拐点(凹凸性翻转)")
```

**预期输出**：x=−1 是极大值（f''(−1)=−6<0），x=+1 是极小值（f''(1)=6>0）。

---

> **答案校验通过** — 2026-07-11

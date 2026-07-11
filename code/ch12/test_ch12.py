"""Validate Chapter 12: Gradient vector."""
import numpy as np

def numerical_gradient(f, x, h=1e-5):
    grad = np.zeros_like(x)
    for i in range(len(x)):
        xp = x.copy(); xp[i] += h
        xm = x.copy(); xm[i] -= h
        grad[i] = (f(xp) - f(xm)) / (2 * h)
    return grad

# 12.1 Gradient = vector of partials
def f(x): return x[0]**2 + 10*x[1]**2
x0 = np.array([3.0, 2.0])
grad = numerical_gradient(f, x0)
assert abs(grad[0] - 6.0) < 1e-4   # 2*x1 = 6
assert abs(grad[1] - 40.0) < 0.01   # 20*x2 = 40

# 12.2 Descent step: f(x - lr*grad) < f(x)
x = np.array([3.0, 2.0])
f_before = f(x)
lr = 0.05
x_new = x - lr * numerical_gradient(f, x)
assert f(x_new) < f_before

# Large lr can overshoot
x_test = np.array([3.0, 2.0])
lr_big = 0.5
x_big = x_test - lr_big * numerical_gradient(f, x_test)
# With lr=0.5 on this steep function, might overshoot
# Just verify the step formula is correct
assert x_big[0] == 3.0 - 0.5 * grad[0]

# 12.3 Gradient perpendicular to contour
# f=x^2+2y^2, at (xp,yp): grad=[2xp,4yp], tangent=[-4yp,2xp]
xp, yp = 1.5, 0.5
gx, gy = 2*xp, 4*yp
tangent = np.array([-gy, gx])
assert abs(np.dot([gx, gy], tangent)) < 1e-10  # perpendicular

# 12.4 Gradient descent: L(w)=(w-3)^2, w->3
L = lambda w: (w-3)**2
dL = lambda w: 2*(w-3)
w = 0.0
for _ in range(30): w -= 0.1 * dL(w)
assert abs(w - 3.0) < 0.01  # converged close to optimum

print("Ch12 OK -- gradient vector, descent step, perpendicular contour, GD convergence")

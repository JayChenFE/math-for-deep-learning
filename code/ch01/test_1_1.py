import math

print('=== 1.1 类型与浮点精度 ===')
print(type(42))
print(type(3.14))
print(type(True))

a = 0.1 + 0.2
print(a)
print(a == 0.3)

print(f"0.1 的实际存储值: {0.1:.20f}")

EPS = 1e-10
print(abs(a - 0.3) < EPS)
print(math.isclose(a, 0.3))
print('1.1 OK')

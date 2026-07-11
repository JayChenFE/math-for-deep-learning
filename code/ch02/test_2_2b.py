def secant_slope(f, x, h):
    return (f(x + h) - f(x)) / h

def f(x):
    return x ** 2

x_target = 2

print(f"\n{'h':>12}  {'slope':>16}")
print("-" * 32)
for h in [1e-10, 1e-11, 1e-12, 1e-13, 1e-14, 1e-15, 1e-16]:
    slope = secant_slope(f, x_target, h)
    print(f"{h:12.1e}  {slope:16.12f}")

# Verify: some of these should deviate significantly from 4.0
# due to floating-point rounding
slope_1e16 = secant_slope(f, x_target, 1e-16)
# At h=1e-16, slope may be 0.0 or wildly off
print(f"\nAt h=1e-16, slope={slope_1e16:.12f} (should NOT be close to 4.0)")
print('2.2b OK - demonstrates float rounding pitfall')

def secant_slope(f, x, h):
    return (f(x + h) - f(x)) / h

def f(x):
    return x ** 2

x_target = 2
print(f"f(x)=x^2 at x={x_target}: secant slope convergence:")
print(f"{'h':>10}  {'slope':>12}  {'diff from true(4)':>20}")
print("-" * 48)

for h in [1.0, 0.5, 0.1, 0.05, 0.01, 0.001, 0.0001]:
    slope = secant_slope(f, x_target, h)
    true_val = 2 * x_target
    diff = abs(slope - true_val)
    print(f"{h:10.4f}  {slope:12.8f}  {diff:20.16f}")
    # Verify convergence: as h gets smaller, error should decrease
    # (except at the very end where float rounding kicks in)

# Final check: at h=0.0001, should be very close to 4
final_slope = secant_slope(f, x_target, 0.0001)
assert abs(final_slope - 4.0) < 0.001, f"Expected ~4.0, got {final_slope}"
print('2.2a OK - converges to 4.0')

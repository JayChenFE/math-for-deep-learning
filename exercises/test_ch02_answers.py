# Test ch02 exercise 3: f(x)=x^3 at x=2
def secant_slope(f, x, h):
    return (f(x + h) - f(x)) / h
def f3(x): return x ** 3
x_target = 2
true_val = 3 * x_target ** 2  # = 12
final_slope = secant_slope(f3, x_target, 0.0001)
assert abs(final_slope - true_val) < 0.01, f"ex3: expected ~12, got {final_slope}"
print(f"ch02-ex3 OK: slope at h=0.0001 = {final_slope:.6f}")

# Test ch02 exercise 4: f(x)=1/x at x=1
def f_inv(x): return 1 / x
x_target2 = 1
true_val2 = -1 / (x_target2 ** 2)  # = -1
final_slope2 = secant_slope(f_inv, x_target2, 0.0001)
assert abs(final_slope2 - true_val2) < 0.01, f"ex4: expected ~-1, got {final_slope2}"
print(f"ch02-ex4 OK: slope at h=0.0001 = {final_slope2:.6f}")
print("All answer code verified")

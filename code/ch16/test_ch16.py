"""Validate Chapter 16: Conditional Probability & Bayes."""
p_disease = 0.01; p_pos_given_disease = 0.99; p_pos_given_healthy = 0.01
p_pos = p_pos_given_disease * p_disease + p_pos_given_healthy * (1 - p_disease)
p_disease_given_pos = p_pos_given_disease * p_disease / p_pos
assert 0.4 < p_disease_given_pos < 0.6
def naive_bayes(f):
    p0=0.5*(0.3 if f[0] else 0.7)*(0.2 if f[1] else 0.8)
    p1=0.5*(0.8 if f[0] else 0.2)*(0.7 if f[1] else 0.3)
    return 0 if p0>p1 else 1
assert naive_bayes([1,1]) == 1
assert naive_bayes([0,0]) == 0
print("Ch16 ALL PASSED")

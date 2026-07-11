"""Validate Chapter 16: Conditional probability & Bayes theorem."""
import numpy as np

# 16.1 Conditional probability
np.random.seed(42)
n = 10000
age = np.random.normal(35, 12, size=n)
income = np.random.normal(5000, 2000, size=n)
clicked = ((age > 25) & (income > 4000)).astype(int)

p_click = clicked.mean()
p_click_given_age = clicked[age > 25].mean()
# With more conditions, probability changes
both = (age > 25) & (income > 6000)
p_click_given_both = clicked[both].mean()
assert p_click_given_both != p_click  # conditioning changes probability

# 16.2 Bayes theorem
def bayes(p_h, p_e_given_h, p_e_given_not_h):
    p_e = p_e_given_h * p_h + p_e_given_not_h * (1 - p_h)
    return p_e_given_h * p_h / p_e

p_sick = bayes(0.01, 0.99, 0.01)
assert 0.45 < p_sick < 0.55  # ~50%, not 99%!

# Simulation verification
n_p = 100000
is_sick = np.random.random(n_p) < 0.01
test_pos = np.zeros(n_p, dtype=bool)
test_pos[is_sick] = np.random.random(is_sick.sum()) < 0.99
test_pos[~is_sick] = np.random.random((~is_sick).sum()) < 0.01
sim = test_pos[is_sick].sum() / test_pos.sum()
assert abs(sim - p_sick) < 0.05  # simulation matches theory

# 16.3 Naive Bayes — simplified test
spam = ["免费 大奖 点击", "恭喜 中奖 免费", "优惠 限时 免费"]
ham = ["明天 开会 时间", "晚上 一起 吃饭", "会议 纪要 查收"]
all_spam = ' '.join(spam).split()
all_ham = ' '.join(ham).split()
# spam words should have higher prob in spam
assert all_spam.count('免费') / len(all_spam) > all_ham.count('免费') / max(len(all_ham), 1)

print("Ch16 OK -- conditional prob, Bayes theorem (~50%), Naive Bayes")

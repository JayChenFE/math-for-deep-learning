"""Validate Ch16.4: Agent self-awareness — posterior + threshold -> clarify."""
import numpy as np

np.random.seed(42)

intents = ["book_flight", "check_weather", "set_reminder", "play_music"]
n = len(intents)
threshold = 0.7

# Simulate 8 queries: half high-conf, half low-conf
high_count = 0
low_count = 0
for i in range(8):
    probs = np.random.dirichlet(np.ones(n) * 0.5)
    if i % 2 == 0:  # high confidence
        top = np.random.randint(0, n)
        probs[top] += np.random.uniform(0.4, 0.6)
    probs = probs / probs.sum()
    max_prob = probs.max()

    if max_prob >= threshold:
        high_count += 1
    else:
        low_count += 1
        # Should list top-2 intents
        top2 = np.argsort(probs)[-2:][::-1]
        assert len(top2) == 2
        assert max_prob < threshold

# Should have roughly half high, half low
assert high_count >= 2 and low_count >= 2

# Verify threshold logic
p_high = np.array([0.9, 0.05, 0.03, 0.02])
assert p_high.max() >= threshold  # should execute
p_low = np.array([0.4, 0.35, 0.15, 0.1])
assert p_low.max() < threshold  # should clarify
top2_low = np.argsort(p_low)[-2:][::-1]
assert top2_low[0] != top2_low[1]  # two different candidates

print(f"Ch16.4 OK -- threshold={threshold}, high-conf={high_count}, low-conf(clarify)={low_count}")

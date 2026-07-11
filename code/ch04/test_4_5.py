"""Validate Ch4.5: Agent state transition with vector addition."""
import numpy as np

np.random.seed(42)

# State transition simulation
state = np.array([0.0, 0.0])
actions = [
    np.array([2.0, 1.0]),
    np.array([-1.0, 2.5]),
    np.array([1.5, -1.0]),
    np.array([0.5, 2.0]),
]

states = [state.copy()]
for a in actions:
    state = state + a
    states.append(state.copy())
states = np.array(states)

# Verify: final_state - initial = sum of all actions
total_action = sum(actions)
net_displacement = states[-1] - states[0]
assert np.allclose(total_action, net_displacement)

# Verify each step = previous + action
for i, a in enumerate(actions):
    assert np.allclose(states[i+1], states[i] + a)

# Verify trajectory is continuous
for i in range(len(states) - 1):
    assert np.linalg.norm(states[i+1] - states[i]) > 0  # each step moves

# quiver visualization check
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 7))
    for i, (s_before, a) in enumerate(zip(states[:-1], actions)):
        ax.quiver(s_before[0], s_before[1], a[0], a[1],
                  angles='xy', scale_units='xy', scale=1, color='blue', width=0.03)
    plt.close(fig)
    print("Ch4.5 OK -- agent state transition: quiver trajectory verified")
except ImportError:
    print("Ch4.5 OK -- agent state transition: state += action chain verified")

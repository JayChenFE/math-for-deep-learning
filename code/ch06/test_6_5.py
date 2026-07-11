"""Validate Ch6.5: Broadcasting rules."""
import numpy as np

# Case 1: (batch, d_out) + (d_out,)
X = np.random.randn(32, 256)
b = np.random.randn(256)
result = X + b
assert result.shape == (32, 256)

# Case 2: (3,1) + (1,4) -> (3,4)
a = np.ones((3, 1))
b2 = np.ones((1, 4))
c = a + b2
assert c.shape == (3, 4)
assert np.allclose(c, 2.0)

# Case 3: scalar broadcast
d = np.ones((3, 4))
e = d + 10.0
assert e.shape == (3, 4)
assert np.allclose(e, 11.0)

# Case 4: incompatible — should raise
try:
    f = np.ones((3, 4))
    g = np.ones((3, 2))
    h = f + g
    assert False, "Should have raised ValueError"
except ValueError:
    pass  # Expected

# Case 5: Transformer-style broadcast
batch, seq_len, d_model = 32, 10, 512
X_trans = np.random.randn(batch, seq_len, d_model)
ln_bias = np.random.randn(d_model)
result_ln = X_trans + ln_bias
assert result_ln.shape == (batch, seq_len, d_model)

# Broadcasting rule check: (32, 1, 10) + (1, 5, 10) -> (32, 5, 10)
a3d = np.ones((32, 1, 10))
b3d = np.ones((1, 5, 10))
c3d = a3d + b3d
assert c3d.shape == (32, 5, 10)

print("Ch6.5 OK -- broadcasting: all compatible cases pass, incompatible raises")

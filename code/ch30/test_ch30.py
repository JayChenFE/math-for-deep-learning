"""Validate Chapter 30: Multi-head attention & Transformer Block."""
import numpy as np

np.random.seed(42)

# 30.1 Multi-head shape ballet
batch, seq_len, d_model, n_heads = 2, 5, 512, 8
d_k = d_model // n_heads
X = np.random.randn(batch, seq_len, d_model)
W_Q = np.random.randn(d_model, d_model) * 0.02
Q = X @ W_Q
Q_r = Q.reshape(batch, seq_len, n_heads, d_k)
Q_t = Q_r.transpose(0, 2, 1, 3)
assert Q_t.shape == (batch, n_heads, seq_len, d_k)

# 30.2 Concat + output projection
head_out = np.random.randn(batch, n_heads, seq_len, d_k)
concat = head_out.transpose(0, 2, 1, 3).reshape(batch, seq_len, d_model)
assert concat.shape == X.shape

# 30.3 FFN
d_ff = 2048
W1 = np.random.randn(d_model, d_ff) * 0.02
b1 = np.zeros(d_ff)
W2 = np.random.randn(d_ff, d_model) * 0.02
b2 = np.zeros(d_model)
h = np.maximum(0, concat @ W1 + b1)
ffn_out = h @ W2 + b2
assert ffn_out.shape == (batch, seq_len, d_model)

# 30.4 LayerNorm + residual (with learnable gamma/beta)
def layer_norm(x, gamma, beta, eps=1e-5):
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return gamma * (x - mean) / np.sqrt(var + eps) + beta

gamma = np.ones(d_model); beta = np.zeros(d_model)
ln_out = layer_norm(X, gamma, beta)
assert ln_out.shape == X.shape
residual = X + ffn_out
assert residual.shape == X.shape

# 30.5 Full Block
class TransformerBlock:
    def __init__(self):
        self.d_model = 256; self.n_heads = 8; self.d_k = 32; self.d_ff = 1024
        self.W_Q = np.random.randn(256, 256) * 0.02
        self.W_K = np.random.randn(256, 256) * 0.02
        self.W_V = np.random.randn(256, 256) * 0.02
        self.W_O = np.random.randn(256, 256) * 0.02
        self.W1 = np.random.randn(256, 1024) * 0.02
        self.b1 = np.zeros(1024)
        self.W2 = np.random.randn(1024, 256) * 0.02
        self.b2 = np.zeros(256)
        self.gamma1 = np.ones(256); self.beta1 = np.zeros(256)
        self.gamma2 = np.ones(256); self.beta2 = np.zeros(256)

    def ln(self, x, gamma, beta, eps=1e-5):
        m = x.mean(axis=-1, keepdims=True); v = x.var(axis=-1, keepdims=True)
        return gamma * (x - m) / np.sqrt(v + eps) + beta

    def attn(self, x):
        B, L, D = x.shape
        Q = (x @ self.W_Q).reshape(B, L, self.n_heads, self.d_k).transpose(0, 2, 1, 3)
        K = (x @ self.W_K).reshape(B, L, self.n_heads, self.d_k).transpose(0, 2, 1, 3)
        V = (x @ self.W_V).reshape(B, L, self.n_heads, self.d_k).transpose(0, 2, 1, 3)
        s = Q @ K.transpose(0, 1, 3, 2) / np.sqrt(self.d_k)
        s = s - s.max(axis=-1, keepdims=True)
        a = np.exp(s) / np.exp(s).sum(axis=-1, keepdims=True)
        out = (a @ V).transpose(0, 2, 1, 3).reshape(B, L, D)
        return out @ self.W_O

    def ffn(self, x):
        h = np.maximum(0, x @ self.W1 + self.b1)
        return h @ self.W2 + self.b2

    def forward(self, x):
        x = x + self.attn(self.ln(x, self.gamma1, self.beta1))
        x = x + self.ffn(self.ln(x, self.gamma2, self.beta2))
        return x

block = TransformerBlock()
x = np.random.randn(2, 10, 256)
out = block.forward(x)
assert out.shape == x.shape  # Residual ensures shape invariance

print("Ch30 OK -- multi-head shape ballet, FFN, Pre-Norm, full Block forward")

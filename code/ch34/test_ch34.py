"""Validate Chapter 34: Fine-tuning & LoRA."""
import numpy as np

np.random.seed(42)

# 34.1 MiniLM parameter count
vocab_size, d_model, n_blocks, d_ff = 10000, 128, 2, 512
embed_params = vocab_size * d_model
per_block = (d_model * d_model * 4  # QKV + O
             + d_model * d_ff * 2   # FFN W1 + W2
             + d_ff + d_model)       # FFN biases (W2 uses d_model)
lm_head_params = d_model * vocab_size
total = embed_params + n_blocks * per_block + lm_head_params
print(f"MiniLM: embed={embed_params:,}, per_block={per_block:,}, head={lm_head_params:,}")
print(f"Total: {total:,} params (equivalent to model.count_params())")

# 34.3 LoRA parameter compression
d, r = 768, 8
W_qkv_full = d * (3 * d)     # 768 × 2304
W_o_full = d * d              # 768 × 768
full_params = W_qkv_full + W_o_full

# LoRA: B_A(d,r) @ A_A(r,3d) + B_O(d,r) @ A_O(r,d)
lora_params = d * r + r * (3*d) + d * r + r * d
ratio = lora_params / full_params * 100
print(f"\nLoRA compression: {lora_params:,} / {full_params:,} = {ratio:.1f}%")
assert ratio < 5.0, f"LoRA should compress to <5%, got {ratio:.1f}%"

# 34.5 EarlyStopping logic
class EarlyStopping:
    def __init__(self, patience=3, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return False  # not stopping
        else:
            self.counter += 1
            return self.counter >= self.patience

# Test: improving losses
es = EarlyStopping(patience=3)
assert not es(1.0)  # new best
assert not es(0.8)  # improved
assert es.best_loss == 0.8
assert not es(0.9)  # worse, counter=1
assert not es(0.85) # worse, counter=2
assert es(0.9)      # worse, counter=3 >= patience -> stop
print("\nEarlyStopping: correct (stopped after 3 non-improving epochs)")

# Test: improvement with min_delta
es2 = EarlyStopping(patience=2, min_delta=0.01)
assert not es2(1.0)
assert not es2(0.995)  # improvement < min_delta, counter=1
assert es2(1.0)        # worse, counter=2 >= patience
print("EarlyStopping with min_delta: correct")

print("\nCh34 OK -- LoRA compression, EarlyStopping logic")

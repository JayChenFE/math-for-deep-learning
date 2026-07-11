"""Validate Ch3.4: Tensor — 3-D and above."""
import numpy as np

# 3D: batch of grayscale images
batch, height, width = 32, 28, 28
images = np.random.randn(batch, height, width)
assert images.ndim == 3
assert images.shape == (32, 28, 28)
assert images[0].shape == (28, 28)

# 4D: batch of RGB images
images_rgb = np.random.randn(32, 224, 224, 3)
assert images_rgb.ndim == 4
assert images_rgb.shape == (32, 224, 224, 3)
assert images_rgb[0, :, :, 0].shape == (224, 224)

# Transformer-style 3D tensor
seq_len, d_model = 10, 512
X_transformer = np.random.randn(batch, seq_len, d_model)
assert X_transformer.shape == (32, 10, 512)
assert X_transformer[0, 3].shape == (512,)

# ndim verification for all ranks
assert np.array(2.37).ndim == 0
assert np.array([1, 2, 3]).ndim == 1
assert np.random.randn(3, 4).ndim == 2
assert np.random.randn(3, 4, 5).ndim == 3

print("Ch3.4 OK -- tensor: 3D+, shape tells the physical meaning of the data")

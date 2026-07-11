"""Validate Ch4.4: Word embedding semantic arithmetic."""
import numpy as np

word_vectors = {
    "国王": np.array([0.9, 0.7, 0.5, 0.1]),
    "男人": np.array([0.6, 0.7, 0.2, 0.1]),
    "女王": np.array([0.4, 0.2, 0.6, 0.0]),
    "女人": np.array([0.1, 0.2, 0.3, 0.0]),
}

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

result = word_vectors["国王"] - word_vectors["男人"] + word_vectors["女人"]

queen_sim = cosine_similarity(result, word_vectors["女王"])
king_sim = cosine_similarity(result, word_vectors["国王"])
man_sim = cosine_similarity(result, word_vectors["男人"])
woman_sim = cosine_similarity(result, word_vectors["女人"])

# Result should be closest to "女王"
assert queen_sim > king_sim
assert queen_sim > man_sim
assert queen_sim > woman_sim

# Cosine similarity range check
assert 0 <= queen_sim <= 1
assert 0 <= king_sim <= 1

print(f"Ch4.4 OK -- king-man+woman -> queen: sim={queen_sim:.4f} (highest)")
print(f"  queen={queen_sim:.4f}, king={king_sim:.4f}, man={man_sim:.4f}, woman={woman_sim:.4f}")

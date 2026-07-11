"""Validate Ch5.3: Recommender system with dot product."""
import numpy as np

movies = {
    "星际穿越": np.array([0.3, 0.0, 0.9, 0.2, 0.1]),
    "泰坦尼克": np.array([0.1, 0.1, 0.0, 0.95, 0.1]),
    "盗梦空间": np.array([0.6, 0.0, 0.7, 0.1, 0.5]),
    "喜剧之王": np.array([0.0, 0.95, 0.0, 0.2, 0.0]),
    "黑客帝国": np.array([0.8, 0.0, 0.6, 0.1, 0.3]),
    "爱乐之城": np.array([0.0, 0.3, 0.0, 0.9, 0.0]),
}

user = np.array([0.2, 0.05, 0.8, 0.05, 0.6])

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)

scores = {name: cosine_similarity(user, vec) for name, vec in movies.items()}
top3 = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]

# User likes sci-fi + suspense -> 盗梦空间 and 星际穿越 should top
assert top3[0][0] in ["盗梦空间", "星际穿越"]
assert top3[1][0] in ["盗梦空间", "星际穿越"]

# 喜剧之王 (comedy) should be ranked low
assert scores["喜剧之王"] < scores["星际穿越"]

print(f"Ch5.3 OK -- Top-3: {[t[0] for t in top3]}")

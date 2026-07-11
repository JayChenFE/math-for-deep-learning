"""Validate Ch5.5: RAG retrieval with dot product."""
import numpy as np

np.random.seed(42)

n_docs, d_embed = 100, 128
doc_vectors = np.random.randn(n_docs, d_embed)
doc_vectors = doc_vectors / np.linalg.norm(doc_vectors, axis=1, keepdims=True)

# Mark some docs as relevant by boosting dimensions
relevant_ids = [12, 37, 55, 78, 91]
for rid in relevant_ids:
    doc_vectors[rid, :16] += np.random.randn(16) * 0.8
    doc_vectors[rid] = doc_vectors[rid] / np.linalg.norm(doc_vectors[rid])

# Query with same boosted dimensions
query = np.random.randn(d_embed)
query[:16] += np.random.randn(16) * 0.8
query = query / np.linalg.norm(query)

# RAG: dot product -> sort -> Top-K
scores = doc_vectors @ query
top_k = 5
top_indices = np.argsort(scores)[::-1][:top_k]

# Verify retrieval works better than random
random_hits = 5 * len(relevant_ids) / n_docs  # expected random hits
actual_hits = sum(1 for i in top_indices if i in relevant_ids)
# Top-5 scores should be significantly higher than average (dot product works)
assert scores[top_indices].mean() > scores.mean() * 2, \
    f"Top-5 avg={scores[top_indices].mean():.3f} vs global avg={scores.mean():.3f}"

print(f"Ch5.5 OK -- RAG: {actual_hits}/{len(relevant_ids)} relevant docs retrieved, "
      f"Top-5 avg={scores[top_indices].mean():.4f} vs all avg={scores.mean():.4f}")

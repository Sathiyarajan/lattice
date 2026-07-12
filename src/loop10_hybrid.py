import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder

model = SentenceTransformer("all-MiniLM-L6-v2")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

docs = [
    "Python is a popular programming language for data science.",
    "The Eiffel Tower is a famous landmark in Paris.",
    "PostgreSQL is a powerful open-source relational database.",
    "Snakes are reptiles found on every continent except Antarctica.",
    "Vector databases enable fast similarity search over embeddings.",
]

tokenized = [d.lower().split() for d in docs]
bm25 = BM25Okapi(tokenized)
vecs = model.encode(docs)

def cos_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def vector_search(query, k=3):
    qvec = model.encode([query])[0]
    scores = [cos_sim(qvec, v) for v in vecs]
    ranked = sorted(zip(scores, docs), reverse=True)
    return ranked[:k]

def hybrid_search(query, k=3, alpha=0.5):
    qvec = model.encode([query])[0]
    vec_scores = np.array([cos_sim(qvec, v) for v in vecs])
    bm25_scores = np.array(bm25.get_scores(query.lower().split()))
    vec_norm = (vec_scores - vec_scores.min()) / (np.ptp(vec_scores) + 1e-9)
    bm25_norm = (bm25_scores - bm25_scores.min()) / (np.ptp(bm25_scores) + 1e-9)
    combined = alpha * vec_norm + (1 - alpha) * bm25_norm
    ranked = sorted(zip(combined, docs), reverse=True)
    return ranked[:k]

def rerank(query, candidates, k=3):
    pairs = [[query, doc] for _, doc in candidates]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(scores, [doc for _, doc in candidates]), reverse=True)
    return ranked[:k]

query = "Python database"

vec_only = vector_search(query)
hybrid = hybrid_search(query)
reranked = rerank(query, hybrid)

print("vector-only:", [d for _, d in vec_only])
print("hybrid:", [d for _, d in hybrid])
print("reranked:", [d for _, d in reranked])

top_reranked = {d for _, d in reranked}
assert "PostgreSQL is a powerful open-source relational database." in top_reranked
assert "Python is a popular programming language for data science." in top_reranked
print("OK")

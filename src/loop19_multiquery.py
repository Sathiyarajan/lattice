import numpy as np
import requests
from sentence_transformers import SentenceTransformer, CrossEncoder

model = SentenceTransformer("all-MiniLM-L6-v2")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

docs = [
    "Python is a popular programming language for data science.",
    "PostgreSQL is a powerful open-source relational database.",
    "The Eiffel Tower is a famous landmark in Paris.",
    "Vector databases enable fast similarity search over embeddings.",
    "Snakes are reptiles found on every continent except Antarctica.",
]
vecs = model.encode(docs)

def cos_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def single_query_search(query, k=3):
    qvec = model.encode([query])[0]
    scores = [cos_sim(qvec, v) for v in vecs]
    return sorted(zip(scores, docs), reverse=True)[:k]

def generate_query_variants(query, n=3):
    prompt = (
        f"Generate {n} alternative phrasings of this search query, one per line, no numbering:\n{query}"
    )
    resp = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": "llama3.2", "prompt": prompt, "stream": False},
        timeout=60,
    )
    lines = [l.strip("-* ").strip() for l in resp.json()["response"].strip().split("\n") if l.strip()]
    return [query] + lines[:n]

def multi_query_search(query, k=3):
    variants = generate_query_variants(query)
    seen = {}
    for v in variants:
        qvec = model.encode([v])[0]
        for doc, vec in zip(docs, vecs):
            score = cos_sim(qvec, vec)
            seen[doc] = max(seen.get(doc, -1), score)
    ranked = sorted(seen.items(), key=lambda x: -x[1])[:k]
    return [(score, doc) for doc, score in ranked]

ambiguous_query = "database technology"

single = single_query_search(ambiguous_query)
multi = multi_query_search(ambiguous_query)

print("single-query:", [d for _, d in single])
print("multi-query:", [d for _, d in multi])

multi_docs = {d for _, d in multi}
assert "PostgreSQL is a powerful open-source relational database." in multi_docs
assert "Vector databases enable fast similarity search over embeddings." in multi_docs
print("OK")

import time
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

docs = [
    "The cat sat on the mat.",
    "A feline rested on the rug.",
    "Stock markets fell sharply today.",
    "Investors worried about inflation.",
    "Dogs are loyal companions.",
] * 200  # scale up for speed comparison

vecs = model.encode(docs, show_progress_bar=False).astype("float32")
faiss.normalize_L2(vecs)

query = "kittens on furniture"
qvec = model.encode([query]).astype("float32")
faiss.normalize_L2(qvec)

# brute-force baseline
t0 = time.time()
sims = vecs @ qvec[0]
brute_top = np.argsort(-sims)[:3]
brute_time = time.time() - t0

# faiss index
index = faiss.IndexFlatIP(vecs.shape[1])
index.add(vecs)
t0 = time.time()
D, I = index.search(qvec, 3)
faiss_time = time.time() - t0

print("brute top:", [docs[i] for i in brute_top], f"{brute_time*1000:.3f}ms")
print("faiss top:", [docs[i] for i in I[0]], f"{faiss_time*1000:.3f}ms")

brute_texts = {docs[i] for i in brute_top}
faiss_texts = {docs[i] for i in I[0]}
assert brute_texts == faiss_texts, "faiss and brute-force disagree"
print("OK")

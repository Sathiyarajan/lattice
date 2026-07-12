from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")

sentences = [
    "The cat sat on the mat.",
    "A feline rested on the rug.",
    "Stock markets fell sharply today.",
]

vecs = model.encode(sentences)

dims = {v.shape[0] for v in vecs}
assert len(dims) == 1, f"inconsistent dims: {dims}"

def cos_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

sim_related = cos_sim(vecs[0], vecs[1])
sim_unrelated = cos_sim(vecs[0], vecs[2])

print("dim:", vecs[0].shape[0])
print("sim(cat,feline):", sim_related)
print("sim(cat,stocks):", sim_unrelated)
assert sim_related > sim_unrelated, "similar sentences should be closer"
print("OK")

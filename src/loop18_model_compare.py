import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

docs = [
    "The cat sat on the mat.",
    "A feline rested on the rug.",
    "Stock markets fell sharply today.",
    "Investors worried about inflation and rate hikes.",
    "Dogs are loyal companions to humans.",
    "Puppies love to play fetch in the park.",
]
labels = ["animal", "animal", "finance", "finance", "animal", "animal"]

models_to_try = ["all-MiniLM-L6-v2", "paraphrase-MiniLM-L3-v2"]

def intra_vs_inter_cluster_sim(vecs, labels):
    sim = cosine_similarity(vecs)
    n = len(labels)
    intra, inter = [], []
    for i in range(n):
        for j in range(i + 1, n):
            if labels[i] == labels[j]:
                intra.append(sim[i, j])
            else:
                inter.append(sim[i, j])
    return np.mean(intra), np.mean(inter)

results = {}
for name in models_to_try:
    model = SentenceTransformer(name)
    vecs = model.encode(docs)
    intra, inter = intra_vs_inter_cluster_sim(vecs, labels)
    results[name] = {"intra": intra, "inter": inter, "separation": intra - inter}
    print(f"{name}: intra={intra:.4f} inter={inter:.4f} separation={intra-inter:.4f}")

best = max(results, key=lambda k: results[k]["separation"])
print(f"chosen model: {best} (highest intra/inter cluster separation)")

assert all(r["intra"] > r["inter"] for r in results.values()), "similar docs should cluster tighter than dissimilar"
print("OK")

import numpy as np
from sentence_transformers import SentenceTransformer
from neo4j import GraphDatabase

model = SentenceTransformer("all-MiniLM-L6-v2")
driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "password"))

def cos_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

candidates = ["Lattice", "Lattice Docs", "lattice system", "PostgreSQL", "Postgres", "pgvector"]

def find_duplicate_clusters(names, threshold=0.75):
    vecs = model.encode(names)
    n = len(names)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        parent[find(x)] = find(y)

    for i in range(n):
        for j in range(i + 1, n):
            if cos_sim(vecs[i], vecs[j]) >= threshold:
                union(i, j)

    clusters = {}
    for i in range(n):
        root = find(i)
        clusters.setdefault(root, []).append(names[i])
    return list(clusters.values())

clusters = find_duplicate_clusters(candidates)
print("clusters:", clusters)

lattice_cluster = [c for c in clusters if any("attice" in n for n in c)]
postgres_cluster = [c for c in clusters if any("ostgres" in n or "pgvector" in n for n in c)]

assert any(len(c) >= 2 for c in lattice_cluster), "Lattice variants should merge into one cluster"
assert not any("PostgreSQL" in c and "pgvector" in c for c in clusters), "distinct entities over-merged"
print("OK")

driver.close()

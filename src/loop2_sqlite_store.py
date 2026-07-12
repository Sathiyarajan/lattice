import sqlite3
import json
import os
import numpy as np
from sentence_transformers import SentenceTransformer

DB_PATH = os.path.expanduser("~/projects/lattice/data/vectors.db")

docs = [
    "The cat sat on the mat.",
    "A feline rested on the rug.",
    "Stock markets fell sharply today.",
    "Investors worried about inflation.",
    "Dogs are loyal companions.",
]

model = SentenceTransformer("all-MiniLM-L6-v2")
vecs = model.encode(docs)

os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
if os.path.exists(DB_PATH):
    os.remove(DB_PATH)

conn = sqlite3.connect(DB_PATH)
conn.execute("CREATE TABLE items (id INTEGER PRIMARY KEY, text TEXT, embedding TEXT)")
for text, vec in zip(docs, vecs):
    conn.execute(
        "INSERT INTO items (text, embedding) VALUES (?, ?)",
        (text, json.dumps(vec.tolist())),
    )
conn.commit()

def cos_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def search(query, k=3):
    qvec = model.encode([query])[0]
    rows = conn.execute("SELECT text, embedding FROM items").fetchall()
    scored = []
    for text, emb_json in rows:
        emb = np.array(json.loads(emb_json))
        scored.append((cos_sim(qvec, emb), text))
    scored.sort(reverse=True)
    return scored[:k]

results = search("kittens on furniture", k=3)
for score, text in results:
    print(f"{score:.4f}  {text}")

assert results[0][1] in ("The cat sat on the mat.", "A feline rested on the rug.")
print("OK")

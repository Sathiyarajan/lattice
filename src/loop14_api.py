import time
from functools import lru_cache
from fastapi import FastAPI
import psycopg2
from sentence_transformers import SentenceTransformer

app = FastAPI()
model = SentenceTransformer("all-MiniLM-L6-v2")
conn = psycopg2.connect(host="localhost", port=5432, user="postgres", password="pass", dbname="postgres")

@lru_cache(maxsize=256)
def cached_retrieve(query, k=2):
    qvec = model.encode([query])[0]
    cur = conn.cursor()
    cur.execute(
        "SELECT text FROM lattice_chunks ORDER BY embedding <-> %s::vector LIMIT %s",
        (qvec.tolist(), k),
    )
    result = tuple(r[0] for r in cur.fetchall())
    cur.close()
    return result

@app.get("/query")
def query(q: str, k: int = 2):
    t0 = time.time()
    results = cached_retrieve(q, k)
    return {"query": q, "results": list(results), "latency_ms": (time.time() - t0) * 1000}

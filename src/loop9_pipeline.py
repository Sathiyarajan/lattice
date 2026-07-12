import sys
sys.path.insert(0, "/home/sathi/projects/lattice/src")
from pathlib import Path
import psycopg2
from sentence_transformers import SentenceTransformer
from loop7_ingest import ingest, RAW_DIR
from loop8_chunking import chunk_text

model = SentenceTransformer("all-MiniLM-L6-v2")

docs = ingest(RAW_DIR)
all_chunks = []
for d in docs:
    all_chunks.extend(chunk_text(d["text"], d["source"], max_chars=150, overlap=15))

texts = [c["text"] for c in all_chunks]
vecs = model.encode(texts)

conn = psycopg2.connect(host="localhost", port=5432, user="postgres", password="pass", dbname="postgres")
conn.autocommit = True
cur = conn.cursor()
cur.execute("DROP TABLE IF EXISTS lattice_chunks")
cur.execute(f"""
    CREATE TABLE lattice_chunks (
        id SERIAL PRIMARY KEY,
        source TEXT,
        chunk_id INTEGER,
        text TEXT,
        embedding vector({vecs.shape[1]})
    )
""")
for chunk, vec in zip(all_chunks, vecs):
    cur.execute(
        "INSERT INTO lattice_chunks (source, chunk_id, text, embedding) VALUES (%s, %s, %s, %s)",
        (chunk["source"], chunk["position"], chunk["text"], vec.tolist()),
    )

cur.execute("SELECT COUNT(*) FROM lattice_chunks")
row_count = cur.fetchone()[0]
print("chunks:", len(all_chunks), "rows:", row_count)
assert row_count == len(all_chunks), "row count mismatch"

cur.execute("SELECT source, text, embedding FROM lattice_chunks LIMIT 1")
sample = cur.fetchone()
dims = sample[2].count(",") + 1
assert sample is not None and dims == vecs.shape[1], f"expected {vecs.shape[1]} dims, got {dims}"
print("spot-check:", sample[0], sample[1][:50])
print("OK")

cur.close()
conn.close()

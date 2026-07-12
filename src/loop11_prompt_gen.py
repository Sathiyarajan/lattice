import requests
import psycopg2
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

conn = psycopg2.connect(host="localhost", port=5432, user="postgres", password="pass", dbname="postgres")
cur = conn.cursor()

def retrieve(query, k=2):
    qvec = model.encode([query])[0]
    cur.execute(
        "SELECT source, chunk_id, text FROM lattice_chunks ORDER BY embedding <-> %s::vector LIMIT %s",
        (qvec.tolist(), k),
    )
    return cur.fetchall()

def build_prompt(query, chunks):
    context_block = "\n".join(
        f"[{i+1}] (source: {source}#{chunk_id}) {text}"
        for i, (source, chunk_id, text) in enumerate(chunks)
    )
    return (
        "Answer the question using only the numbered context below. "
        "Cite sources using [n] notation.\n\n"
        f"Context:\n{context_block}\n\nQuestion: {query}\nAnswer:"
    )

def generate(prompt):
    resp = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": "llama3.2", "prompt": prompt, "stream": False},
        timeout=120,
    )
    return resp.json()["response"].strip()

query = "What does Lattice combine for retrieval?"
chunks = retrieve(query)
prompt = build_prompt(query, chunks)
answer = generate(prompt)

print("Chunks:", [(s, cid) for s, cid, _ in chunks])
print("Answer:", answer)

assert "[1]" in answer or "[2]" in answer, "answer missing citation marker"
print("OK")

cur.close()
conn.close()

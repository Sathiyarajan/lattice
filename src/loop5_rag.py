import psycopg2
import requests
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

docs = [
    "The Eiffel Tower is located in Paris, France, and was completed in 1889.",
    "The Great Wall of China stretches over 13,000 miles.",
    "Mount Everest is the tallest mountain above sea level at 8,849 meters.",
    "The Amazon rainforest produces about 20% of the world's oxygen.",
    "Python was created by Guido van Rossum and released in 1991.",
]
vecs = model.encode(docs)

conn = psycopg2.connect(host="localhost", port=5432, user="postgres", password="pass", dbname="postgres")
conn.autocommit = True
cur = conn.cursor()
cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
cur.execute("DROP TABLE IF EXISTS rag_items")
cur.execute(f"CREATE TABLE rag_items (id SERIAL PRIMARY KEY, text TEXT, embedding vector({vecs.shape[1]}))")
for text, vec in zip(docs, vecs):
    cur.execute("INSERT INTO rag_items (text, embedding) VALUES (%s, %s)", (text, vec.tolist()))

def retrieve(query, k=2):
    qvec = model.encode([query])[0]
    cur.execute(
        "SELECT text FROM rag_items ORDER BY embedding <-> %s::vector LIMIT %s",
        (qvec.tolist(), k),
    )
    return [r[0] for r in cur.fetchall()]

def generate(query, context):
    prompt = (
        "Answer the question using ONLY the context below. "
        "If the answer isn't in the context, say you don't know.\n\n"
        f"Context:\n{chr(10).join(context)}\n\nQuestion: {query}\nAnswer:"
    )
    resp = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": "llama3.2", "prompt": prompt, "stream": False},
        timeout=120,
    )
    return resp.json()["response"].strip()

query = "When was the Eiffel Tower completed?"
context = retrieve(query)
answer = generate(query, context)

print("Retrieved:", context)
print("Answer:", answer)

assert "1889" in answer, f"answer not grounded in retrieved fact: {answer}"
print("OK")

cur.close()
conn.close()

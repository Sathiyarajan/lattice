import psycopg2
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

docs = [
    "The cat sat on the mat.",
    "A feline rested on the rug.",
    "Stock markets fell sharply today.",
    "Investors worried about inflation.",
    "Dogs are loyal companions.",
]
vecs = model.encode(docs)

conn = psycopg2.connect(host="localhost", port=5432, user="postgres", password="pass", dbname="postgres")
conn.autocommit = True
cur = conn.cursor()

cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
cur.execute("DROP TABLE IF EXISTS items")
cur.execute(f"CREATE TABLE items (id SERIAL PRIMARY KEY, text TEXT, embedding vector({vecs.shape[1]}))")

for text, vec in zip(docs, vecs):
    cur.execute(
        "INSERT INTO items (text, embedding) VALUES (%s, %s)",
        (text, vec.tolist()),
    )

query = "kittens on furniture"
qvec = model.encode([query])[0]

cur.execute(
    "SELECT text, embedding <-> %s::vector AS dist FROM items ORDER BY dist LIMIT 3",
    (qvec.tolist(),),
)
results = cur.fetchall()
for text, dist in results:
    print(f"{dist:.4f}  {text}")

assert results[0][0] in ("The cat sat on the mat.", "A feline rested on the rug.")
print("OK")

cur.close()
conn.close()

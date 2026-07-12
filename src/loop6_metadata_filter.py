import psycopg2
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

docs = [
    ("The Eiffel Tower is located in Paris, France.", "travel"),
    ("Python was created by Guido van Rossum in 1991.", "tech"),
    ("Mount Everest is the tallest mountain on Earth.", "travel"),
    ("Rust emphasizes memory safety without garbage collection.", "tech"),
]
vecs = model.encode([d[0] for d in docs])

conn = psycopg2.connect(host="localhost", port=5432, user="postgres", password="pass", dbname="postgres")
conn.autocommit = True
cur = conn.cursor()
cur.execute("DROP TABLE IF EXISTS meta_items")
cur.execute(f"CREATE TABLE meta_items (id SERIAL PRIMARY KEY, text TEXT, category TEXT, embedding vector({vecs.shape[1]}))")
for (text, cat), vec in zip(docs, vecs):
    cur.execute("INSERT INTO meta_items (text, category, embedding) VALUES (%s, %s, %s)", (text, cat, vec.tolist()))

def retrieve(query, category=None, k=2):
    qvec = model.encode([query])[0]
    if category:
        cur.execute(
            "SELECT text, category FROM meta_items WHERE category = %s ORDER BY embedding <-> %s::vector LIMIT %s",
            (category, qvec.tolist(), k),
        )
    else:
        cur.execute(
            "SELECT text, category FROM meta_items ORDER BY embedding <-> %s::vector LIMIT %s",
            (qvec.tolist(), k),
        )
    return cur.fetchall()

query = "programming languages"
unfiltered = retrieve(query, k=4)
filtered = retrieve(query, category="tech", k=4)

print("unfiltered:", unfiltered)
print("filtered (tech):", filtered)

assert all(cat == "tech" for _, cat in filtered), "filter leaked other categories"
assert len(filtered) == 2, "expected exactly 2 tech docs"
print("OK")

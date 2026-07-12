import psycopg2
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
conn = psycopg2.connect(host="localhost", port=5432, user="postgres", password="pass", dbname="postgres")
cur = conn.cursor()

def retrieve(query, k=2):
    qvec = model.encode([query])[0]
    cur.execute(
        "SELECT text FROM lattice_chunks ORDER BY embedding <-> %s::vector LIMIT %s",
        (qvec.tolist(), k),
    )
    return [r[0] for r in cur.fetchall()]

test_cases = [
    {"query": "What does Lattice combine?", "expected_keyword": "vector"},
    {"query": "What database does the system use?", "expected_keyword": "PostgreSQL"},
]

def context_precision_at_k(query, expected_keyword, k=2):
    chunks = retrieve(query, k)
    hits = sum(1 for c in chunks if expected_keyword.lower() in c.lower())
    return hits / k

def run_eval(cases):
    scores = []
    for case in cases:
        score = context_precision_at_k(case["query"], case["expected_keyword"])
        scores.append(score)
        print(f"{case['query']!r} -> precision@2={score:.2f}")
    return sum(scores) / len(scores)

avg1 = run_eval(test_cases)
avg2 = run_eval(test_cases)

print("avg precision@2:", avg1)
assert avg1 == avg2, "metric not reproducible across runs"
assert avg1 > 0, "baseline eval found nothing relevant"
print("OK — baseline precision@2 =", avg1)

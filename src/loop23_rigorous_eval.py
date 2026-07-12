import requests
import psycopg2
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
conn = psycopg2.connect(host="localhost", port=5432, user="postgres", password="pass", dbname="postgres")

def call_llm(prompt):
    resp = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": "llama3.2", "prompt": prompt, "stream": False},
        timeout=60,
    )
    return resp.json()["response"].strip()

def get_chunks():
    cur = conn.cursor()
    cur.execute("SELECT text FROM lattice_chunks")
    chunks = [r[0] for r in cur.fetchall()]
    cur.close()
    return chunks

def generate_qa_pair(chunk):
    prompt = f"Generate one factual question and its answer from this text. Format as 'Q: ...\\nA: ...'\n\nText: {chunk}"
    response = call_llm(prompt)
    q, a = None, None
    for line in response.split("\n"):
        if line.strip().startswith("Q:"):
            q = line.split("Q:", 1)[1].strip()
        elif line.strip().startswith("A:"):
            a = line.split("A:", 1)[1].strip()
    return q, a

def retrieve(query, k=2):
    qvec = model.encode([query])[0]
    cur = conn.cursor()
    cur.execute(
        "SELECT text FROM lattice_chunks ORDER BY embedding <-> %s::vector LIMIT %s",
        (qvec.tolist(), k),
    )
    result = [r[0] for r in cur.fetchall()]
    cur.close()
    return result

def context_precision_at_k(retrieved, source_chunk):
    return 1.0 if source_chunk in retrieved else 0.0

def cos_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def faithfulness_score(answer, context):
    avec = model.encode([answer])[0]
    cvecs = model.encode(context)
    return float(max(cos_sim(avec, c) for c in cvecs))

chunks = get_chunks()
qa_pairs = []
for chunk in chunks[:3]:
    q, a = generate_qa_pair(chunk)
    if q:
        qa_pairs.append({"question": q, "expected_answer": a, "source_chunk": chunk})

precision_scores, faithfulness_scores = [], []
for case in qa_pairs:
    retrieved = retrieve(case["question"])
    precision_scores.append(context_precision_at_k(retrieved, case["source_chunk"]))
    faithfulness_scores.append(faithfulness_score(case["expected_answer"], retrieved))
    print(f"Q: {case['question']}")
    print(f"  precision@2={precision_scores[-1]:.2f}  faithfulness={faithfulness_scores[-1]:.3f}")

avg_precision = np.mean(precision_scores) if precision_scores else 0
avg_faithfulness = np.mean(faithfulness_scores) if faithfulness_scores else 0
print(f"avg precision@2: {avg_precision:.3f}, avg faithfulness: {avg_faithfulness:.3f}")

assert len(qa_pairs) > 0, "no QA pairs auto-generated"
assert avg_precision > 0, "retrieval never found source chunk"
print("OK")

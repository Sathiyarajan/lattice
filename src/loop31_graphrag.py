import requests
import psycopg2
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
pg_conn = psycopg2.connect(host="localhost", port=5432, user="postgres", password="pass", dbname="postgres")
driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "password"))

def vector_retrieve(query, k=2):
    qvec = model.encode([query])[0]
    cur = pg_conn.cursor()
    cur.execute("SELECT text FROM lattice_chunks ORDER BY embedding <-> %s::vector LIMIT %s", (qvec.tolist(), k))
    result = [r[0] for r in cur.fetchall()]
    cur.close()
    return result

def graph_retrieve(entity_name):
    with driver.session() as session:
        result = session.run("""
            MATCH (l:System {name: $name})-[:USES]->(t:Tool)-[:EXTENDS]->(base:Tool)
            RETURN t.name AS tool, base.name AS base
        """, name=entity_name)
        return [f"{r['tool']} extends {r['base']}" for r in result]

def call_llm(prompt):
    resp = requests.post("http://localhost:11434/api/generate", json={"model": "llama3.2", "prompt": prompt, "stream": False}, timeout=60)
    return resp.json()["response"].strip()

query = "What database does Lattice's vector storage tool build on?"

vec_context = vector_retrieve(query)
graph_context = graph_retrieve("Lattice")

vec_only_prompt = f"Context:\n{chr(10).join(vec_context)}\n\nQuestion: {query}\nAnswer (say 'unknown' if not in context):"
vec_only_answer = call_llm(vec_only_prompt)

combined_prompt = f"Context:\n{chr(10).join(vec_context + graph_context)}\n\nQuestion: {query}\nAnswer:"
combined_answer = call_llm(combined_prompt)

print("vector-only context:", vec_context)
print("vector-only answer:", vec_only_answer)
print("graph context:", graph_context)
print("combined answer:", combined_answer)

assert "postgres" in combined_answer.lower() or "postgresql" in combined_answer.lower(), \
    "GraphRAG failed to answer multi-hop question that vector-only context can't resolve"
print("OK")

driver.close()
pg_conn.close()

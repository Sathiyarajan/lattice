import json
import requests
from neo4j import GraphDatabase

driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "password"))

SCHEMA_DESC = """
Node labels: System(name), Tool(name, category), Document(title), Concept(name), Person(name)
Relationships: (System)-[:USES]->(Tool), (Tool)-[:EXTENDS]->(Tool), (Document)-[:MENTIONS]->(Concept),
(Person)-[:CREATED]->(Tool), (Concept)-[:RELATED_TO]->(Concept)
"""

def call_llm(prompt):
    resp = requests.post("http://localhost:11434/api/generate", json={"model": "llama3.2", "prompt": prompt, "stream": False}, timeout=60)
    return resp.json()["response"].strip()

def nl_to_cypher(question):
    prompt = f"""Given this graph schema:
{SCHEMA_DESC}

Convert this question to a single Cypher query. Return ONLY the Cypher query, no explanation, no markdown fences.

Question: {question}
Cypher:"""
    cypher = call_llm(prompt)
    cypher = cypher.strip().strip("`").replace("cypher\n", "").strip()
    return cypher

def summarize_results(question, rows):
    prompt = f"Question: {question}\nRaw results: {rows}\nGive a one-sentence natural language answer:"
    return call_llm(prompt)

test_questions = [
    "Who created Python?",
    "What tools does Lattice use?",
]

with driver.session() as session:
    for q in test_questions:
        cypher = nl_to_cypher(q)
        print(f"Q: {q}\nCypher: {cypher}")
        try:
            result = session.run(cypher)
            rows = [dict(r) for r in result]
        except Exception as e:
            rows = []
            print("  query error:", e)
        summary = summarize_results(q, rows) if rows else "no results"
        print("  rows:", rows)
        print("  summary:", summary)
        print()

print("OK — pipeline runs end-to-end (validity depends on small-model Cypher accuracy)")
driver.close()

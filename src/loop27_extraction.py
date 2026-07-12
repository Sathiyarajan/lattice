import json
import re
import requests
import psycopg2
from neo4j import GraphDatabase

pg_conn = psycopg2.connect(host="localhost", port=5432, user="postgres", password="pass", dbname="postgres")
driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "password"))

def call_llm(prompt):
    resp = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": "llama3.2", "prompt": prompt, "stream": False, "format": "json"},
        timeout=60,
    )
    return resp.json()["response"].strip()

EXTRACTION_PROMPT = """Extract entities and relationships from this text as JSON with this exact schema:
{{"triples": [{{"subject": "...", "relation": "...", "object": "..."}}]}}
Only include clear, factual relationships. Text: {text}"""

cur = pg_conn.cursor()
cur.execute("SELECT source, text FROM lattice_chunks LIMIT 3")
chunks = cur.fetchall()
cur.close()

all_triples = []
for source, text in chunks:
    raw = call_llm(EXTRACTION_PROMPT.format(text=text))
    try:
        parsed = json.loads(raw)
        triples = parsed.get("triples", [])
    except json.JSONDecodeError:
        triples = []
    for t in triples:
        if "subject" in t and "relation" in t and "object" in t:
            all_triples.append({**t, "source": source, "source_text": text})

print(f"extracted {len(all_triples)} triples")
for t in all_triples[:5]:
    print(t["subject"], "-[", t["relation"], "]->", t["object"])

with driver.session() as session:
    session.run("MATCH (n:ExtractedEntity) DETACH DELETE n")
    skipped = []
    for t in all_triples:
        rel = re.sub(r"_+", "_", re.sub(r"[^A-Z_]", "_", str(t["relation"]).upper().replace(" ", "_"))).strip("_")
        if not re.fullmatch(r"[A-Z][A-Z_]*", rel or ""):
            rel = "RELATED_TO"
        try:
            session.run(
                f"""
                MERGE (a:ExtractedEntity {{name: $subj}})
                MERGE (b:ExtractedEntity {{name: $obj}})
                MERGE (a)-[:{rel}]->(b)
                """,
                subj=t["subject"], obj=t["object"],
            )
        except Exception as e:
            skipped.append((t, str(e)))
    if skipped:
        print(f"skipped {len(skipped)} malformed triple(s): {skipped}")
    count = session.run("MATCH (n:ExtractedEntity) RETURN count(n) AS c").single()["c"]
    print("extracted entity nodes in graph:", count)

# spot-check: at least one triple's subject or object appears in source text
spot_checks = [t for t in all_triples if t["subject"].lower() in t["source_text"].lower() or t["object"].lower() in t["source_text"].lower()]
accuracy = len(spot_checks) / len(all_triples) if all_triples else 0
print(f"spot-check accuracy: {accuracy:.2f}")

assert len(all_triples) > 0, "no triples extracted"
assert accuracy > 0.3, "extraction accuracy too low against source text"
print("OK")

driver.close()
pg_conn.close()

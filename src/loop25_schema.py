"""
Lattice knowledge graph schema (drafted before ingestion code).

Node types:
  - System      {name}
  - Tool        {name, category}     e.g. pgvector, Postgres, Neo4j, FAISS
  - Document    {title, source}
  - Concept     {name}               e.g. "vector search", "graph reasoning"
  - Person      {name}               e.g. authors/creators referenced in docs

Relationship types:
  - (System)-[:USES]->(Tool)
  - (Tool)-[:EXTENDS]->(Tool)
  - (Document)-[:MENTIONS]->(Concept)
  - (Document)-[:MENTIONS]->(Tool)
  - (Concept)-[:RELATED_TO]->(Concept)
  - (Person)-[:CREATED]->(Tool)

Target questions this schema must answer:
  1. What tools does Lattice use?                         -> (System)-[:USES]->(Tool)
  2. What does pgvector extend?                            -> (Tool)-[:EXTENDS]->(Tool)
  3. Which documents mention "graph reasoning"?             -> (Document)-[:MENTIONS]->(Concept)
  4. Who created Python?                                    -> (Person)-[:CREATED]->(Tool)
  5. What concepts are related to "vector search"?          -> (Concept)-[:RELATED_TO]->(Concept)
"""
from neo4j import GraphDatabase

driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "password"))

SCHEMA_QUESTIONS = [
    "What tools does Lattice use?",
    "What does pgvector extend?",
    "Which documents mention graph reasoning?",
    "Who created Python?",
    "What concepts are related to vector search?",
]

with driver.session() as session:
    session.run("MATCH (n) DETACH DELETE n")
    session.run("""
        CREATE (lattice:System {name:'Lattice'})
        CREATE (pgvector:Tool {name:'pgvector', category:'extension'})
        CREATE (postgres:Tool {name:'PostgreSQL', category:'database'})
        CREATE (python:Tool {name:'Python', category:'language'})
        CREATE (guido:Person {name:'Guido van Rossum'})
        CREATE (doc:Document {title:'sample.txt', source:'raw'})
        CREATE (vs:Concept {name:'vector search'})
        CREATE (gr:Concept {name:'graph reasoning'})
        CREATE (lattice)-[:USES]->(pgvector)
        CREATE (pgvector)-[:EXTENDS]->(postgres)
        CREATE (guido)-[:CREATED]->(python)
        CREATE (doc)-[:MENTIONS]->(gr)
        CREATE (doc)-[:MENTIONS]->(vs)
        CREATE (vs)-[:RELATED_TO]->(gr)
    """)

    checks = {
        "What tools does Lattice use?": "MATCH (:System {name:'Lattice'})-[:USES]->(t:Tool) RETURN t.name",
        "What does pgvector extend?": "MATCH (:Tool {name:'pgvector'})-[:EXTENDS]->(t:Tool) RETURN t.name",
        "Which documents mention graph reasoning?": "MATCH (d:Document)-[:MENTIONS]->(:Concept {name:'graph reasoning'}) RETURN d.title",
        "Who created Python?": "MATCH (p:Person)-[:CREATED]->(:Tool {name:'Python'}) RETURN p.name",
        "What concepts are related to vector search?": "MATCH (:Concept {name:'vector search'})-[:RELATED_TO]->(c:Concept) RETURN c.name",
    }

    all_pass = True
    for q, cypher in checks.items():
        result = session.run(cypher)
        rows = [r.value() for r in result]
        print(f"{q} -> {rows}")
        if not rows:
            all_pass = False

assert all_pass, "schema failed to answer one or more target questions"
print("OK")
driver.close()

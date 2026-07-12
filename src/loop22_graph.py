import time
from neo4j import GraphDatabase

driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "password"))

def wait_for_neo4j(retries=20, delay=3):
    for _ in range(retries):
        try:
            with driver.session() as session:
                session.run("RETURN 1")
            return True
        except Exception:
            time.sleep(delay)
    return False

assert wait_for_neo4j(), "neo4j did not become ready in time"

with driver.session() as session:
    session.run("MATCH (n) DETACH DELETE n")
    session.run("""
        CREATE (lattice:System {name: 'Lattice'})
        CREATE (pgvector:Tool {name: 'pgvector'})
        CREATE (postgres:Tool {name: 'PostgreSQL'})
        CREATE (neo4j:Tool {name: 'Neo4j'})
        CREATE (lattice)-[:USES]->(pgvector)
        CREATE (pgvector)-[:EXTENDS]->(postgres)
        CREATE (lattice)-[:USES]->(neo4j)
    """)

    # multi-hop question: what database does Lattice's vector tool extend?
    result = session.run("""
        MATCH (l:System {name: 'Lattice'})-[:USES]->(t:Tool)-[:EXTENDS]->(base:Tool)
        RETURN t.name AS tool, base.name AS base
    """)
    rows = [dict(r) for r in result]
    print("multi-hop result:", rows)

    assert any(r["tool"] == "pgvector" and r["base"] == "PostgreSQL" for r in rows), \
        "multi-hop graph query failed to find pgvector->PostgreSQL relationship"

print("OK")
driver.close()

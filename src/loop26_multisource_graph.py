import psycopg2
from neo4j import GraphDatabase

pg_conn = psycopg2.connect(host="localhost", port=5432, user="postgres", password="pass", dbname="postgres")
driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "password"))

# structured source: chunk sources table already in postgres (lattice_chunks)
cur = pg_conn.cursor()
cur.execute("SELECT DISTINCT source FROM lattice_chunks")
structured_sources = [r[0] for r in cur.fetchall()]
cur.close()

# unstructured source: entity mentioned across docs — same real-world entity "Lattice" appears in graph already (System node) and now linked from Postgres-sourced Document nodes
with driver.session() as session:
    session.run("MATCH (d:Document) WHERE d.source_system = 'postgres' DETACH DELETE d")

    for src in structured_sources:
        session.run(
            "MERGE (d:Document {title: $title}) SET d.source_system = 'postgres'",
            title=src,
        )
        # resolve to same canonical Lattice entity instead of creating a duplicate
        session.run(
            """
            MATCH (d:Document {title: $title}), (l:System {name: 'Lattice'})
            MERGE (d)-[:DESCRIBES]->(l)
            """,
            title=src,
        )

    result = session.run("""
        MATCH (d:Document)-[:DESCRIBES]->(l:System {name: 'Lattice'})
        RETURN d.title AS title, d.source_system AS source_system
    """)
    rows = [dict(r) for r in result]
    print("merged documents describing Lattice:", rows)

    # check no duplicate System node was created for 'Lattice'
    dup_check = session.run("MATCH (s:System {name: 'Lattice'}) RETURN count(s) AS c").single()["c"]
    print("Lattice System node count:", dup_check)

assert len(rows) == len(structured_sources), "not all structured sources merged into graph"
assert dup_check == 1, "duplicate entity created instead of resolving to one node"
print("OK")

driver.close()
pg_conn.close()

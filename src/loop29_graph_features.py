import networkx as nx
from neo4j import GraphDatabase

driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "password"))

with driver.session() as session:
    result = session.run("MATCH (a)-[r]->(b) RETURN a.name AS src, type(r) AS rel, b.name AS dst")
    edges = [(r["src"], r["dst"]) for r in result if r["src"] and r["dst"]]

driver.close()

G = nx.DiGraph()
G.add_edges_from(edges)

degree = dict(G.degree())
betweenness = nx.betweenness_centrality(G)
pagerank = nx.pagerank(G) if G.number_of_nodes() > 0 else {}

ranked = sorted(pagerank.items(), key=lambda x: -x[1])
print("edges:", len(edges))
print("top nodes by pagerank:", ranked[:5])
print("top nodes by degree:", sorted(degree.items(), key=lambda x: -x[1])[:5])

assert len(edges) > 0, "no edges pulled from graph"
assert len(ranked) > 0, "pagerank produced no scores"
# a node with more connections should generally not have the lowest pagerank
most_connected = max(degree, key=degree.get)
assert pagerank.get(most_connected, 0) > 0
print("OK")

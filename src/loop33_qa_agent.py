import requests
import psycopg2
from typing import TypedDict
from langgraph.graph import StateGraph, END
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
pg_conn = psycopg2.connect(host="localhost", port=5432, user="postgres", password="pass", dbname="postgres")
driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "password"))

def call_llm(prompt):
    resp = requests.post("http://localhost:11434/api/generate", json={"model": "llama3.2", "prompt": prompt, "stream": False}, timeout=60)
    return resp.json()["response"].strip()

class QAState(TypedDict):
    question: str
    route: str
    context: list
    answer: str

def router(state: QAState) -> QAState:
    q = state["question"].lower()
    relational_signals = ["extend", "use", "created", "relationship", "connect", "chain", "multi-hop", "who made"]
    route = "graph" if any(s in q for s in relational_signals) else "vector"
    return {**state, "route": route}

def vector_node(state: QAState) -> QAState:
    qvec = model.encode([state["question"]])[0]
    cur = pg_conn.cursor()
    cur.execute("SELECT text FROM lattice_chunks ORDER BY embedding <-> %s::vector LIMIT 2", (qvec.tolist(),))
    context = [r[0] for r in cur.fetchall()]
    cur.close()
    return {**state, "context": context}

def graph_node(state: QAState) -> QAState:
    with driver.session() as session:
        result = session.run("""
            MATCH (l:System {name:'Lattice'})-[:USES]->(t:Tool)
            OPTIONAL MATCH (t)-[:EXTENDS]->(base:Tool)
            RETURN t.name AS tool, base.name AS base
        """)
        context = [f"{r['tool']}" + (f" extends {r['base']}" if r["base"] else "") for r in result]
    return {**state, "context": context}

def route_decision(state: QAState) -> str:
    return state["route"]

def answer_node(state: QAState) -> QAState:
    prompt = f"Context:\n{chr(10).join(state['context'])}\n\nQuestion: {state['question']}\nAnswer:"
    return {**state, "answer": call_llm(prompt)}

graph = StateGraph(QAState)
graph.add_node("router", router)
graph.add_node("vector", vector_node)
graph.add_node("graph_lookup", graph_node)
graph.add_node("answer", answer_node)
graph.set_entry_point("router")
graph.add_conditional_edges("router", route_decision, {"vector": "vector", "graph": "graph_lookup"})
graph.add_edge("vector", "answer")
graph.add_edge("graph_lookup", "answer")
graph.add_edge("answer", END)
workflow = graph.compile()

test_set = [
    {"question": "What does Lattice combine for retrieval?", "expected_route": "vector"},
    {"question": "What database does pgvector extend?", "expected_route": "graph"},
]

results = []
for case in test_set:
    result = workflow.invoke({"question": case["question"], "route": "", "context": [], "answer": ""})
    results.append(result)
    print(f"Q: {case['question']}")
    print(f"  route: {result['route']} (expected: {case['expected_route']})")
    print(f"  answer: {result['answer']}")

assert all(r["route"] == c["expected_route"] for r, c in zip(results, test_set)), "routing did not match expected strategy"
assert all(r["answer"] for r in results), "agent failed to produce an answer"
print("OK")

driver.close()
pg_conn.close()

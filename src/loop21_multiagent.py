import requests
import psycopg2
from typing import TypedDict
from langgraph.graph import StateGraph, END
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

class AgentState(TypedDict):
    query: str
    context: list
    draft_answer: str
    final_answer: str
    critique: str

def retriever_agent(state: AgentState) -> AgentState:
    qvec = model.encode([state["query"]])[0]
    cur = conn.cursor()
    cur.execute(
        "SELECT text FROM lattice_chunks ORDER BY embedding <-> %s::vector LIMIT 2",
        (qvec.tolist(),),
    )
    context = [r[0] for r in cur.fetchall()]
    cur.close()
    prompt = f"Answer using this context:\n{chr(10).join(context)}\n\nQuestion: {state['query']}\nAnswer:"
    draft = call_llm(prompt)
    return {**state, "context": context, "draft_answer": draft}

def critique_agent(state: AgentState) -> AgentState:
    prompt = (
        f"Context:\n{chr(10).join(state['context'])}\n\n"
        f"Question: {state['query']}\nDraft answer: {state['draft_answer']}\n\n"
        "Is the draft answer fully supported by the context? Reply with 'SUPPORTED' or 'UNSUPPORTED', "
        "then a one-sentence improved answer."
    )
    critique = call_llm(prompt)
    verdict = critique.upper().split()[0].strip(".:,")
    final = state["draft_answer"] if verdict == "SUPPORTED" else critique
    return {**state, "critique": critique, "final_answer": final}

graph = StateGraph(AgentState)
graph.add_node("retriever", retriever_agent)
graph.add_node("critic", critique_agent)
graph.set_entry_point("retriever")
graph.add_edge("retriever", "critic")
graph.add_edge("critic", END)
workflow = graph.compile()

result = workflow.invoke({"query": "What does Lattice combine?", "context": [], "draft_answer": "", "final_answer": "", "critique": ""})

print("draft:", result["draft_answer"])
print("critique:", result["critique"])
print("final:", result["final_answer"])

assert result["final_answer"], "workflow produced no final answer"
assert result["critique"], "critic agent did not run"
print("OK")

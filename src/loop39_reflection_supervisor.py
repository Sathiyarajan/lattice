from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_ollama import OllamaLLM

llm = OllamaLLM(model="llama3.2", base_url="http://localhost:11434")

# --- Reflection: agent critiques and corrects its own output ---
class ReflectState(TypedDict):
    question: str
    draft: str
    critique: str
    final: str

def draft_node(state: ReflectState) -> ReflectState:
    draft = llm.invoke(state["question"])
    return {**state, "draft": draft}

def reflect_node(state: ReflectState) -> ReflectState:
    prompt = (
        f"Question: {state['question']}\nDraft answer: {state['draft']}\n\n"
        "Is this factually correct and complete? Reply with 'OK' if fine, "
        "or 'FIX: <corrected answer>' if it needs correction."
    )
    critique = llm.invoke(prompt).strip()
    final = state["draft"] if critique.upper().startswith("OK") else critique.split("FIX:", 1)[-1].strip()
    return {**state, "critique": critique, "final": final}

reflect_graph = StateGraph(ReflectState)
reflect_graph.add_node("draft", draft_node)
reflect_graph.add_node("reflect", reflect_node)
reflect_graph.set_entry_point("draft")
reflect_graph.add_edge("draft", "reflect")
reflect_graph.add_edge("reflect", END)
reflect_app = reflect_graph.compile()

# deliberately flawed input to see if reflection catches it
flawed_result = reflect_app.invoke({"question": "Is 7 a prime number? Answer 'no' and explain why not.", "draft": "", "critique": "", "final": ""})
print("flawed-prompt draft:", flawed_result["draft"][:150])
print("critique:", flawed_result["critique"][:150])
print("final:", flawed_result["final"][:150])

# --- Supervisor: coordinates 2+ subgraph agents ---
class SupervisorState(TypedDict):
    query: str
    agent: str
    answer: str

def supervisor_route(state: SupervisorState) -> SupervisorState:
    q = state["query"].lower()
    agent = "math_agent" if any(c.isdigit() for c in q) else "knowledge_agent"
    return {**state, "agent": agent}

def math_agent(state: SupervisorState) -> SupervisorState:
    return {**state, "answer": f"[math_agent] {llm.invoke(state['query'])}"}

def knowledge_agent(state: SupervisorState) -> SupervisorState:
    return {**state, "answer": f"[knowledge_agent] {llm.invoke(state['query'])}"}

sup_graph = StateGraph(SupervisorState)
sup_graph.add_node("supervisor", supervisor_route)
sup_graph.add_node("math_agent", math_agent)
sup_graph.add_node("knowledge_agent", knowledge_agent)
sup_graph.set_entry_point("supervisor")
sup_graph.add_conditional_edges("supervisor", lambda s: s["agent"], {"math_agent": "math_agent", "knowledge_agent": "knowledge_agent"})
sup_graph.add_edge("math_agent", END)
sup_graph.add_edge("knowledge_agent", END)
sup_app = sup_graph.compile()

test_cases = [
    ("What is 15 + 27?", "math_agent"),
    ("Who wrote Romeo and Juliet?", "knowledge_agent"),
]
delegation_results = []
for q, expected in test_cases:
    r = sup_app.invoke({"query": q, "agent": "", "answer": ""})
    delegation_results.append((q, r["agent"], expected))
    print(f"Q: {q} -> delegated to {r['agent']} (expected {expected}) -> {r['answer'][:80]}")

assert flawed_result["critique"], "reflection step did not produce a critique"
assert all(a == e for _, a, e in delegation_results), "supervisor did not delegate to the correct subgraph agent"
print("OK")

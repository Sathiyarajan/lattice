from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_ollama import OllamaLLM

llm = OllamaLLM(model="llama3.2", base_url="http://localhost:11434")

def calculator(expression: str) -> str:
    try:
        return str(eval(expression, {"__builtins__": {}}, {}))
    except Exception:
        return "error: invalid expression"

class AgentState(TypedDict):
    query: str
    route: str
    plan: list
    answer: str

# --- Architecture #3: Router (classify then dispatch) ---
def classify(state: AgentState) -> AgentState:
    q = state["query"].lower()
    if any(c.isdigit() for c in q) and any(op in q for op in ["+", "-", "*", "/", "times", "plus", "minus"]):
        route = "tool"
    else:
        route = "direct"
    return {**state, "route": route}

def direct_answer(state: AgentState) -> AgentState:
    return {**state, "answer": llm.invoke(state["query"])}

def tool_answer(state: AgentState) -> AgentState:
    prompt = f"Extract only the arithmetic expression (digits and operators) from: '{state['query']}'. Reply with just the expression."
    expr = llm.invoke(prompt).strip()
    result = calculator(expr)
    return {**state, "answer": f"{expr} = {result}"}

router_graph = StateGraph(AgentState)
router_graph.add_node("classify", classify)
router_graph.add_node("direct", direct_answer)
router_graph.add_node("tool", tool_answer)
router_graph.set_entry_point("classify")
router_graph.add_conditional_edges("classify", lambda s: s["route"], {"direct": "direct", "tool": "tool"})
router_graph.add_edge("direct", END)
router_graph.add_edge("tool", END)
router_app = router_graph.compile()

# --- Plan-Do Loop: always calls a tool/step first, then decides if done ---
class PlanState(TypedDict):
    query: str
    steps_done: list
    answer: str
    done: bool

def plan_step(state: PlanState) -> PlanState:
    if len(state["steps_done"]) == 0:
        step_result = f"looked up: {state['query']}"
    else:
        step_result = f"synthesized answer from {len(state['steps_done'])} prior step(s)"
    steps = state["steps_done"] + [step_result]
    done = len(steps) >= 2
    return {**state, "steps_done": steps, "done": done}

def finalize(state: PlanState) -> PlanState:
    return {**state, "answer": f"Completed after steps: {state['steps_done']}"}

plan_graph = StateGraph(PlanState)
plan_graph.add_node("plan_step", plan_step)
plan_graph.add_node("finalize", finalize)
plan_graph.set_entry_point("plan_step")
plan_graph.add_conditional_edges("plan_step", lambda s: "finalize" if s["done"] else "plan_step", {"finalize": "finalize", "plan_step": "plan_step"})
plan_graph.add_edge("finalize", END)
plan_app = plan_graph.compile()

test_queries = [
    ("What is the capital of France?", "direct"),
    ("What is 12 times 8?", "tool"),
]

routing_results = []
for q, expected in test_queries:
    result = router_app.invoke({"query": q, "route": "", "plan": [], "answer": ""})
    routing_results.append((q, result["route"], expected, result["answer"]))
    print(f"Q: {q} -> route={result['route']} (expected {expected}) -> {result['answer']}")

plan_result = plan_app.invoke({"query": "multi-step question", "steps_done": [], "answer": "", "done": False})
print("plan-do-loop result:", plan_result["answer"])

assert all(r[1] == r[2] for r in routing_results), "router did not classify all query types correctly"
assert plan_result["done"] and len(plan_result["steps_done"]) >= 2, "plan-do loop did not execute multiple steps before finishing"
print("OK — handled direct-answer, tool-call, and multi-step plan query types")

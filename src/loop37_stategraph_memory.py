from typing import TypedDict, Annotated
import operator
from langgraph.graph import StateGraph, END
from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_postgres import PGVector

CONNECTION = "postgresql+psycopg://postgres:pass@localhost:5432/postgres"
COLLECTION = "lattice_langchain"

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = PGVector(embeddings=embeddings, collection_name=COLLECTION, connection=CONNECTION, use_jsonb=True)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
llm = OllamaLLM(model="llama3.2", base_url="http://localhost:11434")

class ChatState(TypedDict):
    history: Annotated[list, operator.add]
    question: str
    answer: str

def retrieve_and_generate(state: ChatState) -> ChatState:
    docs = retriever.invoke(state["question"])
    context = "\n".join(d.page_content for d in docs)
    history_text = "\n".join(f"Q: {h['question']}\nA: {h['answer']}" for h in state["history"])
    prompt = (
        f"Conversation so far:\n{history_text}\n\n"
        f"Context:\n{context}\n\nNew question: {state['question']}\nAnswer:"
    )
    answer = llm.invoke(prompt)
    return {"history": [{"question": state["question"], "answer": answer}], "question": state["question"], "answer": answer}

graph = StateGraph(ChatState)
graph.add_node("chat", retrieve_and_generate)
graph.set_entry_point("chat")
graph.add_edge("chat", END)
app = graph.compile()

state = {"history": [], "question": "What does Lattice combine?", "answer": ""}
state = app.invoke(state)
print("turn 1:", state["answer"])

state["question"] = "What database does it use?"
state = app.invoke(state)
print("turn 2:", state["answer"])
print("history length after 2 turns:", len(state["history"]))

assert len(state["history"]) == 2, "graph state did not accumulate chat history across turns"
assert state["history"][0]["question"] == "What does Lattice combine?"
print("OK")

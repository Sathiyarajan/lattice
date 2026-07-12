from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_postgres import PGVector

CONNECTION = "postgresql+psycopg://postgres:pass@localhost:5432/postgres"
COLLECTION = "lattice_langchain"

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = PGVector(embeddings=embeddings, collection_name=COLLECTION, connection=CONNECTION, use_jsonb=True)
llm = OllamaLLM(model="llama3.2", base_url="http://localhost:11434")

class State(TypedDict):
    question: str
    answer: str

def retrieve_and_answer(state: State) -> State:
    docs = vectorstore.similarity_search(state["question"], k=2)
    context = "\n".join(d.page_content for d in docs)
    answer = llm.invoke(f"Context:\n{context}\n\nQuestion: {state['question']}\nAnswer:")
    return {**state, "answer": answer}

graph = StateGraph(State)
graph.add_node("answer", retrieve_and_answer)
graph.set_entry_point("answer")
graph.add_edge("answer", END)

# module-level compiled graph, referenced by langgraph.json for the dev server / platform
app = graph.compile()

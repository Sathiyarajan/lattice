import os
import time
from dotenv import load_dotenv
load_dotenv(os.path.expanduser("~/projects/lattice/.env"))

from langsmith import Client, traceable
from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_postgres import PGVector

CONNECTION = "postgresql+psycopg://postgres:pass@localhost:5432/postgres"
COLLECTION = "lattice_langchain"

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = PGVector(embeddings=embeddings, collection_name=COLLECTION, connection=CONNECTION, use_jsonb=True)
llm = OllamaLLM(model="llama3.2", base_url="http://localhost:11434")
client = Client()

@traceable(name="retrieve")
def retrieve(query, k=2):
    return vectorstore.similarity_search(query, k=k)

@traceable(name="generate")
def generate(query, docs):
    context = "\n".join(d.page_content for d in docs)
    return llm.invoke(f"Context:\n{context}\n\nQuestion: {query}\nAnswer:")

@traceable(name="rag_pipeline", tags=["production", "lattice"])
def rag_pipeline(query):
    docs = retrieve(query)
    answer = generate(query, docs)
    return answer

# simulate a handful of "production" queries, each fully traced
production_queries = [
    "What does Lattice combine for retrieval?",
    "What database does Lattice use?",
    "How does Lattice chunk documents?",
]

run_ids = []
for q in production_queries:
    answer = rag_pipeline(q)
    print(f"Q: {q}\n  A: {answer[:100]}")

# give LangSmith a moment to ingest the traces, then verify they're queryable
time.sleep(3)
recent_runs = list(client.list_runs(project_name=os.environ.get("LANGCHAIN_PROJECT", "lattice"), limit=5))
print(f"traced runs visible via API: {len(recent_runs)}")

# feedback collection: attach a thumbs-up/down style score to the most recent run
if recent_runs:
    latest_run = recent_runs[0]
    client.create_feedback(run_id=latest_run.id, key="user_score", score=1, comment="grounded and relevant")
    print(f"attached feedback to run {latest_run.id}")

assert len(recent_runs) > 0, "no traces found in LangSmith for the production_queries run"
print("OK — traces are queryable and feedback was recorded; inspect in LangSmith UI under project 'lattice'")

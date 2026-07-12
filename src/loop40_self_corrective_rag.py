import os
from dotenv import load_dotenv
load_dotenv(os.path.expanduser("~/projects/lattice/.env"))

from langsmith import Client
from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_postgres import PGVector

CONNECTION = "postgresql+psycopg://postgres:pass@localhost:5432/postgres"
COLLECTION = "lattice_langchain"

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = PGVector(embeddings=embeddings, collection_name=COLLECTION, connection=CONNECTION, use_jsonb=True)
llm = OllamaLLM(model="llama3.2", base_url="http://localhost:11434")

def retrieve(query, k=2):
    return vectorstore.similarity_search(query, k=k)

def grade_retrieval(query, docs):
    context = "\n".join(d.page_content for d in docs)
    prompt = (
        f"Question: {query}\nRetrieved context:\n{context}\n\n"
        "Does this context contain enough information to answer the question? Reply only 'YES' or 'NO'."
    )
    return llm.invoke(prompt).strip().upper().startswith("YES")

def rewrite_query(query):
    prompt = (
        f"Rewrite this question to be more specific and retrieval-friendly. "
        f"Reply with ONLY the rewritten question, one line, no options, no explanation.\n\nQuestion: {query}"
    )
    rewritten = llm.invoke(prompt).strip().split("\n")[0].strip().strip('"')
    return rewritten or query

def self_corrective_rag(query, max_retries=2):
    attempts = []
    current_query = query
    for attempt in range(max_retries + 1):
        docs = retrieve(current_query)
        confident = grade_retrieval(current_query, docs)
        attempts.append({"query": current_query, "confident": confident, "n_docs": len(docs)})
        if confident:
            break
        current_query = rewrite_query(current_query)
    context = "\n".join(d.page_content for d in docs)
    answer = llm.invoke(f"Context:\n{context}\n\nQuestion: {query}\nAnswer:")
    return {"answer": answer, "attempts": attempts}

# ambiguous query likely to need at least one rewrite
result = self_corrective_rag("Tell me about the storage thing")
print("attempts:", result["attempts"])
print("answer:", result["answer"])

# --- LangSmith eval dataset ---
client = Client()
DATASET_NAME = "lattice-loop40-eval"

existing = list(client.list_datasets(dataset_name=DATASET_NAME))
if existing:
    dataset = existing[0]
else:
    dataset = client.create_dataset(DATASET_NAME, description="Lattice RAG eval set, Loop 40")
    examples = [
        {"inputs": {"question": "What does Lattice combine for retrieval?"}, "outputs": {"expected_keyword": "vector"}},
        {"inputs": {"question": "What database extension does Lattice use?"}, "outputs": {"expected_keyword": "pgvector"}},
    ]
    for ex in examples:
        client.create_example(inputs=ex["inputs"], outputs=ex["outputs"], dataset_id=dataset.id)

stored_examples = list(client.list_examples(dataset_id=dataset.id))
print(f"LangSmith dataset '{DATASET_NAME}' has {len(stored_examples)} examples")

scores = []
for ex in stored_examples:
    q = ex.inputs["question"]
    expected_kw = ex.outputs["expected_keyword"]
    r = self_corrective_rag(q)
    hit = expected_kw.lower() in r["answer"].lower()
    scores.append(hit)
    print(f"  eval: {q!r} -> contains {expected_kw!r}: {hit}")

reproduced = list(client.list_examples(dataset_id=dataset.id))
assert len(reproduced) == len(stored_examples), "dataset not reproducible across list calls"
assert len(result["attempts"]) >= 1
print("OK — LangSmith-tracked baseline established:", sum(scores), "/", len(scores), "passed")

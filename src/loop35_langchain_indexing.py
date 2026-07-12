from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_postgres import PGVector
import os

RAW_DIR = os.path.expanduser("~/projects/lattice/data/raw")
CONNECTION = "postgresql+psycopg://postgres:pass@localhost:5432/postgres"
COLLECTION = "lattice_langchain"

# 1. load via LangChain document loader
loader = TextLoader(os.path.join(RAW_DIR, "sample.txt"))
docs = loader.load()
assert len(docs) == 1 and len(docs[0].page_content) > 0

# 2. split via LangChain text splitter
splitter = RecursiveCharacterTextSplitter(chunk_size=150, chunk_overlap=15)
chunks = splitter.split_documents(docs)
print(f"split into {len(chunks)} chunks")
assert len(chunks) >= 1

# 3. embed + index via LangChain's PGVector integration (drop-in for custom Loop 9 pipeline)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = PGVector(
    embeddings=embeddings,
    collection_name=COLLECTION,
    connection=CONNECTION,
    use_jsonb=True,
)
vectorstore.add_documents(chunks)

# 4. retrieve and compare quality against custom pipeline (Loop 9) on the same query
query = "What does Lattice combine for retrieval?"
results = vectorstore.similarity_search(query, k=2)
print("LangChain PGVector results:")
for r in results:
    print(" -", r.page_content[:80])

assert any("vector search" in r.page_content.lower() or "graph" in r.page_content.lower() for r in results), \
    "LangChain-indexed retrieval did not surface the expected content"
print("OK")

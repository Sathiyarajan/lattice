import re

def chunk_text(text, source, max_chars=200, overlap=20):
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    chunks = []
    current = ""
    for sent in sentences:
        if len(current) + len(sent) + 1 > max_chars and current:
            chunks.append(current.strip())
            current = current[-overlap:] + " " + sent
        else:
            current = (current + " " + sent).strip()
    if current.strip():
        chunks.append(current.strip())

    return [
        {"source": source, "position": i, "text": c}
        for i, c in enumerate(chunks)
    ]

sample = (
    "Lattice is a system for building retrieval-augmented generation pipelines. "
    "It combines vector search with graph-based reasoning. "
    "This allows multi-hop questions to be answered accurately. "
    "The system uses PostgreSQL with pgvector for storage. "
    "Embeddings are generated using sentence-transformers models."
)

chunks = chunk_text(sample, "sample.txt", max_chars=100, overlap=10)
for c in chunks:
    print(c["position"], repr(c["text"]))

assert len(chunks) > 1, "expected multiple chunks"
assert all(not c["text"].startswith((" ", "and", "the")) for c in chunks), "chunk starts mid-word oddly"
assert all(len(c["text"]) <= 130 for c in chunks), "chunk too large"
print("OK")

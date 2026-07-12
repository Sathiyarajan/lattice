import re

ABBREVIATIONS = {
    "e.g.": "for example",
    "i.e.": "that is",
    "etc.": "and so on",
    "w/": "with",
}

def normalize_abbreviations(text):
    for abbr, full in ABBREVIATIONS.items():
        text = text.replace(abbr, full)
    return text

def recursive_split(text, max_chars=150, separators=("\n\n", "\n", ". ", " ")):
    if len(text) <= max_chars:
        return [text.strip()] if text.strip() else []
    for sep in separators:
        if sep in text:
            parts = text.split(sep)
            chunks, current = [], ""
            for part in parts:
                candidate = (current + sep + part) if current else part
                if len(candidate) > max_chars and current:
                    chunks.extend(recursive_split(current, max_chars, separators))
                    current = part
                else:
                    current = candidate
            if current:
                chunks.extend(recursive_split(current, max_chars, separators))
            return chunks
    return [text[i:i+max_chars] for i in range(0, len(text), max_chars)]

def generate_hypothetical_questions(chunk_text):
    # lightweight heuristic stand-in (no LLM call): turn key noun phrase into a question
    first_sentence = chunk_text.split(".")[0]
    return f"What is described by: {first_sentence}?"

sample = "Lattice uses pgvector, e.g. for storing embeddings. It also supports BM25, i.e. keyword search, etc."
normalized = normalize_abbreviations(sample)
print("normalized:", normalized)
assert "for example" in normalized and "that is" in normalized

long_text = (
    "Lattice is a retrieval-augmented generation system. "
    "It combines vector search with graph reasoning for multi-hop questions. "
    "The storage layer uses PostgreSQL with the pgvector extension. "
    "Chunking is sentence-aware with configurable overlap."
)
chunks = recursive_split(long_text, max_chars=80)
print("chunks:", chunks)
assert all(len(c) <= 100 for c in chunks)
assert len(chunks) > 1

hq = generate_hypothetical_questions(chunks[0])
print("hypothetical question:", hq)
assert hq.endswith("?")
print("OK")

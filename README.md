# Lattice

Incremental RAG / GraphRAG learning build. 34 loops (0-33), each a standalone script in `src/`, following `lattice-loop-engineering-prompt.md`. Local-only: Postgres+pgvector, Neo4j, and Ollama all run in WSL/Docker on localhost.

## Docs

- `docs/architecture.md` — class diagrams (Postgres schema, Neo4j schema, agent state machines, GNN model) and system data-flow.
- `CLAUDE.md` — environment/conventions for an AI agent (or human) picking this repo back up.
- `CHANGE_LOG.md` — what was added/fixed, loop by loop.
- `ISSUES.md` — every bug/blocker hit during the build, root cause, fix.
- `SECURITY_NOTICE.md` — `eval()`/dynamic-Cypher risk, local-dev credentials, no-auth surfaces — read before reusing anything beyond localhost.

## Quick start

```bash
cd ~/projects/lattice
source lattice-env/bin/activate
docker ps          # confirm lattice-pg and lattice-neo4j are Up
curl -s localhost:11434/api/version   # confirm Ollama is running
python src/loop33_qa_agent.py         # run the final integrated agent
```

Run any earlier loop the same way: `python src/loopNN_name.py`. Each script prints its own progress and ends with `OK` on success.

## Loop index

- [Loop 0 — Environment Setup](#loop-0-environment-setup)
- [Loop 1 — Embeddings](#loop-1-embeddings)
- [Loop 2 — Naive Vector Store (SQLite)](#loop-2-naive-vector-store-sqlite)
- [Loop 3 — Real Vector Index (FAISS)](#loop-3-real-vector-index-faiss)
- [Loop 4 — Production Vector DB (pgvector)](#loop-4-production-vector-db-pgvector)
- [Loop 5 — RAG Pipeline](#loop-5-rag-pipeline)
- [Loop 6 — Metadata Filtering (Hardening)](#loop-6-metadata-filtering-hardening)
- [Loop 7 — Real Document Ingestion](#loop-7-real-document-ingestion)
- [Loop 8 — Chunking Strategy](#loop-8-chunking-strategy)
- [Loop 9 — Production Embedding + Storage Pipeline](#loop-9-production-embedding-storage-pipeline)
- [Loop 10 — Two-Stage Retrieval (Hybrid + Reranking)](#loop-10-two-stage-retrieval-hybrid-reranking)
- [Loop 11 — Prompt Assembly + Generation](#loop-11-prompt-assembly-generation)
- [Loop 12 — Hallucination Guardrails](#loop-12-hallucination-guardrails)
- [Loop 13 — Evaluation Harness](#loop-13-evaluation-harness)
- [Loop 14 — Production Hardening (API + caching)](#loop-14-production-hardening-api-caching)
- [Loop 15 — Multimodal/Table Extension](#loop-15-multimodaltable-extension)
- [Loop 16 — Multi-Format Data Loading](#loop-16-multi-format-data-loading)
- [Loop 17 — Preprocessing Upgrades](#loop-17-preprocessing-upgrades)
- [Loop 18 — Embedding Model Comparison](#loop-18-embedding-model-comparison)
- [Loop 19 — Hybrid Search + Reranking (Multi-Query)](#loop-19-hybrid-search-reranking-multi-query)
- [Loop 20 — Structured Output + Tool-Calling Agent](#loop-20-structured-output-tool-calling-agent)
- [Loop 21 — Multi-Agent Workflow (LangGraph)](#loop-21-multi-agent-workflow-langgraph)
- [Loop 22 — Knowledge Graph Layer (Neo4j)](#loop-22-knowledge-graph-layer-neo4j)
- [Loop 23 — Rigorous Evaluation Suite](#loop-23-rigorous-evaluation-suite)
- [Loop 24 — Deployment (Streamlit + Docker)](#loop-24-deployment-streamlit-docker)
- [Loop 25 — KG Design from Ontology](#loop-25-kg-design-from-ontology)
- [Loop 26 — Multisource Graph Integration](#loop-26-multisource-graph-integration)
- [Loop 27 — LLM-Driven Entity & Relationship Extraction](#loop-27-llm-driven-entity-relationship-extraction)
- [Loop 28 — Named Entity Disambiguation (NED)](#loop-28-named-entity-disambiguation-ned)
- [Loop 29 — Graph Feature Engineering + ML Primer](#loop-29-graph-feature-engineering-ml-primer)
- [Loop 30 — Graph Representation Learning (GNN)](#loop-30-graph-representation-learning-gnn)
- [Loop 31 — KG-Powered RAG (GraphRAG)](#loop-31-kg-powered-rag-graphrag)
- [Loop 32 — Natural Language to Cypher (KG Q&A)](#loop-32-natural-language-to-cypher-kg-qa)
- [Loop 33 — QA Agent with LangGraph (final integration)](#loop-33-qa-agent-with-langgraph-final-integration)
- [Loop 34 — LangChain Fundamentals Setup](#loop-34-langchain-fundamentals-setup)
- [Loop 35 — RAG Indexing with LangChain](#loop-35-rag-indexing-with-langchain)
- [Loop 36 — RAG Retrieval + Generation Chain](#loop-36-rag-retrieval-generation-chain)
- [Loop 37 — Introducing LangGraph (StateGraph)](#loop-37-introducing-langgraph-stategraph)
- [Loop 38 — Agent Architectures (Router, Plan-Do Loop, Tool-Calling)](#loop-38-agent-architectures-router-plan-do-loop-tool-calling)
- [Loop 39 — Reflection, Subgraphs, and Multi-Agent Supervisor](#loop-39-reflection-subgraphs-and-multi-agent-supervisor)
- [Loop 40 — Self-Corrective RAG + Testing](#loop-40-self-corrective-rag-testing)
- [Loop 41 — Production Tracing & Monitoring](#loop-41-production-tracing-monitoring)
- [Loop 42 — Deploy to LangGraph Platform](#loop-42-deploy-to-langgraph-platform)
- [Loop 43 — Application Layer (Chatbot / Collaborative / Ambient)](#loop-43-application-layer-chatbot-collaborative-ambient)

---

## Loop 0 — Environment Setup

**Builds on:** —
**Script:** `(no script — manual environment setup)`

### Goal
See `lattice-loop-engineering-prompt.md` (Loop 0 section) for the original spec.

### Dependencies
python3-venv, postgresql, docker.io, sentence-transformers, faiss-cpu, psycopg2-binary, numpy, pandas

### How to run
```bash
cd ~/projects/lattice
source lattice-env/bin/activate
python3 -m venv lattice-env && source lattice-env/bin/activate && pip install sentence-transformers faiss-cpu psycopg2-binary numpy pandas
```

### What the test verifies (QA)
python -c "import faiss, sentence_transformers, psycopg2, numpy, pandas" exits 0

### Manual QA steps
Run the import check above inside the activated venv. No output = pass.

### Unit testing this loop
N/A (environment check, not a unit under test). CI equivalent: run the same import line in a fresh venv build step.

### Issues encountered
python3.14-venv package was missing from apt despite python3-venv being requested; venv creation failed with 'ensurepip is not available' until `sudo apt install -y python3.14-venv` was run explicitly. sudo required an interactive password each time, blocking non-interactive automation.

---

## Loop 1 — Embeddings

**Builds on:** Loop 0
**Script:** `src/loop1_embeddings.py`

### Goal
See `lattice-loop-engineering-prompt.md` (Loop 1 section) for the original spec.

### Dependencies
sentence-transformers (all-MiniLM-L6-v2)

### How to run
```bash
cd ~/projects/lattice
source lattice-env/bin/activate
python src/loop1_embeddings.py
```

### What the test verifies (QA)
Vector dimension is consistent (384) across inputs; cosine similarity of semantically related sentences > unrelated sentences.

### Manual QA steps
Run script, confirm printed 'OK' and sim(cat,feline) > sim(cat,stocks).

### Unit testing this loop
Wrap `model.encode` in a fixture, assert `vec.shape == (384,)` and `cos_sim(a,b) > cos_sim(a,c)` for a known related/unrelated triplet.

### Issues encountered
None — worked first try.

---

## Loop 2 — Naive Vector Store (SQLite)

**Builds on:** Loop 1
**Script:** `src/loop2_sqlite_store.py`

### Goal
See `lattice-loop-engineering-prompt.md` (Loop 2 section) for the original spec.

### Dependencies
sqlite3 (stdlib), numpy, sentence-transformers

### How to run
```bash
cd ~/projects/lattice
source lattice-env/bin/activate
python src/loop2_sqlite_store.py
```

### What the test verifies (QA)
Brute-force cosine search over embeddings stored as JSON text in SQLite returns correct nearest neighbor.

### Manual QA steps
Run script, confirm top result is 'A feline rested on the rug.' for query 'kittens on furniture'.

### Unit testing this loop
Seed a small in-memory SQLite DB with known vectors, assert `search(query)[0]` matches the expected nearest text.

### Issues encountered
None — worked first try.

---

## Loop 3 — Real Vector Index (FAISS)

**Builds on:** Loop 2
**Script:** `src/loop3_faiss_index.py`

### Goal
See `lattice-loop-engineering-prompt.md` (Loop 3 section) for the original spec.

### Dependencies
faiss-cpu, sentence-transformers, numpy

### How to run
```bash
cd ~/projects/lattice
source lattice-env/bin/activate
python src/loop3_faiss_index.py
```

### What the test verifies (QA)
FAISS IndexFlatIP top-3 results match a brute-force baseline, and search is faster.

### Manual QA steps
Run script, confirm 'OK' and observe faiss_time < brute_time in printed ms.

### Unit testing this loop
Build a small FAISS index from fixed vectors, assert `index.search` returns the same top-k *texts* as a numpy brute-force dot product (not raw indices, since duplicate rows tie).

### Issues encountered
Test data had 5 sentences duplicated 200x for scale; comparing result *indices* between brute-force and FAISS failed because duplicate rows tie and get selected in different order. Fixed by comparing the retrieved *text sets* instead of index sets.

---

## Loop 4 — Production Vector DB (pgvector)

**Builds on:** Loop 3
**Script:** `src/loop4_pgvector.py`

### Goal
See `lattice-loop-engineering-prompt.md` (Loop 4 section) for the original spec.

### Dependencies
psycopg2-binary, pgvector Postgres extension, Docker (lattice-pg container)

### How to run
```bash
cd ~/projects/lattice
source lattice-env/bin/activate
docker run -d --name lattice-pg -e POSTGRES_PASSWORD=pass -p 5432:5432 pgvector/pgvector:pg16 && python src/loop4_pgvector.py
```

### What the test verifies (QA)
`embedding <-> query` vector distance operator returns the same nearest neighbor as the FAISS/SQLite versions.

### Manual QA steps
Confirm `docker ps` shows lattice-pg Up, then run script and check 'OK'.

### Unit testing this loop
Point a test at a disposable Postgres/pgvector container (or testcontainers), insert fixed rows, assert `ORDER BY embedding <-> %s LIMIT 1` returns the expected row.

### Issues encountered
Needed `sudo service docker start` before Docker was usable inside WSL; sudo password prompt again blocked non-interactive automation and had to be run manually by the user.

---

## Loop 5 — RAG Pipeline

**Builds on:** Loop 4
**Script:** `src/loop5_rag.py`

### Goal
See `lattice-loop-engineering-prompt.md` (Loop 5 section) for the original spec.

### Dependencies
requests, psycopg2-binary, sentence-transformers, Ollama (llama3.2 model) running on localhost:11434

### How to run
```bash
cd ~/projects/lattice
source lattice-env/bin/activate
ollama serve & 
ollama pull llama3.2
python src/loop5_rag.py
```

### What the test verifies (QA)
Retrieved context + LLM-generated answer is grounded in the retrieved fact (contains '1889' for the Eiffel Tower question).

### Manual QA steps
Confirm `ollama list` shows llama3.2, run script, check answer contains the expected grounded fact.

### Unit testing this loop
Mock `requests.post` to Ollama's `/api/generate` with a canned response; assert `generate()` embeds retrieved context into the prompt sent, and that grounding-assertion logic flags ungrounded responses.

### Issues encountered
`ollama` binary was not installed; `curl -fsSL https://ollama.com/install.sh | sh` had to be run manually (non-sudo, but needed a real terminal). Model pull is a ~2GB download, run in background.

---

## Loop 6 — Metadata Filtering (Hardening)

**Builds on:** Loop 5
**Script:** `src/loop6_metadata_filter.py`

### Goal
See `lattice-loop-engineering-prompt.md` (Loop 6 section) for the original spec.

### Dependencies
psycopg2-binary, sentence-transformers

### How to run
```bash
cd ~/projects/lattice
source lattice-env/bin/activate
python src/loop6_metadata_filter.py
```

### What the test verifies (QA)
`WHERE category = %s` filter returns only rows of that category and never leaks other categories.

### Manual QA steps
Run script, confirm 'filtered (tech)' list contains only tech-category docs.

### Unit testing this loop
Insert mixed-category rows, assert filtered query returns exactly the expected subset and unfiltered query returns the superset.

### Issues encountered
None — worked first try.

---

## Loop 7 — Real Document Ingestion

**Builds on:** Loop 6
**Script:** `src/loop7_ingest.py`

### Goal
See `lattice-loop-engineering-prompt.md` (Loop 7 section) for the original spec.

### Dependencies
pypdf, beautifulsoup4

### How to run
```bash
cd ~/projects/lattice
source lattice-env/bin/activate
python src/loop7_ingest.py
```

### What the test verifies (QA)
Parsers for .txt, .html, .pdf all extract non-empty clean text.

### Manual QA steps
Run script, confirm each source file prints char count > 0 and 'OK'.

### Unit testing this loop
Feed each parser a minimal fixture file (txt/html/pdf) and assert `len(parse(path)) > 0` and no HTML tags/binary junk leak into the text.

### Issues encountered
No real PDF fixture was generated (reportlab not installed) — pdf parser path is implemented but only exercised structurally, not with a live test file. Documented as a fallback: start with txt/html, add PDF later.

---

## Loop 8 — Chunking Strategy

**Builds on:** Loop 7
**Script:** `src/loop8_chunking.py`

### Goal
See `lattice-loop-engineering-prompt.md` (Loop 8 section) for the original spec.

### Dependencies
stdlib re

### How to run
```bash
cd ~/projects/lattice
source lattice-env/bin/activate
python src/loop8_chunking.py
```

### What the test verifies (QA)
Sentence-aware chunker produces chunks under the max_chars bound with configurable overlap, no chunk starts on a stray filler word.

### Manual QA steps
Run script, inspect printed chunks, confirm 'OK'.

### Unit testing this loop
Feed a known multi-sentence string, assert `len(chunk_text(...)) > 1`, assert no returned chunk exceeds `max_chars + overlap`, assert overlap text appears at chunk boundaries.

### Issues encountered
None — worked first try.

---

## Loop 9 — Production Embedding + Storage Pipeline

**Builds on:** Loop 8
**Script:** `src/loop9_pipeline.py`

### Goal
See `lattice-loop-engineering-prompt.md` (Loop 9 section) for the original spec.

### Dependencies
psycopg2-binary, sentence-transformers, loop7_ingest.py, loop8_chunking.py

### How to run
```bash
cd ~/projects/lattice
source lattice-env/bin/activate
python src/loop9_pipeline.py
```

### What the test verifies (QA)
Every generated chunk is embedded and inserted into `lattice_chunks`; row count == chunk count; a spot-checked row has an embedding of the expected dimension.

### Manual QA steps
Run script, confirm 'chunks: N rows: N' matches and 'OK'.

### Unit testing this loop
After running the pipeline against a fixture doc set, assert `SELECT COUNT(*) FROM lattice_chunks` equals the number of chunks produced by `chunk_text`, and that `embedding` is non-null for every row.

### Issues encountered
psycopg2 returns pgvector `embedding` columns as a string, not a Python list, so `len(sample[2])` measured string length (e.g. character count) instead of vector dimensionality and failed the assertion. Fixed by counting commas in the string to derive dimension instead of using `len()` directly.

---

## Loop 10 — Two-Stage Retrieval (Hybrid + Reranking)

**Builds on:** Loop 9
**Script:** `src/loop10_hybrid.py`

### Goal
See `lattice-loop-engineering-prompt.md` (Loop 10 section) for the original spec.

### Dependencies
rank-bm25, sentence-transformers (bi-encoder + cross-encoder ms-marco-MiniLM-L-6-v2), numpy

### How to run
```bash
cd ~/projects/lattice
source lattice-env/bin/activate
python src/loop10_hybrid.py
```

### What the test verifies (QA)
Combined BM25 + vector score (normalized, alpha-weighted) followed by cross-encoder reranking surfaces the correct docs above vector-only search for a keyword+semantic query.

### Manual QA steps
Run script, confirm 'reranked' list contains both the Python and PostgreSQL docs, 'OK'.

### Unit testing this loop
Assert `hybrid_search()` output changes ranking relative to `vector_search()` for a query with a strong keyword signal that vector search alone under-weights; assert `rerank()` output is a strict subset of its input candidates.

### Issues encountered
`numpy.ndarray.ptp()` method was removed in the installed NumPy version, raising `AttributeError: 'numpy.ndarray' object has no attribute 'ptp'`. Fixed by switching to the free function `np.ptp(arr)`.

---

## Loop 11 — Prompt Assembly + Generation

**Builds on:** Loop 10
**Script:** `src/loop11_prompt_gen.py`

### Goal
See `lattice-loop-engineering-prompt.md` (Loop 11 section) for the original spec.

### Dependencies
requests, psycopg2-binary, sentence-transformers, Ollama

### How to run
```bash
cd ~/projects/lattice
source lattice-env/bin/activate
python src/loop11_prompt_gen.py
```

### What the test verifies (QA)
Generated answer includes an inline citation marker (`[1]` or `[2]`) referencing the numbered context block built from retrieved chunks.

### Manual QA steps
Run script, confirm printed answer contains a `[n]` marker, 'OK'.

### Unit testing this loop
Mock the LLM call to return a fixed string containing `[1]`; assert `build_prompt()` numbers each chunk and includes its source/chunk_id in the prompt text.

### Issues encountered
None — worked first try.

---

## Loop 12 — Hallucination Guardrails

**Builds on:** Loop 11
**Script:** `src/loop12_guardrails.py`

### Goal
See `lattice-loop-engineering-prompt.md` (Loop 12 section) for the original spec.

### Dependencies
sentence-transformers, numpy

### How to run
```bash
cd ~/projects/lattice
source lattice-env/bin/activate
python src/loop12_guardrails.py
```

### What the test verifies (QA)
A known-good (grounded) answer passes the guardrail; a known-bad (fabricated) answer is flagged.

### Manual QA steps
Run script, confirm 'good: ... True' and 'bad: ... False', 'OK'.

### Unit testing this loop
Assert `is_grounded(good_answer, context)` is True and `is_grounded(bad_answer, context)` is False for fixed fixture pairs; assert the function is monotonic (higher overlap/similarity never flips a pass to a fail).

### Issues encountered
Pure cosine-similarity grounding score alone was too permissive — a fabricated answer ('aliens... Mars... year 3000') still scored 0.61 similarity against real context and passed threshold 0.5. Fixed per the loop's documented fallback: added a keyword-overlap heuristic and required BOTH cosine similarity AND keyword overlap to pass.

---

## Loop 13 — Evaluation Harness

**Builds on:** Loop 12
**Script:** `src/loop13_eval.py`

### Goal
See `lattice-loop-engineering-prompt.md` (Loop 13 section) for the original spec.

### Dependencies
psycopg2-binary, sentence-transformers

### How to run
```bash
cd ~/projects/lattice
source lattice-env/bin/activate
python src/loop13_eval.py
```

### What the test verifies (QA)
Hand-written context-precision@k metric is reproducible across repeated runs on the same data and produces a nonzero baseline score.

### Manual QA steps
Run script twice (or note the script runs it twice internally), confirm both runs produce identical precision@2 scores, 'OK'.

### Unit testing this loop
Assert `run_eval(cases)` returns the same float across two consecutive calls given unchanged data; assert precision-at-k stays within [0,1].

### Issues encountered
`ragas` and `deepeval` install cleanly but their default LLM-as-judge metrics require a hosted API key (OpenAI) not available in this environment. Used the loop's documented fallback instead: hand-written context-precision@k with no external judge dependency.

---

## Loop 14 — Production Hardening (API + caching)

**Builds on:** Loop 13
**Script:** `src/loop14_api.py`

### Goal
See `lattice-loop-engineering-prompt.md` (Loop 14 section) for the original spec.

### Dependencies
fastapi, uvicorn, functools.lru_cache (in-process; redis installed but not required by the fallback used)

### How to run
```bash
cd ~/projects/lattice
source lattice-env/bin/activate
uvicorn src.loop14_api:app --host 0.0.0.0 --port 8000
curl 'localhost:8000/query?q=...'
```

### What the test verifies (QA)
`/query` endpoint returns retrieval results; a repeated identical query is served from cache and is dramatically faster than the first (cold) call.

### Manual QA steps
curl the endpoint twice with the same `q`, compare `latency_ms` in the JSON body of the first call vs `time_total` of the curl timing on the second.

### Unit testing this loop
Use FastAPI's `TestClient`, call `/query?q=X` twice, assert both return 200 and identical `results`, and assert the underlying `cached_retrieve` function's call count only increments once (via `cached_retrieve.cache_info().hits`).

### Issues encountered
First curl attempt got `exit 7` (connection refused) because the model was still loading inside the background-started uvicorn process; had to wait for the 'Application startup complete' log line before the port accepted connections.

---

## Loop 15 — Multimodal/Table Extension

**Builds on:** Loop 14
**Script:** `src/loop15_table_parsing.py`

### Goal
See `lattice-loop-engineering-prompt.md` (Loop 15 section) for the original spec.

### Dependencies
pandas

### How to run
```bash
cd ~/projects/lattice
source lattice-env/bin/activate
python src/loop15_table_parsing.py
```

### What the test verifies (QA)
CSV/table rows are converted into a flat 'col: value' text representation retrievable like any other chunk.

### Manual QA steps
Run script, confirm 'Widget'/'9.99' appear in output and 'OK'.

### Unit testing this loop
Feed a fixed CSV string, assert `table_to_text()` produces one text line per data row (excluding header) with all column values present.

### Issues encountered
None — worked first try. (Chose the 'lightest' fallback option — table parsing — over graph/agentic extensions per the loop's own guidance.)

---

## Loop 16 — Multi-Format Data Loading

**Builds on:** Loop 15
**Script:** `src/loop16_multiformat.py`

### Goal
See `lattice-loop-engineering-prompt.md` (Loop 16 section) for the original spec.

### Dependencies
python-docx, openpyxl, sqlalchemy/psycopg2

### How to run
```bash
cd ~/projects/lattice
source lattice-env/bin/activate
python src/loop16_multiformat.py
```

### What the test verifies (QA)
Loaders for .docx, .xlsx, and a live Postgres table all produce non-empty unified text, alongside the .txt/.html loaders from Loop 7 (5 formats total).

### Manual QA steps
Run script, confirm each of the 3 '--- format ---' sections prints non-empty text, 'OK'.

### Unit testing this loop
For each format, assert `parse_X(fixture_path)` returns a non-empty string containing a known token from the fixture (e.g. 'Widget' for xlsx, a known paragraph for docx).

### Issues encountered
None blocking — the Postgres table dump in the printed sample output showed column values in an unexpected order (column/value pairs did not visually align to header names in the console print), but the assertions on non-empty content still passed; this is a cosmetic issue worth revisiting if that dump is surfaced to users.

---

## Loop 17 — Preprocessing Upgrades

**Builds on:** Loop 16
**Script:** `src/loop17_preprocessing.py`

### Goal
See `lattice-loop-engineering-prompt.md` (Loop 17 section) for the original spec.

### Dependencies
stdlib re

### How to run
```bash
cd ~/projects/lattice
source lattice-env/bin/activate
python src/loop17_preprocessing.py
```

### What the test verifies (QA)
Abbreviation normalization expands common abbreviations (e.g., i.e., etc., w/); recursive splitter respects max_chars; a lightweight hypothetical-question generator produces a question-formatted string.

### Manual QA steps
Run script, confirm normalized text contains 'for example'/'that is', all chunks <= bound, hypothetical question ends in '?', 'OK'.

### Unit testing this loop
Assert `normalize_abbreviations('e.g.')` expands to 'for example'; assert `recursive_split(text, max_chars=N)` never returns a chunk longer than N (+ small tolerance); assert `generate_hypothetical_questions(text).endswith('?')`.

### Issues encountered
None — worked first try. Hypothetical-question generation uses a lightweight heuristic instead of an LLM call, since Ollama round-trips were already the bottleneck elsewhere in the pipeline; documented as a stand-in, swap for an LLM call for production quality.

---

## Loop 18 — Embedding Model Comparison

**Builds on:** Loop 17
**Script:** `src/loop18_model_compare.py`

### Goal
See `lattice-loop-engineering-prompt.md` (Loop 18 section) for the original spec.

### Dependencies
umap-learn, scikit-learn, matplotlib, sentence-transformers

### How to run
```bash
cd ~/projects/lattice
source lattice-env/bin/activate
python src/loop18_model_compare.py
```

### What the test verifies (QA)
Two embedding models (all-MiniLM-L6-v2, paraphrase-MiniLM-L3-v2) are compared via intra-cluster vs inter-cluster cosine similarity; the model with higher separation is selected with a documented reason.

### Manual QA steps
Run script, confirm both models print intra > inter, and the chosen model line prints, 'OK'.

### Unit testing this loop
Assert `intra_vs_inter_cluster_sim(vecs, labels)` returns `intra > inter` for any embedding model tested against clearly-separated label groups (animal vs finance sentences).

### Issues encountered
None — worked first try.

---

## Loop 19 — Hybrid Search + Reranking (Multi-Query)

**Builds on:** Loop 18
**Script:** `src/loop19_multiquery.py`

### Goal
See `lattice-loop-engineering-prompt.md` (Loop 19 section) for the original spec.

### Dependencies
sentence-transformers (bi-encoder + cross-encoder), Ollama, numpy, requests

### How to run
```bash
cd ~/projects/lattice
source lattice-env/bin/activate
python src/loop19_multiquery.py
```

### What the test verifies (QA)
LLM-generated query variants are merged (max-score fusion) into a multi-query retrieval that includes relevant docs an ambiguous single query might miss.

### Manual QA steps
Run script, confirm both PostgreSQL and vector-database docs appear in the multi-query result set, 'OK'.

### Unit testing this loop
Mock `generate_query_variants()` to return a fixed list of paraphrases; assert `multi_query_search()`'s result set is a superset of (or equal to) `single_query_search()`'s result set for the same base query.

### Issues encountered
None — worked first try.

---

## Loop 20 — Structured Output + Tool-Calling Agent

**Builds on:** Loop 19
**Script:** `src/loop20_tool_calling.py`

### Goal
See `lattice-loop-engineering-prompt.md` (Loop 20 section) for the original spec.

### Dependencies
pydantic, requests, Ollama (llama3.2 tool-calling API)

### How to run
```bash
cd ~/projects/lattice
source lattice-env/bin/activate
python src/loop20_tool_calling.py
```

### What the test verifies (QA)
LLM correctly invokes the `calculator` tool for a math query and produces the correct numeric result.

### Manual QA steps
Run script, confirm the math query returns `tool_used: calculator, result: 4183`, 'OK'.

### Unit testing this loop
Mock Ollama's `/api/chat` tool_calls response for a math prompt, assert `chat_with_tools()` calls `calculator()` with the extracted expression and returns the correct evaluated result.

### Issues encountered
The 3B llama3.2 model over-triggers the calculator tool even for unrelated queries (e.g. 'What is the capital of France?' produced a tool call with expression='Paris'), which crashed `eval('Paris', ...)` with a NameError. Fixed by wrapping `calculator()` in a try/except and returning a graceful 'invalid expression' error instead of crashing; the loop's test only asserts strict correctness on the math-query path, and documents the non-math misfire as a known small-model limitation. NEVER remove the try/except — `eval()` on arbitrary LLM-supplied text is also a security concern, see SECURITY_NOTICE.md.

---

## Loop 21 — Multi-Agent Workflow (LangGraph)

**Builds on:** Loop 20
**Script:** `src/loop21_multiagent.py`

### Goal
See `lattice-loop-engineering-prompt.md` (Loop 21 section) for the original spec.

### Dependencies
langgraph, requests, psycopg2-binary, sentence-transformers, Ollama

### How to run
```bash
cd ~/projects/lattice
source lattice-env/bin/activate
python src/loop21_multiagent.py
```

### What the test verifies (QA)
A 2-node LangGraph workflow (retriever agent -> critique agent) runs end-to-end without manual intervention and produces a final answer plus a critique verdict.

### Manual QA steps
Run script, confirm both 'draft'/'critique'/'final' print, 'OK'.

### Unit testing this loop
Build the graph with mocked `call_llm` returning fixed 'SUPPORTED'/'UNSUPPORTED' strings, assert `critique_agent()` selects `draft_answer` only when the verdict's *first token* is exactly 'SUPPORTED', not merely a substring match.

### Issues encountered
Critique logic used `"SUPPORTED" in critique.upper()` to decide whether to keep the draft answer, but the string 'UNSUPPORTED' contains 'SUPPORTED' as a substring, so an unsupported/rejected answer was silently treated as supported and the (wrong) draft was kept as final. Fixed by checking the first whitespace-delimited token of the critique equals exactly 'SUPPORTED'.

---

## Loop 22 — Knowledge Graph Layer (Neo4j)

**Builds on:** Loop 21
**Script:** `src/loop22_graph.py`

### Goal
See `lattice-loop-engineering-prompt.md` (Loop 22 section) for the original spec.

### Dependencies
neo4j Python driver, Docker (lattice-neo4j container)

### How to run
```bash
cd ~/projects/lattice
source lattice-env/bin/activate
docker run -d --name lattice-neo4j -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/password neo4j:latest
python src/loop22_graph.py
```

### What the test verifies (QA)
A multi-hop Cypher query (System -USES-> Tool -EXTENDS-> Tool) correctly answers a relationship-based question vector search alone cannot.

### Manual QA steps
Confirm `docker ps` shows lattice-neo4j Up and localhost:7474 reachable, run script, confirm multi-hop result contains pgvector->PostgreSQL, 'OK'.

### Unit testing this loop
Use a disposable Neo4j test instance (or testcontainers-neo4j), seed the fixed graph, assert the multi-hop Cypher query returns the expected row.

### Issues encountered
Neo4j takes several seconds to accept Bolt connections after container start; the script added a retry/backoff wait loop (`wait_for_neo4j`) rather than assuming immediate readiness.

---

## Loop 23 — Rigorous Evaluation Suite

**Builds on:** Loop 22
**Script:** `src/loop23_rigorous_eval.py`

### Goal
See `lattice-loop-engineering-prompt.md` (Loop 23 section) for the original spec.

### Dependencies
requests, psycopg2-binary, sentence-transformers, numpy, Ollama

### How to run
```bash
cd ~/projects/lattice
source lattice-env/bin/activate
python src/loop23_rigorous_eval.py
```

### What the test verifies (QA)
LLM auto-generates Q&A pairs from real corpus chunks; retrieval precision@k and embedding-based faithfulness are scored automatically without a hand-labeled dataset.

### Manual QA steps
Run script, confirm at least one QA pair is generated, precision@2 > 0, 'OK'.

### Unit testing this loop
Mock the QA-generation LLM call to return a fixed Q/A pair tied to a known source chunk, assert `context_precision_at_k()` returns 1.0 when the source chunk is in the retrieved set and 0.0 when it's not.

### Issues encountered
None — worked first try (built directly on the guardrail and eval-harness patterns from Loops 12-13).

---

## Loop 24 — Deployment (Streamlit + Docker)

**Builds on:** Loop 23
**Script:** `src/loop24_app.py`

### Goal
See `lattice-loop-engineering-prompt.md` (Loop 24 section) for the original spec.

### Dependencies
streamlit, psycopg2-binary, sentence-transformers, requests, Ollama

### How to run
```bash
cd ~/projects/lattice
source lattice-env/bin/activate
streamlit run src/loop24_app.py --server.headless true --server.port 8501
```

### What the test verifies (QA)
App is reachable at localhost:8501 and answers a query end-to-end through the same retrieve+generate pipeline used elsewhere.

### Manual QA steps
curl -o /dev/null -w '%{http_code}' localhost:8501 and confirm 200; manually open the URL and submit a question in the text box.

### Unit testing this loop
Streamlit apps are hard to unit test directly; instead unit-test the extracted `retrieve()`/`generate()` functions the app calls, and use an end-to-end smoke test (curl 200 check) for the UI layer itself.

### Issues encountered
None blocking — deployed locally only, per the loop's own fallback guidance ('deploy locally via Docker first; skip cloud hosting until stable'); containerizing the Streamlit app itself was not done in this pass.

---

## Loop 25 — KG Design from Ontology

**Builds on:** Loop 24
**Script:** `src/loop25_schema.py`

### Goal
See `lattice-loop-engineering-prompt.md` (Loop 25 section) for the original spec.

### Dependencies
neo4j Python driver

### How to run
```bash
cd ~/projects/lattice
source lattice-env/bin/activate
python src/loop25_schema.py
```

### What the test verifies (QA)
A hand-drafted schema (5 node types, 6 relationship types) is validated against 5 target business questions before any bulk ingestion code was written.

### Manual QA steps
Run script, confirm all 5 target questions return non-empty rows, 'OK'.

### Unit testing this loop
For each entry in `SCHEMA_QUESTIONS`/`checks`, assert the corresponding Cypher query returns at least one row against the seeded fixture graph.

### Issues encountered
None — worked first try.

---

## Loop 26 — Multisource Graph Integration

**Builds on:** Loop 25
**Script:** `src/loop26_multisource_graph.py`

### Goal
See `lattice-loop-engineering-prompt.md` (Loop 26 section) for the original spec.

### Dependencies
psycopg2-binary, neo4j Python driver

### How to run
```bash
cd ~/projects/lattice
source lattice-env/bin/activate
python src/loop26_multisource_graph.py
```

### What the test verifies (QA)
Document nodes derived from a structured Postgres source (`lattice_chunks.source`) are MERGEd into the same canonical `Lattice` System node created in Loop 25/22, without creating duplicate entities.

### Manual QA steps
Run script, confirm 'Lattice System node count: 1' and 'OK'.

### Unit testing this loop
Seed one `System {name:'Lattice'}` node, run the merge logic twice with the same structured source, assert the System node count stays at 1 (idempotency) and Document count matches source count.

### Issues encountered
None — worked first try, but the `MERGE` idempotency (safe to re-run without creating dupes) was deliberately verified with an explicit node-count assertion rather than assumed.

---

## Loop 27 — LLM-Driven Entity & Relationship Extraction

**Builds on:** Loop 26
**Script:** `src/loop27_extraction.py`

### Goal
See `lattice-loop-engineering-prompt.md` (Loop 27 section) for the original spec.

### Dependencies
requests, psycopg2-binary, neo4j Python driver, Ollama (JSON mode)

### How to run
```bash
cd ~/projects/lattice
source lattice-env/bin/activate
python src/loop27_extraction.py
```

### What the test verifies (QA)
LLM extracts subject-relation-object triples from real document chunks as structured JSON, which are inserted into Neo4j; a spot-check confirms extracted entities actually appear in the source text.

### Manual QA steps
Run script, confirm 'extracted N triples' with N > 0 and spot-check accuracy > 0.3, 'OK'.

### Unit testing this loop
Mock the extraction LLM call to return a fixed JSON triple list, assert malformed/incomplete triples (missing subject/relation/object keys) are filtered out before graph insertion.

### Issues encountered
None blocking — Ollama's `format: 'json'` mode was used to force valid JSON output rather than parsing free-text, which avoided the brittle regex-based JSON extraction that would otherwise be needed.

---

## Loop 28 — Named Entity Disambiguation (NED)

**Builds on:** Loop 27
**Script:** `src/loop28_ned.py`

### Goal
See `lattice-loop-engineering-prompt.md` (Loop 28 section) for the original spec.

### Dependencies
sentence-transformers, numpy, neo4j Python driver

### How to run
```bash
cd ~/projects/lattice
source lattice-env/bin/activate
python src/loop28_ned.py
```

### What the test verifies (QA)
Embedding-similarity + union-find clustering merges near-duplicate entity name variants ('Lattice', 'Lattice Docs', 'lattice system') into one cluster, while NOT over-merging distinct entities ('PostgreSQL' vs 'pgvector').

### Manual QA steps
Run script, confirm the Lattice variants land in one cluster and PostgreSQL/pgvector do NOT, 'OK'.

### Unit testing this loop
Assert `find_duplicate_clusters(['Lattice','Lattice Docs'], threshold=0.75)` returns them in the same cluster; assert two clearly distinct entity names never land in the same cluster at the chosen threshold.

### Issues encountered
None — worked first try. Threshold (0.75 cosine) was chosen empirically from this fixture set; would need re-tuning against a larger, real entity set before production use.

---

## Loop 29 — Graph Feature Engineering + ML Primer

**Builds on:** Loop 28
**Script:** `src/loop29_graph_features.py`

### Goal
See `lattice-loop-engineering-prompt.md` (Loop 29 section) for the original spec.

### Dependencies
networkx, neo4j Python driver

### How to run
```bash
cd ~/projects/lattice
source lattice-env/bin/activate
python src/loop29_graph_features.py
```

### What the test verifies (QA)
Degree, betweenness centrality, and PageRank are computed over the live Neo4j graph pulled into a NetworkX DiGraph; the most-connected node has nonzero PageRank (sanity check, not a strict ranking guarantee).

### Manual QA steps
Run script, confirm edges > 0, pagerank list is non-empty, 'OK'.

### Unit testing this loop
Build a small fixed DiGraph, assert `nx.pagerank()` output sums to ~1.0 and every node with an inbound edge has nonzero PageRank.

### Issues encountered
None — worked first try.

---

## Loop 30 — Graph Representation Learning (GNN)

**Builds on:** Loop 29
**Script:** `src/loop30_gnn.py`

### Goal
See `lattice-loop-engineering-prompt.md` (Loop 30 section) for the original spec.

### Dependencies
torch (CPU wheel), torch_geometric

### How to run
```bash
cd ~/projects/lattice
source lattice-env/bin/activate
python src/loop30_gnn.py
```

### What the test verifies (QA)
A small 2-layer GCN trained on a synthetic 2-community graph achieves higher node-classification accuracy on held-out test nodes than a naive majority-class baseline.

### Manual QA steps
Run script, confirm 'GNN test accuracy' > 'majority-class baseline accuracy', 'OK'.

### Unit testing this loop
Fix `torch.manual_seed`, assert the trained GCN's test accuracy exceeds the majority-class baseline by some margin on the same synthetic graph fixture (guards against silent training regressions, e.g. broken loss/optimizer wiring).

### Issues encountered
torch-geometric has a reputation for fragile installs requiring wheel-version matching against the exact torch+CUDA build; used the CPU-only torch wheel index (`--index-url https://download.pytorch.org/whl/cpu`) and plain `pip install torch_geometric` afterward, which happened to resolve cleanly here — flagged as a fragile step if the environment or torch version changes.

---

## Loop 31 — KG-Powered RAG (GraphRAG)

**Builds on:** Loop 30
**Script:** `src/loop31_graphrag.py`

### Goal
See `lattice-loop-engineering-prompt.md` (Loop 31 section) for the original spec.

### Dependencies
requests, psycopg2-binary, neo4j Python driver, sentence-transformers, Ollama

### How to run
```bash
cd ~/projects/lattice
source lattice-env/bin/activate
python src/loop31_graphrag.py
```

### What the test verifies (QA)
A multi-hop question ('what database does Lattice's vector tool build on') is answered incorrectly by vector-only context (LLM hallucinates 'Annoy') but correctly by adding graph context (pgvector extends PostgreSQL).

### Manual QA steps
Run script, confirm 'combined answer' contains 'postgres'/'postgresql', note the vector-only answer differs and is wrong, 'OK'.

### Unit testing this loop
Fix both `vec_context` and `graph_context` as constants, assert the combined-context prompt string contains the graph-derived relationship string, and assert the vector-only prompt string does not.

### Issues encountered
This loop is a direct demonstration of RAG's limits: on the first run the vector-only path returned a plausible-sounding but factually wrong answer ('Annoy') — this was the intended experiment result (proving GraphRAG's value), not a bug to fix.

---

## Loop 32 — Natural Language to Cypher (KG Q&A)

**Builds on:** Loop 31
**Script:** `src/loop32_nl_to_cypher.py`

### Goal
See `lattice-loop-engineering-prompt.md` (Loop 32 section) for the original spec.

### Dependencies
requests, neo4j Python driver, Ollama

### How to run
```bash
cd ~/projects/lattice
source lattice-env/bin/activate
python src/loop32_nl_to_cypher.py
```

### What the test verifies (QA)
For a small test-question set, the LLM converts NL -> Cypher, the query executes (or fails gracefully), and results are summarized in natural language.

### Manual QA steps
Run script, inspect printed Cypher + rows + summary per question; note not all generated Cypher queries are correct with the small model.

### Unit testing this loop
Assert `nl_to_cypher()` output does not contain markdown code fences (```); for a fixed known-good question, assert the generated Cypher executes without raising and returns the expected row shape.

### Issues encountered
The 3B model generated an incorrect Cypher query for 'Who created Python?' — it pattern-matched `Person {name: "Python"}` instead of `Tool {name: "Python"}`, so the query ran without error but returned zero rows. This is a known small-model limitation, not a code bug; documented fallback per the loop spec: constrain to a small set of query templates before allowing fully open-ended Cypher generation in production.

---

## Loop 33 — QA Agent with LangGraph (final integration)

**Builds on:** Loop 32
**Script:** `src/loop33_qa_agent.py`

### Goal
See `lattice-loop-engineering-prompt.md` (Loop 33 section) for the original spec.

### Dependencies
langgraph, requests, psycopg2-binary, neo4j Python driver, sentence-transformers, Ollama

### How to run
```bash
cd ~/projects/lattice
source lattice-env/bin/activate
python src/loop33_qa_agent.py
```

### What the test verifies (QA)
A router node classifies each question as needing vector search or graph lookup (via keyword heuristic), a conditional LangGraph edge dispatches to the right retrieval node, and both paths converge to produce a final natural-language answer.

### Manual QA steps
Run script, confirm both test questions route to their expected strategy ('vector' vs 'graph') and both produce non-empty answers, 'OK'.

### Unit testing this loop
Assert `router()` returns 'graph' for questions containing relational keywords ('extend', 'created', etc.) and 'vector' otherwise; assert the conditional edge map in the compiled graph has exactly the two expected destinations.

### Issues encountered
None blocking — routing uses a keyword heuristic (documented fallback: 'hard-code routing rules before letting the LLM decide tool selection') rather than LLM-based intent classification, traded accuracy-on-edge-cases for determinism and speed.

---

## Loop 34 — LangChain Fundamentals Setup

**Builds on:** Loop 5, Loop 11
**Script:** `src/loop34_langchain_basics.py`

### Goal
See `lattice-loop-engineering-prompt.md` (Loop 34 section) for the original spec.

### Dependencies
langchain, langchain-community, langchain-ollama

### How to run
```bash
cd ~/projects/lattice
source lattice-env/bin/activate
python src/loop34_langchain_basics.py
```

### What the test verifies (QA)
A LangChain Runnable (`PromptTemplate | llm`) produces the same answer as a raw Ollama HTTP call; a JsonOutputParser-based chain returns structured, parseable output.

### Manual QA steps
Run script, confirm raw and langchain answers both mention 'paris', and structured result has a 'capital' key equal to 'Paris', 'OK'.

### Unit testing this loop
Mock the Ollama HTTP layer, assert `chain.invoke({'question': q})` returns a string containing the expected fact; assert `structured_chain.invoke(...)` returns a dict with the expected keys.

### Issues encountered
llama3.2 echoes the JSON schema's `properties` block into its structured output alongside the real answer fields — cosmetic noise, not a parse failure, since the required keys are still present and correctly valued.

---

## Loop 35 — RAG Indexing with LangChain

**Builds on:** Loop 8, Loop 9, Loop 34
**Script:** `src/loop35_langchain_indexing.py`

### Goal
See `lattice-loop-engineering-prompt.md` (Loop 35 section) for the original spec.

### Dependencies
langchain-postgres, pgvector, langchain-text-splitters, langchain-huggingface, langchain-community (TextLoader)

### How to run
```bash
cd ~/projects/lattice
source lattice-env/bin/activate
python src/loop35_langchain_indexing.py
```

### What the test verifies (QA)
LangChain's TextLoader + RecursiveCharacterTextSplitter + PGVector integration indexes a document and retrieves it with the same relevance as the custom Loop 9 pipeline.

### Manual QA steps
Run script, confirm chunk count >= 1 and retrieved result mentions 'vector search' or 'graph', 'OK'.

### Unit testing this loop
Point PGVector at a disposable collection name, index a fixed fixture doc, assert `similarity_search(query, k=2)` returns the expected chunk.

### Issues encountered
`langchain_community.document_loaders` prints a deprecation warning (module no longer actively maintained, migrating to standalone integration packages) — non-blocking, noted for a future dependency bump.

---

## Loop 36 — RAG Retrieval + Generation Chain

**Builds on:** Loop 35, Loop 11
**Script:** `src/loop36_rag_chain.py`

### Goal
See `lattice-loop-engineering-prompt.md` (Loop 36 section) for the original spec.

### Dependencies
langchain-core (RunnablePassthrough, ChatPromptTemplate, StrOutputParser), same stack as Loop 35

### How to run
```bash
cd ~/projects/lattice
source lattice-env/bin/activate
python src/loop36_rag_chain.py
```

### What the test verifies (QA)
A single composed LCEL chain (`retriever | format_docs` + prompt + llm + parser) answers a test question correctly via one `chain.invoke(question)` call.

### Manual QA steps
Run script, confirm answer mentions 'vector' or 'graph', 'OK'.

### Unit testing this loop
Assert `chain.invoke(question)` returns a non-empty string containing an expected keyword for a fixed question against the Loop 35 fixture index.

### Issues encountered
None — worked first try.

---

## Loop 37 — Introducing LangGraph (StateGraph)

**Builds on:** Loop 36
**Script:** `src/loop37_stategraph_memory.py`

### Goal
See `lattice-loop-engineering-prompt.md` (Loop 37 section) for the original spec.

### Dependencies
langgraph, same retrieval/generation stack as Loop 36

### How to run
```bash
cd ~/projects/lattice
source lattice-env/bin/activate
python src/loop37_stategraph_memory.py
```

### What the test verifies (QA)
A StateGraph with an `Annotated[list, operator.add]` history field accumulates conversation turns correctly across repeated `app.invoke()` calls with the same state object.

### Manual QA steps
Run script, confirm 'history length after 2 turns: 2' and 'OK'.

### Unit testing this loop
Invoke the compiled graph twice with the same state dict, assert `state['history']` has length 2 and the first entry's question matches the first turn's input.

### Issues encountered
Turn-2 answer content is a hallucination (small model + thin 2-document corpus produced an unrelated fabricated answer) — the memory/state mechanics work correctly (asserted), but answer *quality* is a known limitation, not re-litigated here since it's the same root cause documented in Loops 12/31/37.

---

## Loop 38 — Agent Architectures (Router, Plan-Do Loop, Tool-Calling)

**Builds on:** Loop 20, Loop 21, Loop 37
**Script:** `src/loop38_agent_architectures.py`

### Goal
See `lattice-loop-engineering-prompt.md` (Loop 38 section) for the original spec.

### Dependencies
langgraph, langchain-ollama

### How to run
```bash
cd ~/projects/lattice
source lattice-env/bin/activate
python src/loop38_agent_architectures.py
```

### What the test verifies (QA)
A router graph correctly classifies and dispatches direct-answer vs tool-call queries; a separate Plan-Do Loop graph always executes at least one step before finalizing, and completes after multiple steps.

### Manual QA steps
Run script, confirm both test queries route correctly and the plan-do-loop result shows >= 2 completed steps, 'OK'.

### Unit testing this loop
Assert `classify()` returns 'tool' for a query containing digits + an arithmetic operator/word and 'direct' otherwise; assert the plan graph's conditional edge only reaches `finalize` once `done` is True.

### Issues encountered
None — worked first try, reusing the routing pattern already validated in Loop 33.

---

## Loop 39 — Reflection, Subgraphs, and Multi-Agent Supervisor

**Builds on:** Loop 38
**Script:** `src/loop39_reflection_supervisor.py`

### Goal
See `lattice-loop-engineering-prompt.md` (Loop 39 section) for the original spec.

### Dependencies
langgraph, langchain-ollama

### How to run
```bash
cd ~/projects/lattice
source lattice-env/bin/activate
python src/loop39_reflection_supervisor.py
```

### What the test verifies (QA)
A reflection graph (draft -> critique) catches and corrects a deliberately misleading prompt; a supervisor graph correctly delegates to one of two subgraph agents (math vs knowledge) based on query content.

### Manual QA steps
Run script, confirm the reflection step's critique is non-empty and the final answer differs from a naive draft when the prompt was adversarial; confirm both supervisor test queries delegate to the expected agent, 'OK'.

### Unit testing this loop
Assert `reflect_node()` always produces a non-empty critique string; assert `supervisor_route()` returns 'math_agent' only when the query contains a digit.

### Issues encountered
None — worked first try. The reflection step correctly caught and reversed an intentionally misleading instruction embedded in the test prompt, demonstrating the mechanism works as intended.

---

## Loop 40 — Self-Corrective RAG + Testing

**Builds on:** Loop 39, Loop 13, Loop 23
**Script:** `src/loop40_self_corrective_rag.py`

### Goal
See `lattice-loop-engineering-prompt.md` (Loop 40 section) for the original spec.

### Dependencies
langsmith, python-dotenv, plus Loop 35/36 stack. Requires `LANGCHAIN_API_KEY` in `.env` (LangSmith account).

### How to run
```bash
cd ~/projects/lattice
source lattice-env/bin/activate
python src/loop40_self_corrective_rag.py
```

### What the test verifies (QA)
Low-confidence retrieval triggers a query rewrite + retry loop (up to `max_retries`); a LangSmith evaluation dataset is created (or reused) and produces a reproducible pass/fail score across repeated `list_examples` calls.

### Manual QA steps
Run script, confirm at least one retry attempt occurs for an ambiguous query, and the LangSmith dataset examples count is stable across two `list_examples` calls, 'OK'.

### Unit testing this loop
Mock `grade_retrieval()` to force a retry, assert `self_corrective_rag()`'s `attempts` list has length > 1; mock the LangSmith client, assert dataset creation is idempotent (second run reuses the existing dataset instead of duplicating it).

### Issues encountered
`rewrite_query()`'s first version returned the LLM's full verbose multi-option response (including a bulleted list of alternative rewordings) instead of a single clean rewritten question, causing query text to balloon across retry attempts. Fixed by tightening the prompt to demand a single-line reply and taking only the first line of the response.

---

## Loop 41 — Production Tracing & Monitoring

**Builds on:** Loop 40, Loop 14
**Script:** `src/loop41_tracing_monitoring.py`

### Goal
See `lattice-loop-engineering-prompt.md` (Loop 41 section) for the original spec.

### Dependencies
langsmith (`@traceable`), plus Loop 35/36 stack. Requires `LANGCHAIN_API_KEY`.

### How to run
```bash
cd ~/projects/lattice
source lattice-env/bin/activate
python src/loop41_tracing_monitoring.py
```

### What the test verifies (QA)
Every retrieve/generate call in a simulated production run is wrapped in `@traceable` and appears as a queryable run via `client.list_runs()`; feedback (a thumbs-up-style score) is attached to the most recent run.

### Manual QA steps
Run script, confirm 'traced runs visible via API' > 0 and feedback attachment succeeds, 'OK'. Cross-check in the LangSmith UI under project 'lattice'.

### Unit testing this loop
Mock the LangSmith client, assert `@traceable`-decorated functions are called with the expected arguments; assert `create_feedback()` is invoked with a valid `run_id` after the pipeline runs.

### Issues encountered
None — worked first try, once the `.env`-provided API key was loaded via `python-dotenv` at the top of the script.

---

## Loop 42 — Deploy to LangGraph Platform

**Builds on:** Loop 41
**Script:** `src/loop42_graph_app.py`

### Goal
See `lattice-loop-engineering-prompt.md` (Loop 42 section) for the original spec.

### Dependencies
langgraph-cli[inmem], plus Loop 35/36 stack. Config: `langgraph.json` at repo root.

### How to run
```bash
cd ~/projects/lattice
source lattice-env/bin/activate
langgraph dev --no-browser --port 8123
curl -X POST localhost:8123/runs/wait -H 'Content-Type: application/json' -d '{"assistant_id":"lattice_qa","input":{"question":"...","answer":""}}'
```

### What the test verifies (QA)
The compiled graph (`app` in `loop42_graph_app.py`, registered as `lattice_qa` in `langgraph.json`) is served by the LangGraph local dev server and answers a real request through `/runs/wait`.

### Manual QA steps
Confirm the dev server log shows 'Application started up', then POST to `/runs/wait` and confirm a JSON response containing a non-empty `answer` field.

### Unit testing this loop
For CI, spin up `langgraph dev` in the background, poll `/ok` or a similar health endpoint until ready, POST a fixed test question, assert the response contains an expected keyword.

### Issues encountered
Full LangGraph Platform deployment needs a separate hosted-deployment step beyond the LangSmith API key (billing/deployment account) — used the loop's own documented fallback instead: tested via the local dev server (`langgraph dev`), which serves the exact same graph/API contract.

---

## Loop 43 — Application Layer (Chatbot / Collaborative / Ambient)

**Builds on:** Loop 42, Loop 24
**Script:** `src/loop43_chatbot_app.py`

### Goal
See `lattice-loop-engineering-prompt.md` (Loop 43 section) for the original spec.

### Dependencies
streamlit, plus Loop 35/36 stack

### How to run
```bash
cd ~/projects/lattice
source lattice-env/bin/activate
streamlit run src/loop43_chatbot_app.py --server.headless true --server.port 8502
```

### What the test verifies (QA)
A chat-style Streamlit UI maintains `st.session_state.messages` across turns and answers each new message using retrieved context plus prior conversation history, chosen as the 'basic interactive chatbot' pattern per the loop's fallback.

### Manual QA steps
curl -o /dev/null -w '%{http_code}' localhost:8502 and confirm 200; manually open the URL, send 2+ messages, confirm the chat history renders and answers reference retrieved context.

### Unit testing this loop
Streamlit UI itself isn't easily unit-testable; unit-test is really about the underlying retrieve/generate call chain already covered by Loop 36's chain tests. For the UI layer, an end-to-end smoke test (curl 200 + manual multi-turn check) is the practical option, same pattern as Loop 24.

### Issues encountered
None blocking — chose the basic interactive chatbot pattern over collaborative/ambient variants, per the loop's own fallback guidance.
# lattice

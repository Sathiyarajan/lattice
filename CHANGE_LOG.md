# CHANGE_LOG.md

All notable changes to this project, in chronological order of the loop-driven build. Format loosely follows Keep a Changelog; dates reflect when the work was actually done in this session (2026-07-10).

## [Loop 0] — 2026-07-10 — Environment Setup
### Added
- WSL Ubuntu project directory `~/projects/lattice`.
- Python venv `lattice-env` with sentence-transformers, faiss-cpu, psycopg2-binary, numpy, pandas.
### Fixed
- `python3.14-venv` apt package missing, blocking venv creation — installed explicitly.

## [Loop 1] — Embeddings
### Added
- `src/loop1_embeddings.py` — sentence-transformers `all-MiniLM-L6-v2` embedding sanity check (dimension consistency, semantic similarity ordering).

## [Loop 2] — Naive Vector Store (SQLite)
### Added
- `src/loop2_sqlite_store.py` — brute-force cosine similarity search over embeddings persisted as JSON in SQLite (`data/vectors.db`).

## [Loop 3] — Real Vector Index (FAISS)
### Added
- `src/loop3_faiss_index.py` — FAISS `IndexFlatIP` index, benchmarked against brute-force baseline.
### Fixed
- Result comparison used indices, which broke on duplicate rows; switched to comparing result text sets.

## [Loop 4] — Production Vector DB (pgvector)
### Added
- `lattice-pg` Docker container (Postgres 16 + pgvector).
- `src/loop4_pgvector.py` — vector distance operator (`<->`) query, cross-checked against Loops 2/3.

## [Loop 5] — RAG Pipeline
### Added
- Ollama installed, `llama3.2` model pulled.
- `src/loop5_rag.py` — first end-to-end retrieve -> generate pipeline with a groundedness assertion.

## [Loop 6] — Metadata Filtering (Hardening)
### Added
- `src/loop6_metadata_filter.py` — category-filtered retrieval on top of the pgvector store.

## [Loop 7] — Real Document Ingestion
### Added
- `pypdf`, `beautifulsoup4` installed.
- `src/loop7_ingest.py` — txt/html/pdf parsers unified into `{source, text}` records.
- Sample fixtures under `data/raw/`.

## [Loop 8] — Chunking Strategy
### Added
- `src/loop8_chunking.py` — sentence-aware chunker with configurable size and overlap.

## [Loop 9] — Production Embedding + Storage Pipeline
### Added
- `src/loop9_pipeline.py` — full ingest -> chunk -> embed -> store pipeline into a new `lattice_chunks` Postgres table.
### Fixed
- pgvector embedding column read back as a string by psycopg2; dimension check via `len()` was wrong, switched to counting commas.

## [Loop 10] — Two-Stage Retrieval (Hybrid + Reranking)
### Added
- `rank-bm25`, cross-encoder reranker installed.
- `src/loop10_hybrid.py` — BM25 + vector score fusion, cross-encoder reranking pass.
### Fixed
- `numpy.ndarray.ptp()` removed in installed NumPy version — switched to `np.ptp()`.

## [Loop 11] — Prompt Assembly + Generation
### Added
- `src/loop11_prompt_gen.py` — numbered-context prompt construction with inline citation enforcement.

## [Loop 12] — Hallucination Guardrails
### Added
- `src/loop12_guardrails.py` — grounding check combining embedding cosine similarity with keyword overlap.
### Fixed
- Cosine-similarity-only grounding check passed a fabricated answer; added keyword-overlap requirement.

## [Loop 13] — Evaluation Harness
### Added
- `ragas`, `deepeval` installed (not used directly due to hosted-API dependency).
- `src/loop13_eval.py` — hand-written, reproducible context-precision@k baseline metric.

## [Loop 14] — Production Hardening (API + Caching)
### Added
- `fastapi`, `uvicorn`, `redis` installed.
- `src/loop14_api.py` — `/query` FastAPI endpoint with `lru_cache`-backed retrieval caching.

## [Loop 15] — Multimodal/Table Extension
### Added
- `src/loop15_table_parsing.py` — CSV/table-to-text conversion (lightest optional extension per loop fallback).

## [Loop 16] — Multi-Format Data Loading
### Added
- `python-docx`, `openpyxl`, `sqlalchemy` installed.
- `src/loop16_multiformat.py` — docx, xlsx, and live Postgres-table loaders, unified with the Loop 7 txt/html loaders.

## [Loop 17] — Preprocessing Upgrades
### Added
- `src/loop17_preprocessing.py` — abbreviation normalization, recursive splitter, lightweight hypothetical-question generator.

## [Loop 18] — Embedding Model Comparison
### Added
- `umap-learn`, `scikit-learn`, `matplotlib` installed.
- `src/loop18_model_compare.py` — intra/inter-cluster similarity comparison across two embedding models; `all-MiniLM-L6-v2` selected with documented rationale.

## [Loop 19] — Hybrid Search + Reranking (Multi-Query)
### Added
- `src/loop19_multiquery.py` — LLM-generated query-variant fusion on top of Loop 10's hybrid search.

## [Loop 20] — Structured Output + Tool-Calling Agent
### Added
- `pydantic` installed.
- `src/loop20_tool_calling.py` — Ollama tool-calling with a `calculator` function.
### Fixed
- Model misfired the tool on a non-math query, crashing `eval()`; added try/except guard. See SECURITY_NOTICE.md for the unresolved `eval()`-on-LLM-output risk.

## [Loop 21] — Multi-Agent Workflow (LangGraph)
### Added
- `langgraph` installed.
- `src/loop21_multiagent.py` — retriever agent + critique agent, 2-node LangGraph workflow.
### Fixed
- Substring-match bug (`"SUPPORTED" in "UNSUPPORTED"`) caused rejected drafts to be kept as final; switched to exact first-token check.

## [Loop 22] — Knowledge Graph Layer (Neo4j)
### Added
- `lattice-neo4j` Docker container.
- `neo4j` Python driver installed.
- `src/loop22_graph.py` — first multi-hop Cypher query, with a connection-readiness retry loop.

## [Loop 23] — Rigorous Evaluation Suite
### Added
- `src/loop23_rigorous_eval.py` — auto-generated Q&A pairs from real corpus chunks, scored for retrieval precision and faithfulness.

## [Loop 24] — Deployment (Streamlit + Docker)
### Added
- `streamlit` installed.
- `src/loop24_app.py` — Streamlit UI wrapping the retrieve+generate pipeline, verified reachable on `localhost:8501`.

## [Loop 25] — KG Design from Ontology
### Added
- `src/loop25_schema.py` — hand-drafted graph schema (5 node types, 6 relationship types), validated against 5 target business questions before ingestion code was written.

## [Loop 26] — Multisource Graph Integration
### Added
- `src/loop26_multisource_graph.py` — structured Postgres source data merged into the Neo4j graph via `MERGE`, resolving to the existing canonical `Lattice` node (no duplicates).

## [Loop 27] — LLM-Driven Entity & Relationship Extraction
### Added
- `src/loop27_extraction.py` — Ollama JSON-mode triple extraction from document chunks, inserted into Neo4j, spot-checked against source text.

## [Loop 28] — Named Entity Disambiguation (NED)
### Added
- `src/loop28_ned.py` — embedding-similarity + union-find clustering to merge duplicate entity name variants without over-merging distinct entities.

## [Loop 29] — Graph Feature Engineering + ML Primer
### Added
- `networkx` installed.
- `src/loop29_graph_features.py` — degree, betweenness centrality, PageRank computed over the live Neo4j graph.

## [Loop 30] — Graph Representation Learning (GNN)
### Added
- `torch` (CPU), `torch_geometric` installed.
- `src/loop30_gnn.py` — 2-layer GCN trained on a synthetic 2-community graph, beats naive baseline on held-out nodes.

## [Loop 31] — KG-Powered RAG (GraphRAG)
### Added
- `src/loop31_graphrag.py` — demonstrates vector-only RAG hallucinating a multi-hop answer, fixed by adding graph context.

## [Loop 32] — Natural Language to Cypher (KG Q&A)
### Added
- `src/loop32_nl_to_cypher.py` — schema-aware NL-to-Cypher generation, execution, natural-language result summarization.
### Known limitation
- Small model produces an incorrect (but syntactically valid) Cypher query for one of two test questions — documented, not patched (production fix is query templates, per loop's own fallback).

## [Loop 33] — QA Agent with LangGraph (final integration)
### Added
- `src/loop33_qa_agent.py` — router node (keyword heuristic) dispatching to vector or graph retrieval inside a LangGraph state machine, converging to a shared answer node. Final integration of all prior loops.

## [Documentation pass] — 2026-07-10
### Added
- `docs/architecture.md` — class diagrams (Postgres schema, Neo4j schema, LangGraph state machines, GNN model) and system data-flow diagram.
- `docs/loops/loop00.md` … `docs/loops/loop33.md` — per-loop README (goal, dependencies, how to run, how to QA, how to unit test, issues encountered).
- `CLAUDE.md` — agent-facing guidance for working in this repo.
- `SECURITY_NOTICE.md` — `eval()`/dynamic-Cypher risk documentation, hardcoded local-dev credentials, no-auth surfaces.
- `ISSUES.md` — consolidated table of every issue hit across all 34 loops with root cause and fix.
- `CHANGE_LOG.md` — this file.

### Changed (merge pass)
- Merged all 34 per-loop `docs/loops/loopNN.md` files into top-level `README.md` (single indexed document with TOC anchors); removed `docs/loops/` directory.

## [Loop 34] — LangChain Fundamentals Setup
### Added
- `langchain`, `langchain-community`, `langchain-ollama` installed.
- `src/loop34_langchain_basics.py` — LangChain Runnable (`PromptTemplate | llm`) matched against a raw Ollama call; JSON-structured output via `JsonOutputParser`.

## [Loop 35] — RAG Indexing with LangChain
### Added
- `langchain-postgres`, `pgvector`, `langchain-text-splitters`, `langchain-huggingface` installed.
- `src/loop35_langchain_indexing.py` — LangChain `TextLoader` + `RecursiveCharacterTextSplitter` + `PGVector` integration, drop-in replacement for the custom Loop 9 pipeline.

## [Loop 36] — RAG Retrieval + Generation Chain
### Added
- `src/loop36_rag_chain.py` — single LCEL chain (`retriever | format_docs` + prompt + llm + parser), one-line `chain.invoke(question)`.

## [Loop 37] — Introducing LangGraph (StateGraph)
### Added
- `src/loop37_stategraph_memory.py` — StateGraph with `Annotated[list, operator.add]` conversational memory, verified to accumulate history across turns.

## [Loop 38] — Agent Architectures (Router, Plan-Do Loop, Tool-Calling)
### Added
- `src/loop38_agent_architectures.py` — Router graph (direct-answer vs tool-call dispatch) and a Plan-Do Loop graph (always executes >=1 step before finalizing).

## [Loop 39] — Reflection, Subgraphs, and Multi-Agent Supervisor
### Added
- `src/loop39_reflection_supervisor.py` — reflection graph (draft -> critique -> correction) and a supervisor graph delegating to math/knowledge subgraph agents.

## [Loop 40] — Self-Corrective RAG + Testing
### Added
- `langsmith`, `python-dotenv` installed. `.env` created (gitignored) with `LANGCHAIN_API_KEY` for LangSmith.
- `src/loop40_self_corrective_rag.py` — retry/rewrite loop on low-confidence retrieval; LangSmith evaluation dataset created and scored reproducibly.
### Fixed
- `rewrite_query()` returned the LLM's full verbose multi-option response instead of one clean question, bloating retry queries — tightened prompt to force a single-line reply.

## [Loop 41] — Production Tracing & Monitoring
### Added
- `src/loop41_tracing_monitoring.py` — `@traceable`-wrapped retrieve/generate pipeline; traces verified queryable via `client.list_runs()`; feedback attached to a run.

## [Loop 42] — Deploy to LangGraph Platform
### Added
- `langgraph-cli[inmem]` installed.
- `langgraph.json` — root-level LangGraph app config, registers `lattice_qa` graph.
- `src/loop42_graph_app.py` — compiled graph served via `langgraph dev` local dev server, verified reachable through `/runs/wait`.
### Known limitation
- Full LangGraph Platform (hosted) deployment needs a separate billing/deployment account beyond the LangSmith API key — used the loop's own fallback (local dev server) instead.

## [Loop 43] — Application Layer (Chatbot / Collaborative / Ambient)
### Added
- `src/loop43_chatbot_app.py` — Streamlit chat UI (`st.chat_message`/`st.chat_input`) with session-state conversation history and RAG-backed answers. Final loop of the full 44-loop build (Loop 0-43).

## [Documentation pass 2] — 2026-07-11
### Added
- Loop 34-43 sections appended to `README.md` (TOC + full per-loop docs).
### Changed
- This file, `ISSUES.md`, `SECURITY_NOTICE.md`, `CLAUDE.md` updated to cover Loops 34-43.

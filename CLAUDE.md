# CLAUDE.md

Guidance for Claude Code (or any AI agent) working in this repository.

## What this repo is

Lattice: an incremental RAG/GraphRAG learning project built loop-by-loop per `lattice-loop-engineering-prompt.md`. 44 loops (0-43), each a standalone, runnable script in `src/` that proves one capability and composes on top of earlier loops. Loops 0-33 are the "from scratch" build (custom embedding/retrieval/graph code); loops 34-43 rebuild and extend that stack on top of LangChain/LangGraph/LangSmith. This is not a packaged application — there is no single `main.py` entrypoint; each `src/loopNN_*.py` is independently runnable and independently testable. The loop spec file may grow further — always re-read `lattice-loop-engineering-prompt.md` in full before assuming 43 is the last loop.

## Environment

- WSL Ubuntu, project root `~/projects/lattice`.
- Python venv at `~/projects/lattice/lattice-env` — always `source lattice-env/bin/activate` before running anything.
- External services run in Docker:
  - `lattice-pg` — Postgres + pgvector, port 5432, user `postgres` / password `pass`.
  - `lattice-neo4j` — Neo4j, ports 7474 (browser) / 7687 (bolt), auth `neo4j` / `password`.
  - `ollama serve` — local LLM server, port 11434, model `llama3.2` (pulled via `ollama pull llama3.2`).
- Check `docker ps` before assuming these are up; commands that depend on them will connection-refuse otherwise.
- Any `sudo` step (apt installs, `docker service start`) needs an interactive password prompt in this environment — cannot be run non-interactively. Ask the user to run it themselves in a real WSL terminal rather than silently retrying.
- `.env` at repo root holds `LANGCHAIN_API_KEY` (LangSmith, used by Loops 40-42) — chmod 600, gitignored. Load it with `python-dotenv` (see `src/loop40_self_corrective_rag.py` for the pattern) rather than hardcoding secrets. Never print or log its contents. If a loop needs a hosted-service credential that isn't already in `.env`, ask the user for it rather than trying to work around the requirement.
- `langgraph.json` at repo root configures the LangGraph local dev server (`langgraph dev`, Loop 42) — points at `src/loop42_graph_app.py:app`.

## Layout

```
lattice/
  lattice-loop-engineering-prompt.md   # source spec for all 34 loops
  lattice-env/                          # python venv (gitignored if this becomes a git repo)
  data/
    raw/                                # sample ingestion fixtures (txt, html, docx, xlsx)
    vectors.db                          # SQLite brute-force store (Loop 2)
  src/
    loopNN_<name>.py                    # one script per loop, each independently runnable
  docs/
    architecture.md                     # class diagrams, data flow, component overview
    loops/loopNN.md                     # per-loop README: how to run, how to QA, how to unit test, issues
  CLAUDE.md                             # this file
  CHANGE_LOG.md
  SECURITY_NOTICE.md
  ISSUES.md
```

## Conventions used across loop scripts

- Each script is self-contained: run with `python src/loopNN_name.py`, prints intermediate state, ends with an `assert` block and prints `OK` on success. No test framework (pytest) wired up yet — the assertions ARE the test. See `docs/loops/loopNN.md` for a suggested pytest-style unit test if formalizing this later.
- Scripts that need Postgres/Neo4j/Ollama assume those services are already running (see Environment above) — they do not start/wait for them except where explicitly noted (e.g. Loop 22's `wait_for_neo4j` retry loop).
- Embedding model is `all-MiniLM-L6-v2` (384-dim) everywhere unless a loop is explicitly comparing models (Loop 18).
- LLM generation model is `llama3.2` (3B) via Ollama's HTTP API (`/api/generate`, `/api/chat`) — small model, occasionally misfires on tool-calling and NL-to-Cypher edge cases; this is documented per-loop rather than treated as a bug to chase.

## When extending this repo

- New loops: follow the existing naming (`loopNN_shortname.py`), add a corresponding `docs/loops/loopNN.md` using the same template as existing entries (Goal / Dependencies / How to run / What the test verifies / Manual QA / Unit testing / Issues encountered).
- Don't silently "fix" a small-model misfire (tool-calling, Cypher generation) by hardcoding around it — document it as a known limitation instead, per the pattern in Loops 20 and 32.
- If you touch `eval()`, `exec()`, or dynamic Cypher/SQL construction, read `SECURITY_NOTICE.md` first — several loops intentionally use these for the demo and the notice explains the guardrails already in place and what's still missing for production use.

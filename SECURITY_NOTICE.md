# SECURITY_NOTICE.md

This is a local, single-user research/learning build (34-loop RAG/GraphRAG curriculum). It was never intended to be exposed beyond `localhost`. If any part of this is reused in a shared or production context, address the items below first.

## 1. `eval()` on LLM-supplied input — Loop 20 (`src/loop20_tool_calling.py`)

The `calculator` tool executes `eval(expression, {"__builtins__": {}}, {})` where `expression` comes directly from the LLM's tool-call arguments, which in turn can be influenced by user input reaching the LLM.

- Mitigation currently in place: `__builtins__` stripped from eval's globals, wrapped in try/except so a bad expression fails safe (returns an error) instead of crashing.
- Still exploitable: a sufficiently crafted expression can escape a builtins-stripped `eval` via object introspection (e.g. reaching `__class__.__bases__` chains). **Do not treat the current try/except as a security boundary.**
- Before production use: replace `eval` with a restricted arithmetic parser (e.g. `ast.literal_eval` plus a whitelist of operators, or a dedicated expression-evaluation library) that cannot reach arbitrary Python objects.

## 2. Dynamic Cypher construction from LLM output — Loops 27, 32 (`src/loop27_extraction.py`, `src/loop32_nl_to_cypher.py`)

- Loop 27 builds relationship type names by sanitizing LLM output with a regex (`re.sub(r"[^A-Z_]", "_", ...)`) before interpolating into a Cypher `MERGE` statement, since Cypher doesn't support parameterized relationship *types*. The sanitizer is a basic allowlist (uppercase letters and underscore only) — reviewed as adequate for this demo's threat model (local Ollama output, not adversarial), but not hardened against a determined attacker who controls the LLM's output.
- Loop 32 does full NL-to-Cypher generation and executes the LLM's raw Cypher query text directly (wrapped in try/except for malformed queries, but no query validation or sandboxing). This is a direct Cypher-injection surface if the "natural language question" input ever comes from an untrusted user rather than a developer running local test cases.
- Before production use: constrain to parameterized query templates (the loop's own documented fallback) rather than open-ended LLM-generated Cypher, or run generated queries against a read-only Neo4j role with no write/admin privileges.

## 3. Hardcoded credentials

- Postgres: `postgres` / `pass` (Loops 4, 6, 9-21, 23, 26, 31, 33).
- Neo4j: `neo4j` / `password` (Loops 22, 25-33).

These are fine for local Docker containers only reachable from `localhost` in a dev environment. Never reuse these credentials, and never expose ports 5432/7474/7687 beyond localhost/a trusted private network without changing them and adding real secret management (env vars, a secrets manager) instead of the literal strings in source.

## 4. No authentication/authorization anywhere

The FastAPI service (Loop 14, `src/loop14_api.py`) and the Streamlit app (Loop 24, `src/loop24_app.py`) have zero auth. Both are fine bound to localhost for local testing; neither should be exposed on a public interface or shared network as-is.

## 5. Ollama server exposure

`ollama serve` was run bound to default (localhost:11434) with no auth. If this is ever changed to bind on `0.0.0.0`, anyone on the network could submit prompts to the local LLM and consume compute/cause cost — do not change the bind address without adding a reverse proxy with auth in front of it.

## 6. Data handled

All content processed across the 44 loops is synthetic/sample text (Eiffel Tower facts, product CSVs, etc.) — no real PII or secrets were ingested. If real documents are loaded into this pipeline later, revisit chunk storage (Postgres, Neo4j, SQLite) for at-rest encryption and access control before storing anything sensitive.

## 7. LangSmith API key (`.env`) — Loops 40, 41, 42

- A real `LANGCHAIN_API_KEY` (LangSmith) was supplied interactively by the user and stored in `~/projects/lattice/.env` with `chmod 600`, and `.env` was added to `.gitignore`.
- This key is loaded via `python-dotenv` at the top of `src/loop40_self_corrective_rag.py`, `src/loop41_tracing_monitoring.py`, and implicitly by `langgraph dev` (Loop 42) via the `"env": ".env"` entry in `langgraph.json`.
- **Before committing this repo to any shared/public location:** confirm `.env` is genuinely excluded (`git status` should never show it), and rotate the key if it was ever pasted into a chat log, ticket, or shared terminal history that isn't access-controlled the same way as this repo.
- With `LANGCHAIN_TRACING_V2=true`, every traced call's prompts/responses/context are sent to LangSmith's hosted service. Since Loops 40-43 only ever processed synthetic sample data (see §6), this was acceptable here — if real/sensitive documents are ever indexed into this pipeline, revisit whether tracing those queries to a third-party hosted service is acceptable before re-enabling `LANGCHAIN_TRACING_V2`.

## 8. LangGraph local dev server (Loop 42)

`langgraph dev` was run bound to `localhost:8123` with no auth, same posture as the Ollama/FastAPI/Streamlit services in §4-5 — fine for local testing only, never expose beyond localhost as-is.

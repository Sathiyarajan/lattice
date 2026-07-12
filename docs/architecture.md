# Lattice — Architecture

System built incrementally across 34 loops (0-33). Not a single deployed app — a sequence of standalone scripts in `src/`, each proving one capability, later loops composing earlier ones. This doc gives class/component diagrams for the pieces that have real structure (Postgres schema, Neo4j schema, the LangGraph agent state machines, the GNN model).

## 1. Storage layer — class diagram

```mermaid
classDiagram
    class LatticeChunksTable {
        +int id
        +text source
        +int chunk_id
        +text text
        +vector(384) embedding
    }
    class MetaItemsTable {
        +int id
        +text text
        +text category
        +vector(384) embedding
    }
    class RagItemsTable {
        +int id
        +text text
        +vector(384) embedding
    }
    class SQLiteItemsTable {
        +int id
        +text text
        +text embedding_json
    }
    note for LatticeChunksTable "Postgres + pgvector\nproduction chunk store (Loop 9+)"
    note for SQLiteItemsTable "Loop 2 brute-force store\n(embedding stored as JSON text)"
```

## 2. Neo4j graph schema — class diagram

```mermaid
classDiagram
    class System {
        +string name
    }
    class Tool {
        +string name
        +string category
    }
    class Document {
        +string title
        +string source
        +string source_system
    }
    class Concept {
        +string name
    }
    class Person {
        +string name
    }
    class ExtractedEntity {
        +string name
    }
    System "1" --> "*" Tool : USES
    Tool "1" --> "*" Tool : EXTENDS
    Document "1" --> "*" Concept : MENTIONS
    Document "1" --> "*" Tool : MENTIONS
    Document "1" --> "1" System : DESCRIBES
    Concept "1" --> "*" Concept : RELATED_TO
    Person "1" --> "*" Tool : CREATED
    ExtractedEntity "*" --> "*" ExtractedEntity : (dynamic relation, Loop 27)
```

## 3. RAG retrieval pipeline — component/flow diagram

```mermaid
flowchart LR
    Q[User Question] --> R{Router\nLoop 33}
    R -->|keyword-heuristic:\nfactual| VEC[Vector Retrieve\npgvector Loop 4/9]
    R -->|keyword-heuristic:\nrelational| GRAPH[Graph Lookup\nNeo4j Loop 22/31]
    VEC --> HYB[Hybrid + Rerank\nBM25 + cross-encoder\nLoop 10/19]
    HYB --> PROMPT[Prompt Assembly\n+ citations Loop 11]
    GRAPH --> PROMPT
    PROMPT --> LLM[Ollama llama3.2\nGenerate]
    LLM --> GUARD{Grounding Guardrail\ncosine + keyword overlap\nLoop 12}
    GUARD -->|pass| ANSWER[Final Answer]
    GUARD -->|fail| FLAG[Flagged / Rejected]
```

## 4. Multi-agent state machines (LangGraph) — class diagram

```mermaid
classDiagram
    class AgentState_Loop21 {
        +string query
        +list context
        +string draft_answer
        +string final_answer
        +string critique
    }
    class QAState_Loop33 {
        +string question
        +string route
        +list context
        +string answer
    }
    class RetrieverAgent {
        +retriever_agent(state) AgentState_Loop21
    }
    class CritiqueAgent {
        +critique_agent(state) AgentState_Loop21
    }
    class RouterNode {
        +router(state) QAState_Loop33
    }
    class VectorNode {
        +vector_node(state) QAState_Loop33
    }
    class GraphNode {
        +graph_node(state) QAState_Loop33
    }
    class AnswerNode {
        +answer_node(state) QAState_Loop33
    }
    RetrieverAgent --> AgentState_Loop21
    CritiqueAgent --> AgentState_Loop21
    RouterNode --> QAState_Loop33
    VectorNode --> QAState_Loop33
    GraphNode --> QAState_Loop33
    AnswerNode --> QAState_Loop33
```

## 5. GNN model — class diagram (Loop 30)

```mermaid
classDiagram
    class GCN {
        -GCNConv conv1
        -GCNConv conv2
        +forward(x, edge_index) Tensor
    }
    class GCNConv {
        <<torch_geometric.nn>>
    }
    GCN --> GCNConv : conv1 (num_nodes -> 8)
    GCN --> GCNConv : conv2 (8 -> 2)
```

## Data stores in use

| Store | Purpose | Loops |
|---|---|---|
| SQLite (`data/vectors.db`) | brute-force baseline vector search | 2 |
| FAISS (in-memory index) | fast ANN baseline | 3 |
| Postgres + pgvector (Docker `lattice-pg`, port 5432) | production vector store | 4, 6, 9-21, 23, 26, 31, 33 |
| Neo4j (Docker `lattice-neo4j`, ports 7474/7687) | knowledge graph | 22, 25-33 |
| Ollama (localhost:11434, model `llama3.2`) | local LLM for generation, extraction, agents | 5, 11-14, 19-21, 23, 24, 27, 31-33 |

## Known architectural limitations (see also ISSUES.md, SECURITY_NOTICE.md)

- Router (Loop 33) and hallucination guardrail (Loop 12) both use lightweight heuristics (keyword match / keyword-overlap+cosine) rather than a dedicated classifier — good enough for a demo corpus, not tuned for production scale.
- NL-to-Cypher (Loop 32) is unconstrained free-generation with a 3B model; accuracy is not production-grade — see Loop 32 doc.
- No auth/access-control layer anywhere — this is a local, single-user research build, not a deployed service.

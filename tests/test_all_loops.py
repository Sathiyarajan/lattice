"""
Auto-generated integration test suite — one test per Lattice loop (0-43).

Each test executes the loop's script in-process via runpy (so pytest-cov can
instrument real line coverage) against the LIVE local stack: Postgres+pgvector
(lattice-pg), Neo4j (lattice-neo4j), and Ollama (localhost:11434). These are
integration tests, not isolated unit tests with mocks -- the loop scripts
themselves are thin procedural scripts whose entire point is to prove
end-to-end behavior against real services, so testing them against mocks
would test the mocks, not the system. Each script's own internal `assert`
statements ARE the test assertions; pytest fails the test if the script
raises (AssertionError or otherwise), and passes if the script runs to
completion (every script ends by printing "OK").

Requires: docker ps shows lattice-pg and lattice-neo4j Up; `ollama serve`
running with the llama3.2 model pulled. See CLAUDE.md.
"""
import os
import runpy
import pytest

SRC_DIR = os.path.expanduser("~/projects/lattice/src")


def run_loop_script(filename):
    """Execute a loop script as if run via `python src/<filename>`."""
    path = os.path.join(SRC_DIR, filename)
    runpy.run_path(path, run_name="__main__")


def test_loop00_environment_imports():
    """Loop 0 has no script -- test is the same import check used to validate the venv."""
    import faiss
    import sentence_transformers
    import psycopg2
    import numpy
    import pandas
    assert faiss and sentence_transformers and psycopg2 and numpy and pandas


def test_loop01_embeddings():
    """Embeddings"""
    run_loop_script("loop1_embeddings.py")

def test_loop02_sqlite_store():
    """Naive Vector Store (SQLite)"""
    run_loop_script("loop2_sqlite_store.py")

def test_loop03_faiss_index():
    """Real Vector Index (FAISS)"""
    run_loop_script("loop3_faiss_index.py")

def test_loop04_pgvector():
    """Production Vector DB (pgvector)"""
    run_loop_script("loop4_pgvector.py")

def test_loop05_rag():
    """RAG Pipeline"""
    run_loop_script("loop5_rag.py")

def test_loop06_metadata_filter():
    """Metadata Filtering"""
    run_loop_script("loop6_metadata_filter.py")

def test_loop07_ingest():
    """Real Document Ingestion"""
    run_loop_script("loop7_ingest.py")

def test_loop08_chunking():
    """Chunking Strategy"""
    run_loop_script("loop8_chunking.py")

def test_loop09_pipeline():
    """Production Embedding + Storage Pipeline"""
    run_loop_script("loop9_pipeline.py")

def test_loop10_hybrid():
    """Two-Stage Retrieval (Hybrid + Reranking)"""
    run_loop_script("loop10_hybrid.py")

def test_loop11_prompt_gen():
    """Prompt Assembly + Generation"""
    run_loop_script("loop11_prompt_gen.py")

def test_loop12_guardrails():
    """Hallucination Guardrails"""
    run_loop_script("loop12_guardrails.py")

def test_loop13_eval():
    """Evaluation Harness"""
    run_loop_script("loop13_eval.py")

def test_loop14_api():
    """Production Hardening (API + caching)"""
    run_loop_script("loop14_api.py")

def test_loop15_table_parsing():
    """Multimodal/Table Extension"""
    run_loop_script("loop15_table_parsing.py")

def test_loop16_multiformat():
    """Multi-Format Data Loading"""
    run_loop_script("loop16_multiformat.py")

def test_loop17_preprocessing():
    """Preprocessing Upgrades"""
    run_loop_script("loop17_preprocessing.py")

def test_loop18_model_compare():
    """Embedding Model Comparison"""
    run_loop_script("loop18_model_compare.py")

def test_loop19_multiquery():
    """Hybrid Search + Reranking (Multi-Query)"""
    run_loop_script("loop19_multiquery.py")

def test_loop20_tool_calling():
    """Structured Output + Tool-Calling Agent"""
    run_loop_script("loop20_tool_calling.py")

def test_loop21_multiagent():
    """Multi-Agent Workflow (LangGraph)"""
    run_loop_script("loop21_multiagent.py")

def test_loop22_graph():
    """Knowledge Graph Layer (Neo4j)"""
    run_loop_script("loop22_graph.py")

def test_loop23_rigorous_eval():
    """Rigorous Evaluation Suite"""
    run_loop_script("loop23_rigorous_eval.py")

def test_loop24_app():
    """Deployment (Streamlit)"""
    run_loop_script("loop24_app.py")

def test_loop25_schema():
    """KG Design from Ontology"""
    run_loop_script("loop25_schema.py")

def test_loop26_multisource_graph():
    """Multisource Graph Integration"""
    run_loop_script("loop26_multisource_graph.py")

def test_loop27_extraction():
    """LLM-Driven Entity & Relationship Extraction"""
    run_loop_script("loop27_extraction.py")

def test_loop28_ned():
    """Named Entity Disambiguation (NED)"""
    run_loop_script("loop28_ned.py")

def test_loop29_graph_features():
    """Graph Feature Engineering + ML Primer"""
    run_loop_script("loop29_graph_features.py")

def test_loop30_gnn():
    """Graph Representation Learning (GNN)"""
    run_loop_script("loop30_gnn.py")

def test_loop31_graphrag():
    """KG-Powered RAG (GraphRAG)"""
    run_loop_script("loop31_graphrag.py")

def test_loop32_nl_to_cypher():
    """Natural Language to Cypher (KG Q&A)"""
    run_loop_script("loop32_nl_to_cypher.py")

def test_loop33_qa_agent():
    """QA Agent with LangGraph (final integration)"""
    run_loop_script("loop33_qa_agent.py")

def test_loop34_langchain_basics():
    """LangChain Fundamentals Setup"""
    run_loop_script("loop34_langchain_basics.py")

def test_loop35_langchain_indexing():
    """RAG Indexing with LangChain"""
    run_loop_script("loop35_langchain_indexing.py")

def test_loop36_rag_chain():
    """RAG Retrieval + Generation Chain"""
    run_loop_script("loop36_rag_chain.py")

def test_loop37_stategraph_memory():
    """Introducing LangGraph (StateGraph)"""
    run_loop_script("loop37_stategraph_memory.py")

def test_loop38_agent_architectures():
    """Agent Architectures"""
    run_loop_script("loop38_agent_architectures.py")

def test_loop39_reflection_supervisor():
    """Reflection, Subgraphs, and Multi-Agent Supervisor"""
    run_loop_script("loop39_reflection_supervisor.py")

def test_loop40_self_corrective_rag():
    """Self-Corrective RAG + Testing"""
    run_loop_script("loop40_self_corrective_rag.py")

def test_loop41_tracing_monitoring():
    """Production Tracing & Monitoring"""
    run_loop_script("loop41_tracing_monitoring.py")

def test_loop42_graph_app():
    """Deploy to LangGraph Platform"""
    run_loop_script("loop42_graph_app.py")

def test_loop43_chatbot_app():
    """Application Layer (Chatbot)"""
    run_loop_script("loop43_chatbot_app.py")

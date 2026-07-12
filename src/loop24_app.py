import streamlit as st
import psycopg2
import requests
from sentence_transformers import SentenceTransformer

@st.cache_resource
def get_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource
def get_conn():
    return psycopg2.connect(host="localhost", port=5432, user="postgres", password="pass", dbname="postgres")

model = get_model()
conn = get_conn()

def retrieve(query, k=2):
    qvec = model.encode([query])[0]
    cur = conn.cursor()
    cur.execute("SELECT text FROM lattice_chunks ORDER BY embedding <-> %s::vector LIMIT %s", (qvec.tolist(), k))
    result = [r[0] for r in cur.fetchall()]
    cur.close()
    return result

def generate(query, context):
    prompt = f"Answer using only this context:\n{chr(10).join(context)}\n\nQuestion: {query}\nAnswer:"
    resp = requests.post("http://localhost:11434/api/generate", json={"model": "llama3.2", "prompt": prompt, "stream": False}, timeout=60)
    return resp.json()["response"].strip()

st.title("Lattice RAG")
query = st.text_input("Ask a question about Lattice")
if query:
    context = retrieve(query)
    answer = generate(query, context)
    st.write("**Answer:**", answer)
    st.write("**Sources:**", context)

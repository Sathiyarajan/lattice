import streamlit as st
from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_postgres import PGVector

CONNECTION = "postgresql+psycopg://postgres:pass@localhost:5432/postgres"
COLLECTION = "lattice_langchain"

@st.cache_resource
def get_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return PGVector(embeddings=embeddings, collection_name=COLLECTION, connection=CONNECTION, use_jsonb=True)

@st.cache_resource
def get_llm():
    return OllamaLLM(model="llama3.2", base_url="http://localhost:11434")

vectorstore = get_vectorstore()
llm = get_llm()

st.title("Lattice Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

user_input = st.chat_input("Ask Lattice a question")
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    docs = vectorstore.similarity_search(user_input, k=2)
    context = "\n".join(d.page_content for d in docs)
    history = "\n".join(f"{m['role']}: {m['content']}" for m in st.session_state.messages[:-1])
    prompt = f"Conversation history:\n{history}\n\nContext:\n{context}\n\nUser: {user_input}\nAssistant:"
    answer = llm.invoke(prompt)

    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.write(answer)

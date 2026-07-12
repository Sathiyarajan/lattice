import os
from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_postgres import PGVector
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

CONNECTION = "postgresql+psycopg://postgres:pass@localhost:5432/postgres"
COLLECTION = "lattice_langchain"

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = PGVector(
    embeddings=embeddings,
    collection_name=COLLECTION,
    connection=CONNECTION,
    use_jsonb=True,
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
llm = OllamaLLM(model="llama3.2", base_url="http://localhost:11434")

prompt = ChatPromptTemplate.from_template(
    "Answer using only this context:\n{context}\n\nQuestion: {question}\nAnswer:"
)

def format_docs(docs):
    return "\n".join(d.page_content for d in docs)

chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

question = "What does Lattice combine for retrieval?"
answer = chain.invoke(question)
print("chain answer:", answer)

assert "vector" in answer.lower() or "graph" in answer.lower(), "one-line chain.invoke did not produce a grounded answer"
print("OK")

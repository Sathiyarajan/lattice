import requests
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

llm = OllamaLLM(model="llama3.2", base_url="http://localhost:11434")

# --- raw Ollama call (Loop 5 style, for comparison) ---
def raw_call(prompt_text):
    resp = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": "llama3.2", "prompt": prompt_text, "stream": False},
        timeout=60,
    )
    return resp.json()["response"].strip()

question = "What is the capital of France?"
raw_answer = raw_call(question)

# --- LangChain Runnable: PromptTemplate | LLM ---
prompt = PromptTemplate.from_template("{question}")
chain = prompt | llm
lc_answer = chain.invoke({"question": question})

print("raw:", raw_answer)
print("langchain:", lc_answer)

assert "paris" in raw_answer.lower()
assert "paris" in lc_answer.lower()

# --- structured JSON output via output parser ---
class CapitalAnswer(BaseModel):
    country: str = Field(description="the country asked about")
    capital: str = Field(description="the capital city")

parser = JsonOutputParser(pydantic_object=CapitalAnswer)
structured_prompt = PromptTemplate(
    template="Answer the question.\n{format_instructions}\nQuestion: {question}\n",
    input_variables=["question"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)
structured_chain = structured_prompt | llm | parser
structured_result = structured_chain.invoke({"question": question})

print("structured:", structured_result)
assert structured_result["capital"].lower() == "paris"
print("OK")

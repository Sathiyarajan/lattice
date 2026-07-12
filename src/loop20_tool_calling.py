import json
import requests
from pydantic import BaseModel

class CalculatorArgs(BaseModel):
    expression: str

def calculator(expression: str) -> str:
    return str(eval(expression, {"__builtins__": {}}, {}))

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate a basic arithmetic expression",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
            },
        },
    }
]

def chat_with_tools(user_message):
    resp = requests.post(
        "http://localhost:11434/api/chat",
        json={
            "model": "llama3.2",
            "messages": [{"role": "user", "content": user_message}],
            "tools": TOOLS,
            "stream": False,
        },
        timeout=60,
    )
    data = resp.json()
    message = data["message"]
    tool_calls = message.get("tool_calls", [])
    if tool_calls:
        call = tool_calls[0]
        fn_name = call["function"]["name"]
        args = call["function"]["arguments"]
        if isinstance(args, str):
            args = json.loads(args)
        if fn_name == "calculator":
            try:
                result = calculator(args["expression"])
            except Exception:
                return {"tool_used": fn_name, "args": args, "result": None, "error": "invalid expression"}
            return {"tool_used": fn_name, "args": args, "result": result}
    return {"tool_used": None, "content": message.get("content")}

r1 = chat_with_tools("What is 47 times 89?")
print("math query:", r1)

r2 = chat_with_tools("What is the capital of France?")
print("non-math query:", r2)

assert r1["tool_used"] == "calculator", f"expected calculator tool call, got {r1}"
assert r1["result"] == "4183"
print("OK — tool invocation reliable for math queries")

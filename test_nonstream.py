"""Test non-streaming json_schema response fencing."""
import httpx, json

body = {
    "model": "deepseek-v4-pro",
    "messages": [{"role": "user", "content": "Return JSON with key msg: \"hello\""}],
    "response_format": {
        "type": "json_schema",
        "json_schema": {"name": "resp", "strict": True, "schema": {
            "type": "object",
            "properties": {"msg": {"type": "string"}},
            "required": ["msg"],
            "additionalProperties": False
        }}
    }
}

r = httpx.post("http://127.0.0.1:4003/v1/chat/completions",
    json=body, headers={"Authorization": "Bearer sk-proxy"},
    timeout=60)

data = r.json()
content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
print(f"Starts with backtick: {content.startswith('`')}")
print(f"Starts with brace: {content.startswith('{')}")
print(f"Content: {repr(content[:300])}")

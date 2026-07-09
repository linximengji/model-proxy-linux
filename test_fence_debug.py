"""Debug streaming fence stripping - show raw SSE lines."""
import httpx
import json

body = {
    "model": "deepseek-v4-pro",
    "messages": [
        {"role": "system", "content": "Generate presentation outline with slides array."},
        {"role": "user", "content": "Compare 4 AI assistants. 6 slides."}
    ],
    "stream": True,
    "response_format": {
        "type": "json_schema",
        "json_schema": {
            "name": "response",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "slides": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {"content": {"type": "string"}},
                            "required": ["content"],
                            "additionalProperties": False
                        }
                    }
                },
                "required": ["slides"],
                "additionalProperties": False
            }
        }
    }
}

r = httpx.post("http://127.0.0.1:4003/v1/chat/completions",
    json=body, headers={"Authorization": "Bearer sk-proxy"},
    timeout=120)

# Show raw SSE lines that contain content
lines = r.text.strip().split("\n")
content_count = 0
for i, ln in enumerate(lines):
    if ln.startswith("data: ") and "content" in ln and ln != "data: [DONE]":
        try:
            d = json.loads(ln[6:])
            delta = d.get("choices", [{}])[0].get("delta", {})
            if "content" in delta:
                content_count += 1
                c = delta["content"]
                if content_count <= 10 or content_count % 100 == 0:
                    print(f"  chunk #{content_count}: {repr(c[:120])}")
        except Exception:
            pass

print(f"\nTotal content chunks: {content_count}")

# Show the full accumulated content
content = ""
for ln in lines:
    if ln.startswith("data: ") and ln != "data: [DONE]":
        try:
            d = json.loads(ln[6:])
            choices = d.get("choices", [])
            if choices:
                c = choices[0].get("delta", {}).get("content")
                if c:
                    content += c
        except Exception:
            pass

print(f"Total content length: {len(content)}")
print(f"Has ```: {'```' in content}")
print(f"First 100: {repr(content[:100])}")

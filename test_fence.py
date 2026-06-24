"""Comprehensive test for DeepSeek streaming with json_schema via proxy."""
import httpx, json, sys

body = {
    "model": "deepseek-v4-pro",
    "messages": [
        {"role": "system", "content": "Generate presentation outline with slides array."},
        {"role": "user", "content": "对比ChatGPT、Claude、DeepSeek、通义千问四款AI助手的特点、优势、定价和适用场景。做成决策参考型幻灯片。6页。"}
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
    json=body,
    headers={"Authorization": "Bearer sk-proxy", "Content-Type": "application/json"},
    timeout=120)

sys.stdout.reconfigure(encoding="utf-8")
print("Status:", r.status_code)

# Parse streaming content
lines = r.text.strip().split("\n")
content = ""
for line in lines:
    if line.startswith("data: ") and line != "data: [DONE]":
        try:
            d = json.loads(line[6:])
            choices = d.get("choices", [])
            if choices:
                c = choices[0].get("delta", {}).get("content")
                if c:
                    content += c
        except Exception:
            pass

print(f"Content length: {len(content)}")
print(f"First 200: {repr(content[:200])}")
print(f"Last 200: {repr(content[-200:])}")

# The proxy should have stripped ```fences and preface via _process_chunk
if not content:
    print("NO CONTENT - proxy stripping may have eaten everything")
    sys.exit(1)

# Try to parse directly (proxy should return clean JSON)
try:
    parsed = json.loads(content)
    print(f"\nPARSE OK! Keys: {list(parsed.keys())}")
    slides = parsed.get("slides", [])
    if slides:
        print(f"Slides: {len(slides)}")
        for s in slides[:2]:
            print(f"  fields: {list(s.keys())}")
except json.JSONDecodeError as e:
    print(f"\nPARSE FAILED: {e}")
    # Check if it's still wrapped
    clean = content.replace("```json", "").replace("```", "").strip()
    brace = clean.find("{")
    if brace > 0:
        clean = clean[brace:]
    try:
        parsed = json.loads(clean)
        print(f"Strip-parse OK! Keys: {list(parsed.keys())}")
    except json.JSONDecodeError as e2:
        print(f"Strip-parse also failed: {e2}")

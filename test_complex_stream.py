"""Streaming complex request - test fence stripping."""
import httpx
import json
import sys
sys.stdout.reconfigure(encoding="utf-8")

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
            "name": "response", "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "slides": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {"content": {"type": "string"}},
                            "required": ["content"], "additionalProperties": False
                        }
                    }
                },
                "required": ["slides"], "additionalProperties": False
            }
        }
    }
}

with httpx.Client(timeout=120) as client:
    with client.stream("POST", "http://127.0.0.1:4003/v1/chat/completions",
                        json=body, headers={"Authorization": "Bearer sk-proxy"}) as resp:
        print(f"Status: {resp.status_code}")
        content_chunks = []
        count = 0
        for chunk in resp.iter_bytes():
            text = chunk.decode("utf-8", errors="replace")
            for line in text.split("\n"):
                if line.startswith("data: ") and line != "data: [DONE]":
                    try:
                        d = json.loads(line[6:])
                        delta = d.get("choices", [{}])[0].get("delta", {})
                        c = delta.get("content")
                        if c is not None:
                            count += 1
                            content_chunks.append(c)
                            if count <= 8:
                                print(f"  chunk #{count}: {repr(c[:100])}")
                    except Exception:
                        pass

        print(f"\nTotal content chunks: {count}")
        full = "".join(content_chunks)
        print(f"Full len: {len(full)}")
        print(f"First 100: {repr(full[:100])}")
        print(f"Last 100: {repr(full[-100:])}")
        print(f"Has ```: {'```' in full}")
        try:
            parsed = json.loads(full)
            print(f"PARSE OK! Keys: {list(parsed.keys())}")
        except json.JSONDecodeError as e:
            print(f"PARSE FAILED: {e}")

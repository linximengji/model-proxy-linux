"""Show ONLY chunks where content is not null (actual answer, not reasoning)."""
import httpx, json, sys

sys.stdout.reconfigure(encoding="utf-8")

body = {
    "model": "deepseek-v4-pro",
    "messages": [{"role": "user", "content": "Return {\"msg\":\"hello\"} as JSON."}],
    "stream": True,
    "response_format": {
        "type": "json_schema",
        "json_schema": {
            "name": "response", "strict": True,
            "schema": {
                "type": "object",
                "properties": {"msg": {"type": "string"}},
                "required": ["msg"], "additionalProperties": False
            }
        }
    }
}

with httpx.Client(timeout=60) as client:
    with client.stream("POST", "http://127.0.0.1:4003/v1/chat/completions",
                        json=body, headers={"Authorization": "Bearer sk-proxy"}) as resp:
        print(f"Status: {resp.status_code}")
        count = 0
        content_chunks = []
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
                            if count <= 10:
                                print(f"  content chunk #{count}: {repr(c[:100])}")
                    except:
                        pass

        print(f"\nTotal content chunks: {count}")
        full = "".join(content_chunks)
        print(f"Full content: {repr(full[:300])}")
        print(f"Has backticks: {'```' in full}")

"""Raw SSE test — show exactly what the proxy sends."""
import httpx, json

body = {
    "model": "deepseek-v4-pro",
    "messages": [
        {"role": "system", "content": "Return {\"msg\":\"hello\"}"},
        {"role": "user", "content": "hi"}
    ],
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

import sys
sys.stdout.reconfigure(encoding="utf-8")

with httpx.Client(timeout=60) as client:
    with client.stream("POST", "http://127.0.0.1:4003/v1/chat/completions",
                        json=body, headers={"Authorization": "Bearer sk-proxy"}) as resp:
        print(f"Status: {resp.status_code}")
        count = 0
        for chunk in resp.iter_bytes():
            text = chunk.decode("utf-8", errors="replace")
            # Only show lines with content deltas
            for line in text.split("\n"):
                if line.startswith("data: ") and '"content"' in line:
                    count += 1
                    # Show the actual content value, not the full line
                    try:
                        d = json.loads(line[6:])
                        delta = d.get("choices", [{}])[0].get("delta", {})
                        c = delta.get("content")
                        if c is not None and count <= 10:
                            print(f"  chunk #{count}: content={repr(c[:80])}")
                        elif c is not None and count == 20:
                            print(f"  chunk #{count}: content={repr(c[:80])}")
                    except:
                        pass
        print(f"Total content chunks: {count}")

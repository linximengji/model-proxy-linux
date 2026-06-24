"""Show raw byte chunks from proxy stream."""
import httpx, json, sys
sys.stdout.reconfigure(encoding="utf-8")

body = {
    "model": "deepseek-v4-pro",
    "messages": [{"role": "user", "content": "Return {\"msg\":\"hi\"} as JSON."}],
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
        count = 0
        for chunk in resp.iter_bytes():
            count += 1
            if count <= 15:
                print(f"  byte chunk #{count}: {repr(chunk[:200])}")
            else:
                remaining = b""
            remaining += chunk if count > 15 else b""

print(f"\nTotal byte chunks: {count}")

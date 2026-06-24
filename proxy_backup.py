"""Emergency backup proxy — degrades all requests to DeepSeek Flash (cheapest).

No proxy_lib dependency. Every model → DeepSeek Flash regardless of what the
client requests. Preserves MAAS/DashScope quota for main proxy (4000).

Endpoints: /v1/messages (Anthropic format), /v1/chat/completions (OpenAI format),
/health, /v1/models.
"""
import json
import os
import sys

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from contextlib import asynccontextmanager

sys.stdout.reconfigure(encoding="utf-8")

DEEPSEEK_ANTHROPIC_URL = "https://api.deepseek.com/anthropic/v1/messages"
DEEPSEEK_OPENAI_URL = "https://api.deepseek.com/v1/chat/completions"

# Degrade every model to DeepSeek Flash — cheapest tier, preserves MAAS quota.
DEGRADED_UPSTREAM = "deepseek-chat"

ALLOWED_ANTHROPIC_KEYS = {
    "model", "messages", "max_tokens", "stream",
    "temperature", "top_p", "top_k",
    "frequency_penalty", "presence_penalty",
    "stop", "stop_sequences",
    "tools", "tool_choice", "system", "thinking",
}
ALLOWED_BLOCK_TYPES = {"text", "tool_use", "tool_result",
                       "thinking", "redacted_thinking", "tool_reference"}
ALLOWED_OPENAI_KEYS = {
    "model", "messages", "max_tokens", "stream",
    "temperature", "top_p", "frequency_penalty",
    "presence_penalty", "stop", "tools", "tool_choice",
}

http_client: httpx.AsyncClient | None = None


def _sanitize_anthropic(body: dict):
    """Strip CC-specific fields DeepSeek Anthropic endpoint doesn't support."""
    for k in [k for k in body if k not in ALLOWED_ANTHROPIC_KEYS]:
        del body[k]
    sf = body.get("system")
    if isinstance(sf, list):
        body["system"] = "\n\n".join(
            b.get("text", "") if isinstance(b, dict) else str(b)
            for b in sf if b
        )
    msgs = body.get("messages")
    if not msgs:
        return
    sys_parts = []
    if isinstance(body.get("system"), str) and body["system"].strip():
        sys_parts.append(body["system"])
    cleaned = []
    for m in msgs:
        if m.get("role") == "system":
            c = m.get("content", "")
            if isinstance(c, list):
                c = "\n\n".join(
                    b.get("text", "") for b in c
                    if isinstance(b, dict) and b.get("type") == "text"
                )
            if c.strip():
                sys_parts.append(c)
            continue
        m.pop("cache_control", None)
        content = m.get("content")
        if isinstance(content, list):
            blocks = [b for b in content
                      if isinstance(b, dict) and b.get("type") in ALLOWED_BLOCK_TYPES]
            for b in blocks:
                b.pop("cache_control", None)
            m["content"] = blocks if blocks else [{"type": "text", "text": ""}]
        cleaned.append(m)
    body["messages"] = cleaned
    if sys_parts:
        body["system"] = "\n\n".join(sys_parts)


def _sanitize_openai(body: dict):
    """Strip params DeepSeek OpenAI endpoint doesn't support."""
    for k in [k for k in body if k not in ALLOWED_OPENAI_KEYS]:
        del body[k]


async def _stream_forward(url: str, upstream_body: dict, api_key: str):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
        "Accept": "text/event-stream",
    }
    async def gen():
        async with http_client.stream(
            "POST", url,
            content=json.dumps(upstream_body).encode("utf-8"),
            headers=headers, timeout=300,
        ) as resp:
            if resp.status_code != 200:
                yield await resp.aread()
                return
            async for line in resp.aiter_lines():
                yield (line.encode("utf-8") if isinstance(line, str) else line) + b"\n"
    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache"})


async def _request_forward(url: str, upstream_body: dict, api_key: str):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    resp = await http_client.post(
        url,
        content=json.dumps(upstream_body).encode("utf-8"),
        headers=headers, timeout=180,
    )
    return Response(content=resp.content, media_type="application/json",
                    status_code=resp.status_code)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global http_client
    http_client = httpx.AsyncClient(timeout=httpx.Timeout(300.0))
    yield
    if http_client:
        await http_client.aclose()


app = FastAPI(openapi_url=None, docs_url=None, redoc_url=None, lifespan=lifespan)


@app.get("/health")
async def health():
    return JSONResponse({"status": "ok"})


@app.get("/v1/models")
async def list_models():
    return JSONResponse({
        "object": "list",
        "data": [{"id": "deepseek-v4-flash", "object": "model", "created": 1, "owned_by": "backup"}],
    })


@app.post("/v1/messages")
async def proxy_anthropic(request: Request):
    body = await request.json()
    is_stream = body.get("stream", False)
    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    upstream = body.copy()
    upstream["model"] = DEGRADED_UPSTREAM
    upstream["stream"] = is_stream
    _sanitize_anthropic(upstream)
    if is_stream:
        return await _stream_forward(DEEPSEEK_ANTHROPIC_URL, upstream, api_key)
    return await _request_forward(DEEPSEEK_ANTHROPIC_URL, upstream, api_key)


@app.post("/v1/chat/completions")
async def proxy_openai(request: Request):
    body = await request.json()
    is_stream = body.get("stream", False)
    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    upstream = body.copy()
    upstream["model"] = DEGRADED_UPSTREAM
    _sanitize_openai(upstream)
    if is_stream:
        return await _stream_forward(DEEPSEEK_OPENAI_URL, upstream, api_key)
    return await _request_forward(DEEPSEEK_OPENAI_URL, upstream, api_key)


@app.post("/{path:path}")
async def not_found(path: str):
    return JSONResponse({"error": f"unsupported path: /{path}"}, status_code=404)


def main():
    port = 4002
    for a in sys.argv[1:]:
        if not a.startswith("-"):
            try:
                port = int(a)
                break
            except ValueError:
                pass

    dotenv = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if os.path.isfile(dotenv):
        with open(dotenv, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, _, v = line.partition("=")
                k, v = k.strip(), v.strip().strip('"').strip("'")
                if k and k not in os.environ:
                    os.environ[k] = v

    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")


if __name__ == "__main__":
    main()

"""Emergency backup proxy — routes by model name, no proxy_lib deps."""
import json
import os
import sys

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from contextlib import asynccontextmanager

sys.stdout.reconfigure(encoding="utf-8")

DEEPSEEK_BASE = "https://api.deepseek.com/anthropic/v1/messages"
MAAS_ANTHROPIC_BASE = "https://token-plan.cn-beijing.maas.aliyuncs.com/apps/anthropic/v1/messages"
DEEPSEEK_KEY = ""
MAAS_KEY = ""

# ── Model routing table ───────────────────────────────────────────────────
# Each route specifies provider type and upstream connection details.
# "deepseek" / "anthropic" → sends Anthropic-format body to the upstream.
# "openai"                 → sends OpenAI-format body (needs conversion).
ROUTES = {
    "deepseek-v4-pro": {
        "type": "anthropic",  # Anthropic-format passthrough, no sanitize
        "api_base": DEEPSEEK_BASE,
        "upstream_model": "deepseek-chat",
        "api_key_env": "DEEPSEEK_API_KEY",
    },
    "deepseek-v4-flash": {
        "type": "anthropic",
        "api_base": DEEPSEEK_BASE,
        "upstream_model": "deepseek-chat",
        "api_key_env": "DEEPSEEK_API_KEY",
    },
    "qwen3.7-max": {
        "type": "anthropic",
        "api_base": MAAS_ANTHROPIC_BASE,
        "upstream_model": "qwen3.7-max",
        "api_key_env": "QWEN_MAAS_API_KEY",
    },
    "qwen3.6-plus": {
        "type": "anthropic",
        "api_base": MAAS_ANTHROPIC_BASE,
        "upstream_model": "qwen3.6-plus",
        "api_key_env": "QWEN_MAAS_API_KEY",
    },
}

ALLOWED_KEYS = {
    "model", "messages", "max_tokens", "stream",
    "temperature", "top_p", "top_k",
    "frequency_penalty", "presence_penalty",
    "stop", "stop_sequences",
    "tools", "tool_choice", "system", "thinking",
}
ALLOWED_BLOCK_TYPES = {"text", "tool_use", "tool_result",
                       "thinking", "redacted_thinking", "tool_reference"}


def _sanitize_for_deepseek(body):
    """Strip CC/Claude-specific fields DeepSeek doesn't support.

    Used only for deepseek routes — MAAS Anthropic accepts full format.
    """
    for k in [k for k in body if k not in ALLOWED_KEYS]:
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
            blocks = []
            for b in content:
                if not isinstance(b, dict) or b.get("type") not in ALLOWED_BLOCK_TYPES:
                    continue
                b.pop("cache_control", None)
                blocks.append(b)
            m["content"] = blocks if blocks else [{"type": "text", "text": ""}]
        cleaned.append(m)

    body["messages"] = cleaned
    if sys_parts:
        body["system"] = "\n\n".join(sys_parts)


def _get_api_key(env_name):
    return os.environ.get(env_name, "")


async def _forward_anthropic(body, is_stream, route):
    """Forward to an Anthropic-format upstream (DeepSeek or MAAS)."""
    api_key = _get_api_key(route["api_key_env"])
    upstream = body.copy()
    upstream["model"] = route["upstream_model"]
    upstream["stream"] = is_stream

    # Sanitize only for DeepSeek, not for MAAS Anthropic
    if route.get("api_base", "").startswith("https://api.deepseek.com"):
        _sanitize_for_deepseek(upstream)

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    url = route["api_base"]

    if is_stream:
        headers["Accept"] = "text/event-stream"
        async def gen():
            async with http_client.stream(
                "POST", url,
                content=json.dumps(upstream).encode("utf-8"),
                headers=headers,
                timeout=300,
            ) as resp:
                if resp.status_code != 200:
                    err_body = await resp.aread()
                    yield err_body
                    return
                async for line in resp.aiter_lines():
                    yield (line.encode("utf-8") if isinstance(line, str) else line) + b"\n"
        return StreamingResponse(gen(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache"})

    resp = await http_client.post(
        url, content=json.dumps(upstream).encode("utf-8"),
        headers=headers, timeout=180,
    )
    return Response(content=resp.content, media_type="application/json",
                    status_code=resp.status_code)


http_client: httpx.AsyncClient | None = None


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
    models = [
        {"id": name, "object": "model", "created": 1, "owned_by": "backup"}
        for name in ROUTES
    ]
    return JSONResponse({"object": "list", "data": models})


@app.post("/v1/messages")
async def proxy_messages(request: Request):
    body = await request.json()
    model_name = body.get("model", "deepseek-v4-pro")
    is_stream = body.get("stream", False)

    route = ROUTES.get(model_name)
    if not route:
        return JSONResponse(
            {"error": f"backup proxy: unknown model '{model_name}'"},
            status_code=404,
        )

    return await _forward_anthropic(body, is_stream, route)


@app.post("/{path:path}")
async def not_found(path: str):
    return JSONResponse({"error": f"unsupported path: /{path}"}, status_code=404)


def main():
    port = 4001
    for a in sys.argv[1:]:
        if not a.startswith("-"):
            try:
                port = int(a)
            except ValueError:
                pass
            break

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

    global DEEPSEEK_KEY, MAAS_KEY
    DEEPSEEK_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
    MAAS_KEY = os.environ.get("QWEN_MAAS_API_KEY", "")

    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")


if __name__ == "__main__":
    main()

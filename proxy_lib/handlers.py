"""HTTP handlers for Anthropic and OpenAI message endpoints. Async, depends on full proxy_lib."""
import json
import time
import re
import copy

import httpx
from fastapi.responses import Response, JSONResponse, StreamingResponse

from proxy_lib import config, sanitize, convert, telemetry, fallback

log = telemetry.log


def _strip_strict_from_rf(body: dict) -> dict:
    """Remove 'strict' from response_format for models that don't support it (Qwen etc)."""
    rf = body.get("response_format")
    if rf and isinstance(rf, dict):
        js = rf.get("json_schema")
        if js and isinstance(js, dict) and js.get("strict") is not None:
            body = copy.deepcopy(body)
            body["response_format"] = copy.deepcopy(rf)
            del body["response_format"]["json_schema"]["strict"]
    return body


def sse_encode(event, data):
    return f"event: {event}\ndata: {json.dumps(data)}\n\n".encode("utf-8")


async def _error_response(e):
    if e is None:
        return JSONResponse({"error": "all upstreams exhausted"}, status_code=502)
    if isinstance(e, httpx.HTTPStatusError):
        body = e.response.content
        log(f"UPSTREAM ERR {e.response.status_code}: {body[:500]}", "ERROR", "UPSTREAM")
        await telemetry.record_error()
        return Response(content=body, status_code=e.response.status_code,
                        media_type="application/json")
    return JSONResponse({"error": str(e)}, status_code=502)


# Anthropic non-stream

async def handle_anthropic(body, route, model_name, routes, http_client,
                           work_dir=None, session_id=None):
    def build_kwargs(r, _m):
        if r["provider"] in ("deepseek", "anthropic"):
            upstream = body.copy()
            upstream["model"] = r["model"]
            upstream["stream"] = False
            if r["provider"] == "deepseek":
                sanitize.sanitize_for_deepseek(upstream)
            log(f"Anthropic non-stream body samples: model={body.get('model')}, has_rf={'response_format' in body}, rf_body={bool(body.get('response_format'))}, has_stream=stream", phase="HANDLER")
            return {
                "method": "POST", "url": r["api_base"],
                "content": json.dumps(upstream).encode("utf-8"),
                "headers": {"Content-Type": "application/json",
                            "Authorization": f"Bearer {r['api_key']}"},
            }
        api_base = r.get("api_base", "https://api.openai.com/v1")
        oai_body = convert.anthropic_to_openai(body)
        oai_body["model"] = r["model"]
        oai_body["stream"] = False
        return {
            "method": "POST", "url": f"{api_base}/chat/completions",
            "json": oai_body,
            "headers": {"Content-Type": "application/json",
                        "Authorization": f"Bearer {r['api_key']}"},
        }

    try:
        resp, used_model = await fallback.request_with_fallback(
            route, model_name, routes, http_client, build_kwargs, timeout=180)
        used_route = routes.get(used_model, route)
        if used_route["provider"] in ("deepseek", "anthropic"):
            data = resp.content
            try:
                usage = json.loads(data).get("usage", {})
                if usage:
                    await telemetry.record_tokens(used_model, usage.get("input_tokens", 0),
                                            usage.get("output_tokens", 0),
                                            usage.get("cache_read_input_tokens", 0),
                                            work_dir, session_id)
            except (json.JSONDecodeError, AttributeError):
                pass
            if resp.status_code == 400:
                err_text = data.decode("utf-8", errors="replace") if isinstance(data, bytes) else str(data)
                log(f"<- 400 {used_model} body: {err_text[:2000]}", "WARN", "UPSTREAM")
            return Response(content=data, media_type="application/json",
                            status_code=resp.status_code)
        oai_data = resp.json()
        result = convert.openai_to_anthropic(oai_data, used_model)
        log(f"<- 200 {used_model} non-stream", phase="UPSTREAM")
        usage = oai_data.get("usage", {})
        if usage:
            await telemetry.record_tokens(used_model, usage.get("prompt_tokens", 0),
                                    usage.get("completion_tokens", 0),
                                    work_dir=work_dir, session_id=session_id)
        return JSONResponse(result)
    except httpx.HTTPStatusError as e:
        log(f"<- ERR {e.response.status_code} {model_name}: {e.response.reason_phrase}", "ERROR", "UPSTREAM")
        return await _error_response(e)
    except (httpx.RequestError, httpx.TimeoutException) as e:
        log(f"<- ERR {type(e).__name__} {model_name}", "ERROR", "UPSTREAM")
        return await _error_response(e)


# Anthropic stream

async def handle_anthropic_stream(body, route, model_name, routes, http_client,
                                  work_dir=None, session_id=None):
    models_to_try = fallback.build_fallback_chain(route, model_name, routes)

    async def event_generator():
        nonlocal model_name
        last_err = None
        i = 0
        while i < len(models_to_try):
            r, m = models_to_try[i]
            try:
                if r["provider"] in ("deepseek", "anthropic"):
                    kwargs = _native_stream_kwargs(body, r, m)
                    kwargs["timeout"] = 300
                    async with http_client.stream(**kwargs) as resp:
                        await fallback.telemetry.record_success(m)
                        model_name = m
                        inp = out = cache = 0
                        async for line in resp.aiter_lines():
                            lb = line.encode("utf-8") if isinstance(line, str) else line
                            yield lb + b"\n"
                            s = line.strip()
                            if s.startswith("data: ") and '"usage"' in s:
                                try:
                                    usage = json.loads(s[6:]).get("usage", {})
                                    if usage:
                                        inp = usage.get("input_tokens", inp)
                                        out = usage.get("output_tokens", out)
                                        cache = usage.get("cache_read_input_tokens", cache)
                                except json.JSONDecodeError:
                                    pass
                        if inp or out:
                            await telemetry.record_tokens(model_name, inp, out, cache,
                                                    work_dir, session_id)
                        return
                else:
                    kwargs = _openai_stream_kwargs(body, r, m)
                    kwargs["timeout"] = 300
                    async with http_client.stream(**kwargs) as resp:
                        if resp.status_code != 200:
                            err_body = await resp.aread()
                            raise httpx.HTTPStatusError(
                                f"HTTP {resp.status_code}", request=resp.request, response=resp)
                        await fallback.telemetry.record_success(m)
                        model_name = m
                        inp = out = 0
                        ti = None
                        tbs = {}
                        nbi = 0
                        finished = False
                        ms = False  # message_start sent flag
                        async for line in resp.aiter_lines():
                            d = line.strip()
                            if not d or not d.startswith("data:"):
                                continue
                            ds = d[5:].strip()
                            if ds == "[DONE]" or finished:
                                break
                            try:
                                chunk = json.loads(ds)
                            except json.JSONDecodeError:
                                continue
                            choices = chunk.get("choices", [])
                            if not choices:
                                continue
                            delta = choices[0].get("delta", {})
                            finish = choices[0].get("finish_reason")
                            content = delta.get("content")
                            if content:
                                if ti is None:
                                    if not ms:
                                        ms = True
                                        yield sse_encode("message_start", {
                                            "type": "message_start", "message": {"id": "", "model": m, "role": "assistant", "content": []},
                                        })
                                    ti = nbi; nbi += 1
                                    yield sse_encode("content_block_start", {
                                        "type": "content_block_start", "index": ti,
                                        "content_block": {"type": "text", "text": ""},
                                    })
                                yield sse_encode("content_block_delta", {
                                    "type": "content_block_delta", "index": ti,
                                    "delta": {"type": "text_delta", "text": content},
                                })
                            tcs = delta.get("tool_calls")
                            if tcs:
                                for tc in tcs:
                                    tci = tc.get("index", 0)
                                    func_delta = tc.get("function", {})
                                    if tci not in tbs:
                                        if not ms:
                                            ms = True
                                            yield sse_encode("message_start", {
                                                "type": "message_start", "message": {"id": "", "model": m, "role": "assistant", "content": []},
                                            })
                                        tb = {"idx": nbi, "id": tc.get("id", ""), "name": func_delta.get("name", "")}
                                        tbs[tci] = tb
                                        nbi += 1
                                        yield sse_encode("content_block_start", {
                                            "type": "content_block_start", "index": tb["idx"],
                                            "content_block": {"type": "tool_use", "id": tb["id"], "name": tb["name"], "input": {}},
                                        })
                                    else:
                                        tb = tbs[tci]
                                        if tc.get("id"): tb["id"] = tc["id"]
                                        if func_delta.get("name"): tb["name"] = func_delta["name"]
                                    if func_delta.get("arguments"):
                                        yield sse_encode("content_block_delta", {
                                            "type": "content_block_delta", "index": tb["idx"],
                                            "delta": {"type": "input_json_delta", "partial_json": func_delta["arguments"]},
                                        })
                            usage = chunk.get("usage")
                            if usage:
                                inp = usage.get("prompt_tokens", inp)
                                out = usage.get("completion_tokens", out)
                            if finish:
                                finished = True
                                if ti is not None:
                                    yield sse_encode("content_block_stop",
                                                     {"type": "content_block_stop", "index": ti})
                                for tci in sorted(tbs):
                                    yield sse_encode("content_block_stop",
                                                     {"type": "content_block_stop", "index": tbs[tci]["idx"]})
                                sr = "end_turn"
                                if finish == "length": sr = "max_tokens"
                                elif finish == "tool_calls": sr = "tool_use"
                                yield sse_encode("message_delta", {
                                    "type": "message_delta",
                                    "delta": {"stop_reason": sr},
                                    "usage": {"output_tokens": out},
                                })
                                yield sse_encode("message_stop", {"type": "message_stop"})
                        if inp or out:
                            await telemetry.record_tokens(model_name, inp, out, work_dir=work_dir,
                                                    session_id=session_id)
                        return
            except httpx.HTTPStatusError as e:
                code = e.response.status_code
                await telemetry.record_failure(m, status_code=code)
                last_err = e
                if i == 0 and fallback.is_quota_exhausted(e):
                    qb_name = r.get("quota_backup")
                    if qb_name and qb_name in routes and qb_name not in {mm for _, mm in models_to_try}:
                        models_to_try.insert(1, (routes[qb_name], qb_name))
                        log(f"<- {m} quota exhausted, switching to {qb_name}", phase="FALLBACK")
                        i += 1; continue
                if i < len(models_to_try) - 1:
                    log(f"<- {m} ({code}), fallback to {models_to_try[i+1][1]}", phase="FALLBACK")
                else:
                    log(f"<- {m} ({code}), no more fallback", "WARN", "FALLBACK")
                i += 1; continue
            except (httpx.RequestError, httpx.TimeoutException) as e:
                await telemetry.record_failure(m, error_type="connection")
                last_err = e
                code = type(e).__name__
                if i < len(models_to_try) - 1:
                    log(f"<- {m} ({code}), fallback to {models_to_try[i+1][1]}", phase="FALLBACK")
                else:
                    log(f"<- {m} ({code}), no more fallback", "WARN", "FALLBACK")
                i += 1; continue
        if last_err and isinstance(last_err, httpx.HTTPStatusError):
            err_body = last_err.response.content
            log(f"UPSTREAM ERR {last_err.response.status_code}: {err_body[:500]}", "ERROR", "UPSTREAM")
            await telemetry.record_error()
            yield err_body

    return StreamingResponse(event_generator(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache"})


# OpenAI non-stream

async def handle_openai(body, route, model_name, routes, http_client,
                        work_dir=None, session_id=None):
    def build_kwargs(r, _m):
        if r["provider"] == "deepseek":
            api_base = "https://api.deepseek.com/v1"
        else:
            api_base = r.get("api_base") or "https://api.openai.com/v1"
        url = f"{api_base}/chat/completions"
        _body = body
        # Non-deepseek providers often don't support strict in json_schema
        if r["provider"] != "deepseek":
            _body = _strip_strict_from_rf(body)
        oai_body = convert.anthropic_to_openai(_body)
        upstream = {
            "model": r["model"],
            "messages": oai_body["messages"],
            "max_tokens": _body.get("max_tokens", r.get("max_tokens", 4096)),
        }
        for key in ("temperature", "tools", "tool_choice", "top_p", "frequency_penalty", "presence_penalty", "response_format"):
            if key == "response_format" and r["provider"] == "deepseek":
                rf = _body.get("response_format")
                if rf and isinstance(rf, dict) and rf.get("type") in ("json_schema", "json_object"):
                    upstream["response_format"] = {"type": "json_object"}
                continue
            if _body.get(key) is not None:
                upstream[key] = _body[key]
            elif r.get(key) is not None and key not in upstream:
                upstream[key] = r[key]
        if "response_format" in upstream and upstream["response_format"].get("type") == "json_object" and r["provider"] == "deepseek":
            upstream["messages"].append({
                "role": "system",
                "content": "Output ONLY valid JSON. No other text or explanation."
            })
        return {
            "method": "POST", "url": url,
            "json": upstream,
            "headers": {"Content-Type": "application/json",
                        "Authorization": f"Bearer {r['api_key']}"},
        }
    try:
        resp, _used = await fallback.request_with_fallback(
            route, model_name, routes, http_client, build_kwargs, timeout=120)
        body_json = resp.json()
        usage = body_json.get("usage", {})
        if usage:
            await telemetry.record_tokens(_used, usage.get("prompt_tokens", 0),
                                    usage.get("completion_tokens", 0),
                                    work_dir=work_dir, session_id=session_id)
        if body.get("response_format") and body_json.get("choices"):
            for c in body_json["choices"]:
                msg = c.get("message", {})
                content = msg.get("content", "")
                if not content:
                    continue
                # DeepSeek may wrap JSON in fences, prefix with text, or ignore response_format entirely.
                # Extract the first balanced JSON block.
                cleaned = content.replace("```json", "").replace("```", "").strip()
                # Scan for the first { or [ and extract a balanced block
                json_start = -1
                json_end = -1
                for i, ch in enumerate(cleaned):
                    if ch in ("{", "["):
                        json_start = i
                        break
                if json_start >= 0:
                    depth = 0
                    in_str = False
                    escape = False
                    open_char = cleaned[json_start]
                    close_char = "}" if open_char == "{" else "]"
                    for i in range(json_start, len(cleaned)):
                        ch = cleaned[i]
                        if escape:
                            escape = False
                            continue
                        if ch == "\\":
                            escape = True
                            continue
                        if ch == '"':
                            in_str = not in_str
                            continue
                        if in_str:
                            continue
                        if ch == open_char:
                            depth += 1
                        elif ch == close_char:
                            depth -= 1
                            if depth == 0:
                                json_end = i + 1
                                break
                if json_end > json_start:
                    extracted = cleaned[json_start:json_end]
                    try:
                        json.loads(extracted)
                        c["message"]["content"] = extracted
                    except json.JSONDecodeError:
                        pass  # leave original
        log(f"<- 200 {model_name} openai-nonstream", phase="UPSTREAM")
        return JSONResponse(content=body_json, status_code=resp.status_code)
    except httpx.HTTPStatusError as e:
        log(f"<- ERR {e.response.status_code} {model_name} openai: {e.response.reason_phrase}", "ERROR", "UPSTREAM")
        return await _error_response(e)
    except (httpx.RequestError, httpx.TimeoutException) as e:
        log(f"<- ERR {type(e).__name__} {model_name} openai", "ERROR", "UPSTREAM")
        return await _error_response(e)


# OpenAI stream

async def handle_openai_stream(body, route, model_name, routes, http_client,
                                work_dir=None, session_id=None):
    def build_kwargs(r, _m):
        if r["provider"] == "deepseek":
            api_base = "https://api.deepseek.com/v1"
        else:
            api_base = r.get("api_base") or "https://api.openai.com/v1"
        url = f"{api_base}/chat/completions"
        _body = body
        if r["provider"] != "deepseek":
            _body = _strip_strict_from_rf(body)
        oai_body = convert.anthropic_to_openai(_body)
        upstream = {
            "model": r["model"],
            "messages": oai_body["messages"],
            "max_tokens": _body.get("max_tokens", r.get("max_tokens", 4096)),
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        for key in ("temperature", "tools", "tool_choice", "top_p", "frequency_penalty", "presence_penalty", "response_format"):
            if key == "response_format" and r["provider"] == "deepseek":
                rf = _body.get("response_format")
                if rf and isinstance(rf, dict) and rf.get("type") in ("json_schema", "json_object"):
                    upstream["response_format"] = {"type": "json_object"}
                continue
            if _body.get(key) is not None:
                upstream[key] = _body[key]
            elif r.get(key) is not None and key not in upstream:
                upstream[key] = r[key]
        if "response_format" in upstream and upstream["response_format"].get("type") == "json_object" and r["provider"] == "deepseek":
            upstream["messages"].append({
                "role": "system",
                "content": "Output ONLY valid JSON. No other text or explanation."
            })
        return {
            "method": "POST", "url": url,
            "json": upstream,
            "headers": {"Content-Type": "application/json",
                        "Authorization": f"Bearer {r['api_key']}",
                        "Accept": "text/event-stream"},
        }

    models_to_try = fallback.build_fallback_chain(route, model_name, routes)

    # When response_format is set, DeepSeek may prefix JSON with explanatory text
    # and wrap it in markdown fences. We strip non-JSON content in-stream.
    has_response_format = bool(body.get("response_format"))
    json_started = not has_response_format
    line_buffer = b""

    async def _process_chunk(raw: bytes) -> bytes:
        nonlocal line_buffer, json_started
        line_buffer += raw
        if b"\n" not in line_buffer:
            return b""
        lines = line_buffer.split(b"\n")
        line_buffer = lines.pop()
        out_parts = []
        for ln in lines:
            if not has_response_format or not ln.startswith(b"data: "):
                out_parts.append(ln + b"\n")
                continue
            try:
                decoded = ln[6:].decode("utf-8")
                chunk_data = json.loads(decoded)
                choices = chunk_data.get("choices", [])
                if not choices or "delta" not in choices[0]:
                    out_parts.append(ln + b"\n")
                    continue
                delta = choices[0]["delta"]
                raw_text = delta.get("content")
                if not raw_text:
                    out_parts.append(ln + b"\n")
                    continue
                # Strip markdown fences — DeepSeek wraps JSON in ```json...```
                cleaned = raw_text.replace("```json", "").replace("```", "")
                if not json_started:
                    # Still looking for start of JSON value
                    brace = cleaned.find("{")
                    bracket = cleaned.find("[")
                    start = -1
                    if brace >= 0 and bracket >= 0:
                        start = min(brace, bracket)
                    elif brace >= 0:
                        start = brace
                    elif bracket >= 0:
                        start = bracket
                    if start >= 0:
                        json_started = True
                        cleaned = cleaned[start:]
                    else:
                        # Preface/discard — drop this chunk
                        continue
                delta["content"] = cleaned
                new_line = b"data: " + json.dumps(chunk_data, ensure_ascii=False).encode("utf-8") + b"\n"
                out_parts.append(new_line)
            except (json.JSONDecodeError, UnicodeDecodeError):
                out_parts.append(ln + b"\n")
        return b"".join(out_parts)

    async def event_generator():
        nonlocal line_buffer
        last_err = None
        for i, (r, m) in enumerate(models_to_try):
            try:
                kwargs = build_kwargs(r, m)
                kwargs["timeout"] = 300
                async with http_client.stream(**kwargs) as resp:
                    if resp.status_code != 200:
                        err_body = await resp.aread()
                        raise httpx.HTTPStatusError(
                            f"HTTP {resp.status_code}", request=resp.request, response=resp)
                    await fallback.telemetry.record_success(m)
                    log(f"<- 200 streaming {m} (openai passthrough, has_rf={has_response_format})", phase="UPSTREAM")
                    content_count = 0
                    inp = out = 0
                    async for chunk in resp.aiter_bytes():
                        # Extract usage from raw chunk before processing
                        for raw_line in chunk.split(b"\n"):
                            if raw_line.startswith(b"data: ") and b'"usage"' in raw_line:
                                try:
                                    ud = json.loads(raw_line[6:].decode("utf-8"))
                                    u = ud.get("usage", {})
                                    if u:
                                        inp = u.get("prompt_tokens", inp)
                                        out = u.get("completion_tokens", out)
                                except (json.JSONDecodeError, UnicodeDecodeError):
                                    pass
                        processed = await _process_chunk(chunk)
                        if processed:
                            for ln in processed.split(b"\n"):
                                if ln.startswith(b"data: ") and b'"content":' in ln:
                                    content_count += 1
                            yield processed
                    log(f"...stream output content chunks: {content_count}", phase="UPSTREAM")
                    # flush line buffer
                    if line_buffer:
                        processed = await _process_chunk(line_buffer + b"\n")
                        if processed:
                            yield processed
                    if inp or out:
                        await telemetry.record_tokens(m, inp, out,
                                                work_dir=work_dir, session_id=session_id)
                    return
            except httpx.HTTPStatusError as e:
                code = e.response.status_code
                await fallback.telemetry.record_failure(m, status_code=code)
                last_err = e
                if i < len(models_to_try) - 1:
                    log(f"<- {m} ({code}), fallback to {models_to_try[i+1][1]}", phase="FALLBACK")
                else:
                    log(f"<- {m} ({code}), no more fallback", "WARN", "FALLBACK")
                continue
            except (httpx.RequestError, httpx.TimeoutException) as e:
                await fallback.telemetry.record_failure(m, error_type="connection")
                last_err = e
                code = type(e).__name__
                if i < len(models_to_try) - 1:
                    log(f"<- {m} ({code}), fallback to {models_to_try[i+1][1]}", phase="FALLBACK")
                else:
                    log(f"<- {m} ({code}), no more fallback", "WARN", "FALLBACK")
                continue
        if last_err:
            log(f"openai_stream: all upstreams exhausted", "ERROR", "UPSTREAM")
            await telemetry.record_error()

    return StreamingResponse(event_generator(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache"})


# Stream kwargs builders

def _native_stream_kwargs(body, route, _model_name):
    upstream = body.copy()
    upstream["model"] = route["model"]
    upstream["stream"] = True
    if route["provider"] == "deepseek":
        sanitize.sanitize_for_deepseek(upstream)
    return {
        "method": "POST", "url": route["api_base"],
        "content": json.dumps(upstream).encode("utf-8"),
        "headers": {"Content-Type": "application/json",
                     "Authorization": f"Bearer {route['api_key']}",
                     "Accept": "text/event-stream"},
    }


def _openai_stream_kwargs(body, route, _model_name):
    api_base = route.get("api_base", "https://api.openai.com/v1")
    oai_body = convert.anthropic_to_openai(body)
    oai_body["model"] = route["model"]
    oai_body["stream"] = True
    oai_body.setdefault("stream_options", {"include_usage": True})
    return {
        "method": "POST", "url": f"{api_base}/chat/completions",
        "json": oai_body,
        "headers": {"Content-Type": "application/json",
                     "Authorization": f"Bearer {route['api_key']}",
                     "Accept": "text/event-stream"},
    }

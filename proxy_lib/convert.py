"""Anthropic ↔ OpenAI message format conversion. Pure functions, no side effects."""
import json
import uuid


def anthropic_to_openai(body, strip_images=False):
    messages = []
    system = body.get("system")
    if isinstance(system, str):
        messages.append({"role": "system", "content": system})
    elif isinstance(system, list):
        for s in system:
            if isinstance(s, dict) and s.get("type") == "text":
                messages.append({"role": "system", "content": s["text"]})

    for msg in body.get("messages", []):
        role = msg.get("role", "user")
        content = msg.get("content", "")

        if not isinstance(content, list):
            messages.append({"role": role, "content": content})
            continue

        tool_uses = [b for b in content if isinstance(b, dict) and b.get("type") == "tool_use"]
        if role == "assistant" and tool_uses:
            texts = [b["text"] for b in content if isinstance(b, dict) and b.get("type") == "text"]
            text_content = "\n".join(texts) if texts else None
            tool_calls = []
            for tb in tool_uses:
                inp = tb.get("input", {})
                tool_calls.append({
                    "id": tb["id"],
                    "type": "function",
                    "function": {
                        "name": tb["name"],
                        "arguments": json.dumps(inp) if isinstance(inp, dict) else str(inp),
                    }
                })
            msg_oai = {"role": "assistant"}
            if text_content:
                msg_oai["content"] = text_content
            else:
                msg_oai["content"] = None
            if tool_calls:
                msg_oai["tool_calls"] = tool_calls
            messages.append(msg_oai)
            continue

        tool_results = [b for b in content if isinstance(b, dict) and b.get("type") == "tool_result"]
        if tool_results:
            for tr in tool_results:
                c = tr.get("content", "")
                if isinstance(c, list):
                    c = " ".join(b.get("text", "") for b in c if isinstance(b, dict))
                messages.append({
                    "role": "tool",
                    "tool_call_id": tr["tool_use_id"],
                    "content": c,
                })
            texts = [b["text"] for b in content if isinstance(b, dict) and b.get("type") == "text"]
            if texts:
                messages.append({"role": "user", "content": "\n".join(texts)})
            continue

        has_image = any(isinstance(b, dict) and b.get("type") in ("image", "image_url") for b in content)
        if has_image:
            if strip_images:
                parts = []
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    t = block.get("type", "")
                    if t == "text":
                        parts.append(block["text"])
                    elif t in ("image", "image_url"):
                        parts.append("[图片(已忽略，当前模型不支持图像)]")
                messages.append({"role": role, "content": "\n".join(p for p in parts if p)})
            else:
                blocks = []
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    t = block.get("type", "")
                    if t == "text":
                        blocks.append({"type": "text", "text": block["text"]})
                    elif t == "image":
                        src = block.get("source", {})
                        media_type = src.get("media_type", "image/jpeg")
                        data = src.get("data", "")
                        blocks.append({
                            "type": "image_url",
                            "image_url": {"url": f"data:{media_type};base64,{data}"}
                        })
                    elif t == "image_url":
                        blocks.append(block)
                messages.append({"role": role, "content": blocks})
        else:
            texts = [b["text"] for b in content if isinstance(b, dict) and b.get("type") == "text"]
            messages.append({"role": role, "content": "\n".join(texts)})

    oai = {"messages": messages, "max_tokens": body.get("max_tokens", 4096)}
    if body.get("temperature") is not None:
        oai["temperature"] = body["temperature"]
    if body.get("stop_sequences"):
        oai["stop"] = body["stop_sequences"]

    if body.get("tools"):
        oai["tools"] = [{
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t.get("description", ""),
                "parameters": t.get("input_schema", {}),
            }
        } for t in body["tools"]]

    return oai


def openai_to_anthropic(data, model_name):
    choice = data.get("choices", [{}])[0]
    msg = choice.get("message", {})
    content = msg.get("content") or ""
    fr = choice.get("finish_reason", "")
    if fr == "stop":
        stop_reason = "end_turn"
    elif fr == "length":
        stop_reason = "max_tokens"
    elif fr == "tool_calls":
        stop_reason = "tool_use"
    else:
        stop_reason = "end_turn"
    usage = data.get("usage", {})

    content_blocks = []
    if content:
        content_blocks.append({"type": "text", "text": content})

    for tc in msg.get("tool_calls", []):
        func = tc.get("function", {})
        try:
            arguments = json.loads(func.get("arguments", "{}"))
        except (json.JSONDecodeError, TypeError):
            arguments = func.get("arguments", {})
        content_blocks.append({
            "type": "tool_use",
            "id": tc["id"],
            "name": func.get("name", ""),
            "input": arguments,
        })

    if not content_blocks:
        content_blocks.append({"type": "text", "text": ""})

    return {
        "id": data.get("id", str(uuid.uuid4())),
        "model": model_name,
        "type": "message",
        "role": "assistant",
        "content": content_blocks,
        "stop_reason": stop_reason,
        "usage": {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
        },
    }

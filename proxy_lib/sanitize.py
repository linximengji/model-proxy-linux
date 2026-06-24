"""DeepSeek message sanitization + image embedding. Pure functions, no side effects."""
import re
import os
import base64 as b64

ALLOWED_KEYS = {
    "model", "messages", "max_tokens", "stream",
    "temperature", "top_p", "top_k",
    "frequency_penalty", "presence_penalty",
    "stop", "stop_sequences",
    "tools", "tool_choice", "system", "thinking",
    "response_format",
}

ALLOWED_BLOCK_TYPES = {
    "text", "tool_use", "tool_result",
    "thinking", "redacted_thinking",
}


def sanitize_for_deepseek(body):
    """Mutates body in-place for DeepSeek compatibility. Returns body for chaining."""
    removed = [k for k in body if k not in ALLOWED_KEYS]
    for k in removed:
        del body[k]

    sys_field = body.get("system")
    if isinstance(sys_field, list):
        parts = [b.get("text", "") if isinstance(b, dict) else str(b) for b in sys_field]
        body["system"] = "\n\n".join(p for p in parts if p)

    messages = body.get("messages")
    if not messages:
        return body

    sys_parts = []
    bs = body.get("system")
    if isinstance(bs, str) and bs.strip():
        sys_parts.append(bs)

    cleaned_msgs = []
    for msg in messages:
        if msg.get("role") == "system":
            c = msg.get("content", "")
            if isinstance(c, list):
                c = "\n\n".join(b.get("text", "") for b in c if isinstance(b, dict) and b.get("type") == "text")
            if c.strip():
                sys_parts.append(c)
            continue
        msg.pop("cache_control", None)
        content = msg.get("content")
        if isinstance(content, list):
            cleaned = []
            for b in content:
                if not isinstance(b, dict):
                    continue
                if b.get("type") not in ALLOWED_BLOCK_TYPES:
                    continue
                b.pop("cache_control", None)
                if b.get("type") == "tool_result" and isinstance(b.get("content"), list):
                    b["content"] = _filter_content_blocks(b["content"])
                cleaned.append(b)
            msg["content"] = cleaned if cleaned else [{"type": "text", "text": ""}]
        cleaned_msgs.append(msg)

    body["messages"] = cleaned_msgs
    if sys_parts:
        body["system"] = "\n\n".join(sys_parts)
    return body


def _filter_content_blocks(blocks):
    cleaned = []
    for b in blocks:
        if not isinstance(b, dict):
            continue
        if b.get("type") in ALLOWED_BLOCK_TYPES:
            b.pop("cache_control", None)
            cleaned.append(b)
    return cleaned if cleaned else [{"type": "text", "text": ""}]


def _strip_thinking_from_blocks(blocks):
    """递归清理 content 数组中的 thinking 块，包括 tool_result 嵌套 content。"""
    cleaned = []
    for b in blocks:
        if not isinstance(b, dict):
            continue
        if b.get("type") in ("thinking", "redacted_thinking"):
            continue
        if b.get("type") == "tool_result" and isinstance(b.get("content"), list):
            b = dict(b)
            b["content"] = _strip_thinking_from_blocks(b["content"])
        cleaned.append(b)
    return cleaned


def strip_thinking_blocks(body):
    """移除 messages 中的 thinking/redacted_thinking 块。qwen MAAS / Anthropic 不支持。"""
    for msg in body.get("messages", []):
        content = msg.get("content")
        if isinstance(content, list):
            msg["content"] = _strip_thinking_from_blocks(content)
    return body


def strip_redacted_thinking_only(body):
    """只移除 redacted_thinking 块，保留 thinking 块。DeepSeek 需要 thinking 块回传。"""
    for msg in body.get("messages", []):
        content = msg.get("content")
        if isinstance(content, list):
            msg["content"] = [
                b for b in content
                if not isinstance(b, dict) or b.get("type") != "redacted_thinking"
            ]
    return body


def embed_images(body, verbose=False):
    """Inline image references ([图片: path]) as base64 blocks. Mutates in-place."""
    for msg in body.get("messages", []):
        if msg.get("role") not in ("user",):
            continue
        content = msg.get("content", "")
        if isinstance(content, str):
            if "[图片: " in content:
                blocks = _text_to_blocks(content, verbose)
                if blocks:
                    msg["content"] = blocks
        elif isinstance(content, list):
            new_content = []
            changed = False
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text" and "[图片: " in block.get("text", ""):
                    text_blocks = _text_to_blocks(block["text"], verbose)
                    new_content.extend(text_blocks)
                    changed = True
                else:
                    new_content.append(block)
            if changed:
                msg["content"] = new_content
    return body


def _is_valid_image_path(path):
    if not path or not os.path.isfile(path):
        return False
    name = os.path.basename(path).lower()
    if name in ("path", "...", "xxx.jpg", "xxx.png", "下载失败", "test.jpg", "example.jpg"):
        return False
    return True


def _text_to_blocks(text, verbose=False):
    parts = re.split(r'\[图片:\s*([^\]]+)\]', text)
    blocks = []
    for i, part in enumerate(parts):
        if i % 2 == 0:
            cleaned = part.strip().strip('\n\r')
            if cleaned:
                blocks.append({"type": "text", "text": cleaned})
        else:
            img_path = part.strip()
            if not _is_valid_image_path(img_path):
                blocks.append({"type": "text", "text": f"[图片: {img_path}]"})
                continue
            try:
                with open(img_path, "rb") as f:
                    raw = f.read()
                ext = os.path.splitext(img_path)[1].lower()
                mime = {".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                        ".png": "image/png", ".gif": "image/gif",
                        ".webp": "image/webp"}.get(ext, "image/jpeg")
                data = b64.b64encode(raw).decode("utf-8")
                blocks.append({
                    "type": "image",
                    "source": {"type": "base64", "media_type": mime, "data": data},
                })
                if verbose:
                    print(f"[sanitize] Embedded image: {img_path} ({len(data)} bytes base64)")
            except Exception as e:
                if verbose:
                    print(f"[sanitize] ERR embedding {img_path}: {e}")
                blocks.append({"type": "text", "text": f"[图片加载失败: {img_path}]"})
    return blocks

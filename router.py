"""Router — L1 fast rules + L2 flash classifier for smart model selection."""
import re
import copy

# ── L1 rules ──────────────────────────────────────────────────────────────
# First-match wins. Order matters: more specific / cheaper rules first.

# ── Module version marker for debug ──
_VERSION = "v5-tiers"

# TIERS dict injected by model_proxy on startup (fallback defaults here).
TIERS: dict[str, str] = {
    "flash": "deepseek-v4-flash",
    "pro": "deepseek-v4-pro",
    "max": "qwen3.7-max",
    "vision": "doubao-1.5-vision-pro",
}


def _has_image(messages):
    """Check if the LAST user message contains images that need vision routing.

    Only inspects the most recent user message — scanning full history would
    misroute every subsequent request once an image ever appeared.
    """
    for msg in reversed(messages):
        if msg.get("role") != "user":
            continue
        content = msg.get("content", "")
        if not isinstance(content, list):
            break
        for block in content:
            if not isinstance(block, dict):
                continue
            bt = block.get("type")
            if bt in ("image", "image_url"):
                return True
            if bt == "tool_result" and isinstance(block.get("content"), list):
                for inner in block["content"]:
                    if isinstance(inner, dict) and inner.get("type") in ("image", "image_url"):
                        return True
        break
    return False


def _has_recent_tools(messages, window=5):
    """Check if tools were used in the last N messages (not full history)."""
    recent = messages[-window:] if len(messages) > window else messages
    for msg in recent:
        content = msg.get("content", "")
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, dict) and block.get("type") in ("tool_use", "tool_result"):
                return True
    return False


def _last_user_text(messages):
    """Extract plain text from the last user message."""
    for msg in reversed(messages):
        if msg.get("role") != "user":
            continue
        content = msg.get("content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for b in content:
                if isinstance(b, dict) and b.get("type") == "text":
                    parts.append(b["text"])
            return "\n".join(parts)
        return ""
    return ""


_GREETING_PATTERNS = re.compile(
    r"^(ok|okay|thanks|thank you|yes|no|sure|got it|right|good|fine|great|"
    r"cool|nice|bye|hi|hey|hello|"
    r"好的|谢谢|是的|对|嗯|好|行|可以|没问题|"
    r"继续|下一个|然后呢|还有吗|"
    r"that works|looks good|please continue|go ahead|"
    r"明白了|知道了|了解了|收到"
    r")[!！。.]*$",
    re.IGNORECASE,
)


def _is_greeting_or_ack(text):
    """Match confirmations, greetings, simple follow-ups."""
    t = text.strip().rstrip("!！。.… ")
    if not t:
        return True
    if len(t) <= 3 and not any(c not in "!！。.… ,，、；;？?" for c in t if c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"):
        return True
    if _GREETING_PATTERNS.match(t):
        return True
    if "\n" not in t and "?" not in t and "？" not in t and len(t) < 30 and not _has_code_indicators(t):
        return True
    return False


def _has_code_indicators(text):
    """Quick check for code/technical content in short text."""
    indicators = [
        r"```", r"`[^`]+`", r"def\s+\w+\s*\(", r"function\s+\w+\s*\(",
        r"import\s+\w+", r"from\s+\w+\s+import", r"const\s+\w+\s*=",
        r"let\s+\w+\s*=", r"var\s+\w+\s*=", r"class\s+\w+",
        r"\bgit\s+(push|pull|commit|merge|rebase)\b",
        r"\b(npm|pip|cargo|yarn|bun)\s+(install|run|build)",
        r"https?://", r"\.py\b", r"\.ts\b", r"\.js\b", r"\.rs\b",
    ]
    for pat in indicators:
        if re.search(pat, text):
            return True
    return False


def _all_text(messages):
    """Extract all text content from messages."""
    texts = []
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            texts.append(content)
        elif isinstance(content, list):
            for b in content:
                if isinstance(b, dict) and b.get("type") == "text":
                    texts.append(b["text"])
    return "\n".join(texts)


def estimate_tokens(text):
    """Rough token estimate. 1 token ≈ 4 ASCII chars or ~1.5 CJK chars."""
    if not text:
        return 0
    cjk = sum(1 for c in text if '一' <= c <= '鿿')
    ascii_chars = len(text) - cjk
    return int(ascii_chars / 4 + cjk / 1.5)


def _last_user_tokens(messages):
    return estimate_tokens(_last_user_text(messages))


def classify(body):
    """L1 fast rules. Returns (model_name, reason) or (None, reason_for_l2).

    New intermediate state: ('ocr-qa', ocr_text) — L2 will judge OCR quality.
    """
    messages = body.get("messages", []) or []

    # Rule 1: image → run OCR, let L2 judge quality
    if _has_image(messages):
        try:
            from proxy_lib.ocr import try_ocr_messages
            # Copy to avoid in-place OCR text blocks polluting original body
            ocr_text = try_ocr_messages(copy.deepcopy(messages))
            if ocr_text:
                return "ocr-qa", ocr_text  # L2 decides: text or vision?
            return TIERS["vision"], "L1:image"  # OCR empty → vision
        except Exception:
            return TIERS["vision"], "L1:image"

    last_text = _last_user_text(messages)
    last_tok = estimate_tokens(last_text)

    # Rule 3: trivial → flash
    if last_tok < 400 and _is_greeting_or_ack(last_text):
        return TIERS["flash"], "L1:trivial"

    # Rule 4: single long input (>8000 tok) → pro
    if last_tok > 8000:
        return TIERS["pro"], "L1:very-long"

    # Sub-agent: use adjusted L2 mapping instead of standard
    if body.get("model") == TIERS["flash"] + "-sub":
        return None, "l2-sub-agent"

    # L2 classifier needed (no L1 rule matched)
    return None, "l2-classifier"

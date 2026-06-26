"""Model proxy v3 — FastAPI app backed by proxy_lib modules."""
import json
import os
import sys
import re
import time
import uuid
import signal
from urllib.parse import urlparse

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(__file__))

# OpenTelemetry — initialized in main()

from proxy_lib import config, sanitize, convert, telemetry, fallback
from proxy_lib.allocator import MultiModelAllocator
from proxy_lib.handlers import (
    handle_anthropic, handle_anthropic_stream,
    handle_openai, handle_openai_stream,
)

START_TIME = time.time()

# ── Model tiers ─────────────────────────────────────────────────────────────
# Maps tier key → actual model name (from .env TIER_* vars, fallback defaults).
# _init_tiers() called from main() after load_dotenv() overrides from env.
_TIERS: dict[str, str] = {
    "flash": "deepseek-v4-flash",
    "pro": "deepseek-v4-pro",
    "max": "qwen3.7-max",
    "vision": "doubao-1.5-vision-pro",
}

def _init_tiers():
    for k in _TIERS:
        v = os.environ.get(f"TIER_{k.upper()}")
        if v:
            _TIERS[k] = v

# L2 Classifier constants (values are tier keys, resolved via _TIERS at use time)
_CLASSIFIER_ROUTE = {
    "trivial": "flash",
    "simple": "flash",
    "moderate": "pro",
    "complex": "max",
}

_SUB_AGENT_CLASSIFIER_ROUTE = {
    "trivial": "flash",
    "simple": "flash",
    "moderate": "pro",     # sub-agent moderate → pro
    "complex": "pro",      # sub-agent complex → pro (不碰 qwen 额度)
}

CLASSIFIER_SYSTEM_PROMPT = """Classify the user message along three dimensions.

1. Complexity: trivial, simple, moderate, or complex
   - trivial: greetings, acknowledgments, one-word responses
   - simple: straightforward questions, single-step tasks
   - moderate: multi-step tasks, code generation, debugging
   - complex: architecture design, complex refactoring, multi-file changes

2. Task type: code, creative, reasoning, long_context, or general
   - code: code generation, debugging, refactoring, API design
   - creative: writing, translation, rewriting, summarization, copywriting
   - reasoning: architecture planning, multi-step deduction, trade-off analysis
   - long_context: long document analysis (history >12K tokens)
   - general: anything not matching the above

3. Token budget estimate: low, medium, or high
   - low: short response expected (<2K tokens)
   - medium: moderate response expected (2-8K tokens)
   - high: long response expected (>8K tokens)

{budget_context}
Reply with exactly three words: COMPLEXITY TASK_TYPE BUDGET
Example: "moderate code medium" or "complex reasoning high" """

http_client: httpx.AsyncClient | None = None
ROUTES: dict = {}
ALLOCATOR = MultiModelAllocator()  # Token Plan 多模型分配器


def reload_cfg():
    global ROUTES
    try:
        import importlib, router
        importlib.reload(router)
        router.TIERS = _TIERS
        ROUTES = config.load_routes()
        telemetry.route_health.clear()
        telemetry.log("Config reloaded", phase="SYSTEM")
        return True, f"{len(ROUTES)} routes loaded"
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        telemetry.log(f"Config reload FAILED: {tb}", "ERROR", "SYSTEM")
        return False, str(e)


# ── L2 Classifier ──────────────────────────────────────────────────────────

def _classify_via_flash(user_query, budget_ctx=None, timeout=2.0):
    route = ROUTES.get(_TIERS["flash"])
    if not route:
        return None
    prompt = CLASSIFIER_SYSTEM_PROMPT.format(
        budget_context=budget_ctx or "No budget constraints — route purely by complexity."
    )
    try:
        req_body = {
            "model": route["model"],
            "system": prompt,
            "messages": [{"role": "user", "content": user_query}],
            "max_tokens": 100,
            "temperature": 0,
            "stream": False,
            "thinking": {"type": "disabled"},
        }
        return http_client.post(
            route["api_base"],
            json=req_body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {route['api_key']}",
            },
            timeout=timeout,
        )
    except Exception as e:
        telemetry.log(f"_classify_via_flash request: {type(e).__name__}: {e}", "ERROR", "L2")
        return None


async def _resolve_classifier(resp_future):
    try:
        resp = await resp_future
        if resp is None or resp.status_code != 200:
            return None, None, None
        data = resp.json()
        content = ""
        for b in data.get("content", []):
            if isinstance(b, dict) and b.get("type") == "text":
                content += b.get("text", "")
        words = content.strip().lower().split()
        valid_complexity = {"trivial", "simple", "moderate", "complex"}
        valid_task_type = {"code", "creative", "reasoning", "long_context", "general"}
        valid_budget = {"low", "medium", "high"}
        complexity = words[0].rstrip(".,;:!?") if len(words) >= 1 else ""
        task_type = words[1].rstrip(".,;:!?") if len(words) >= 2 else "general"
        budget_est = words[2].rstrip(".,;:!?") if len(words) >= 3 else ""
        complexity = complexity if complexity in valid_complexity else None
        task_type = task_type if task_type in valid_task_type else "general"
        budget_est = budget_est if budget_est in valid_budget else None
        return complexity, task_type, budget_est
    except Exception as e:
        telemetry.log(f"_resolve_classifier: {type(e).__name__}: {e}", "ERROR", "L2")
        return None, None, None


# ── Budget-aware routing adjustment ─────────────────────────────────────────

def _build_budget_context(policy):
    """计算 moderate→qwen3.7-max 的连续分配比例。

    自算真实余额（不从 policy 中读 stale credits_remaining），60s 缓存。
    并在每次计算后回写 routing_policy.json 使 dashboard 同步。
    """
    remaining, credits_total, days = telemetry.get_real_credits()
    ratio = ALLOCATOR.compute_ratio(remaining, credits_total, days)

    telemetry.log(
        f"real-time: rem={remaining:.0f} ({remaining/credits_total*100:.0f}%) "
        f"ratio={ratio:.2f}",
        phase="BUDGET"
    )

    # 回写正确值，dashboard 能看到同步数据
    telemetry.write_routing_policy(remaining, credits_total, days)

    if days <= 0:
        ctx = "Token Plan is expired — 0% moderate/complex to TP models."
    elif ratio >= 0.8:
        ctx = (f"Token Plan: est {remaining:.0f}/{credits_total} credits "
               f"({remaining/credits_total*100:.0f}%). Burn — {ratio*100:.0f}% TP routing.")
    elif ratio <= 0.2:
        ctx = (f"Token Plan: est {remaining:.0f}/{credits_total} credits "
               f"({remaining/credits_total*100:.0f}%). Conserve — {ratio*100:.0f}% TP routing.")
    else:
        ctx = (f"Token Plan: est {remaining:.0f}/{credits_total} credits "
               f"({remaining/credits_total*100:.0f}%). ratio={ratio:.2f}.")

    return ctx, ratio


def _allocator_select(complexity, task_type, req_id, ratio):
    """Multi-model allocator select. Returns TP model name or None."""
    return ALLOCATOR.select(complexity, task_type, req_id, ratio)


# ── Prompt-level @model routing ──────────────────────────────────────────

_PROMPT_MODEL_RE = re.compile(r'(?:^|\s)@(\S+)\s*$')


def _get_alias(tag):
    """Resolve @tag alias -> model name."""
    if tag in _TIERS:
        return _TIERS[tag]
    static = {
        "plus": "qwen3.7-plus",
        "coder": "kimi-k2.7-code",
        "kimi": "kimi-k2.6",
        "glm": "glm-5.2",
        "minimax": "MiniMax-M2.5",
        "cheap": "qwen3.6-flash",
        "ds": "qwen3.7-max-ds",
        "qwen": _TIERS["max"],
        "deepseek": _TIERS["pro"],
        "doubao": _TIERS["vision"],
    }
    return static.get(tag)


def _fuzzy_resolve_model(tag):
    """@tag -> ROUTES key。优先 exact，再 alias，再前缀唯一匹配。"""
    if tag in ROUTES:
        return tag
    resolved = _get_alias(tag)
    if resolved:
        return resolved
    candidates = [k for k in ROUTES if k.startswith(tag)]
    if len(candidates) == 1:
        return candidates[0]
    return None


def _resolve_prompt_model(body):
    """Scan last user message for trailing @tag, strip it and bypass if recognized."""
    model_name = None
    for msg in reversed(body.get("messages", []) or []):
        if msg.get("role") != "user":
            continue
        content = msg.get("content", "")
        if isinstance(content, str):
            m = _PROMPT_MODEL_RE.search(content)
            if m:
                tag = m.group(1)
                resolved = _fuzzy_resolve_model(tag)
                if resolved:
                    model_name = resolved
                    msg["content"] = _PROMPT_MODEL_RE.sub("", content).rstrip()
                    break
        elif isinstance(content, list):
            for block in reversed(content):
                if isinstance(block, dict) and block.get("type") == "text":
                    text = block.get("text", "")
                    m = _PROMPT_MODEL_RE.search(text)
                    if m:
                        tag = m.group(1)
                        resolved = _fuzzy_resolve_model(tag)
                        if resolved:
                            model_name = resolved
                            block["text"] = _PROMPT_MODEL_RE.sub("", text).rstrip()
                            break
            if model_name:
                break

    if model_name:
        body["model"] = model_name
        route = ROUTES[model_name]
        if route.get("provider") == "deepseek":
            if model_name == _TIERS["flash"] and "thinking" not in body:
                body["thinking"] = {"type": "disabled"}
            sanitize.sanitize_for_deepseek(body)
        elif route.get("provider") == "anthropic":
            sanitize.strip_thinking_blocks(body)
        else:
            sanitize.strip_thinking_blocks(body)
        telemetry.log(f"BYPASS @model: {model_name}", phase="ROUTE")
        return route, model_name, "prompt-bypass"
    return None, None, None


# ── Routing ────────────────────────────────────────────────────────────────

def _resolve_bypass(body, headers):
    """X-Proxy-Model header overrides routing. Returns (route, model, reason) or (None, None, None)."""
    explicit = (headers.get("x-proxy-model", "") or "").strip()
    if explicit and explicit in ROUTES:
        body["model"] = explicit
        route = ROUTES[explicit]
        if route.get("provider") == "deepseek":
            if explicit == _TIERS["flash"] and "thinking" not in body:
                body["thinking"] = {"type": "disabled"}
            sanitize.sanitize_for_deepseek(body)
        elif route.get("provider") == "anthropic":
            sanitize.strip_thinking_blocks(body)
        else:
            sanitize.strip_thinking_blocks(body)
        telemetry.log(f"BYPASS X-Proxy-Model: {explicit}", phase="ROUTE")
        return route, explicit, "header-bypass"
    return None, None, None


def _strip_images_from_body(body):
    """Remove image/image_url blocks from user messages, keep text blocks."""
    for msg in body.get("messages", []):
        content = msg.get("content", "")
        if not isinstance(content, list):
            continue
        msg["content"] = [b for b in content
                          if not isinstance(b, dict) or b.get("type") not in ("image", "image_url")]


def _append_text_to_last_user(body, text):
    """Append a text block to the last user message."""
    for msg in reversed(body.get("messages", [])):
        if msg.get("role") != "user":
            continue
        content = msg.get("content", "")
        if not isinstance(content, list):
            msg["content"] = [{"type": "text", "text": str(content)}]
            msg["content"].append({"type": "text", "text": text})
        else:
            content.append({"type": "text", "text": text})
        break


async def _resolve_l2(body, l2_future, ratio, is_sub_agent=False):
    """Resolve L2 classifier output → route + sanitization."""
    complexity, task_type, budget_est = await _resolve_classifier(l2_future)
    if complexity is None:
        complexity = "moderate"
    if task_type is None:
        task_type = "general"
    route_map = _SUB_AGENT_CLASSIFIER_ROUTE if is_sub_agent else _CLASSIFIER_ROUTE
    tier_key = route_map.get(complexity, "pro")
    model_name = _TIERS.get(tier_key, _TIERS["pro"])
    tag = "L2-sub" if is_sub_agent else "L2"
    # sub-agent: no allocator — always use mapped model
    if is_sub_agent:
        telemetry.log(f"{tag}: {complexity}:{task_type} + {budget_est or '?'} -> {model_name}", phase="L2")
    else:
        adjusted = _allocator_select(complexity, task_type, telemetry.get_req_id(), ratio)
        if adjusted:
            telemetry.log(
                f"{tag}: {complexity}:{task_type} + {budget_est or '?'} -> {model_name}, allocator -> {adjusted} (ratio={ratio:.2f})",
                phase="L2"
            )
            model_name = adjusted
        else:
            telemetry.log(f"{tag}: {complexity}:{task_type} + {budget_est or '?'} -> {model_name} (ratio={ratio:.2f})",
                         phase="L2")
    body["model"] = model_name
    route = ROUTES.get(model_name)
    if route and route.get("provider") == "deepseek":
        if model_name == _TIERS["flash"] and "thinking" not in body:
            body["thinking"] = {"type": "disabled"}
        sanitize.sanitize_for_deepseek(body)
        sanitize.strip_redacted_thinking_only(body)
    elif route and route.get("provider") == "anthropic":
        sanitize.strip_thinking_blocks(body)
    else:
        sanitize.strip_thinking_blocks(body)
    return route, model_name


_OCR_JUDGE_PROMPT = """\
You evaluate whether OCR text extracted from an image is sufficient to answer the user's query.

User query: "{query}"

OCR text from image: "{ocr_text}"

Does the OCR text adequately answer the user, or is the image's visual content (layout, colors, charts, formatting, non-text elements) essential?

Reply with exactly one word: use_ocr or use_vision
- use_ocr: OCR text is sufficient — strip the image and use only text
- use_vision: Image has critical visual information OCR missed (or OCR text is garbage) — keep the image"""


def _judge_ocr_quality(user_query: str, ocr_text: str, timeout=2.0) -> str:
    """Ask flash whether OCR text is good enough. Returns 'use_ocr' or 'use_vision'."""
    route = ROUTES.get(_TIERS["flash"])
    if not route:
        return "use_vision"
    prompt = _OCR_JUDGE_PROMPT.format(query=user_query[:500], ocr_text=ocr_text[:1500])
    try:
        # Use sync client to avoid async dance in sync _route_and_sanitize
        import httpx as _httpx
        with _httpx.Client(timeout=timeout) as sync_client:
            resp = sync_client.post(
                route["api_base"],
                json={
                    "model": route["model"],
                    "system": "Reply with exactly one word: use_ocr or use_vision.",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 10,
                    "temperature": 0,
                    "stream": False,
                    "thinking": {"type": "disabled"},
                },
                headers={"Content-Type": "application/json",
                         "Authorization": f"Bearer {route['api_key']}"},
            )
        if resp.status_code != 200:
            return "use_vision"
        data = resp.json()
        content = ""
        for b in data.get("content", []):
            if isinstance(b, dict) and b.get("type") == "text":
                content += b.get("text", "")
        verdict = content.strip().lower().rstrip(".,;:!?")
        return verdict if verdict in ("use_ocr", "use_vision") else "use_vision"
    except Exception as e:
        telemetry.log(f"_judge_ocr_quality: {type(e).__name__}", "ERROR", "OCR")
        return "use_vision"


def _route_and_sanitize(body):
    from router import classify
    model_name = body.get("model", "")

    # Agent .md 已明确指定模型 → 直达，不走 L1/L2
    if model_name in ROUTES:
        route = ROUTES[model_name]
        if route.get("provider") == "deepseek":
            if model_name == _TIERS["flash"] and "thinking" not in body:
                body["thinking"] = {"type": "disabled"}
            sanitize.sanitize_for_deepseek(body)
            sanitize.strip_redacted_thinking_only(body)
        elif route.get("provider") == "anthropic":
            sanitize.strip_thinking_blocks(body)
        else:
            sanitize.strip_thinking_blocks(body)
        telemetry.log(f"BYPASS agent-model: {model_name}", phase="ROUTE")
        return route, model_name, "agent-model", None, None, False

    try:
        routed_model, reason = classify(body)
    except Exception:
        routed_model = _TIERS["pro"]
        reason = "classify-error"

    # ── OCR quality judgment ──────────────────────────────────────────────
    if routed_model == "ocr-qa":
        ocr_text = reason
        telemetry.log(f"OCR-qa phase, text length={len(ocr_text)}", phase="OCR")
        # Extract user query for context
        user_query = ""
        messages = body.get("messages", []) or []
        for msg in reversed(messages):
            if msg.get("role") == "user":
                c = msg.get("content", "")
                if isinstance(c, str):
                    user_query = c
                elif isinstance(c, list):
                    user_query = " ".join(b.get("text", "") for b in c
                                          if isinstance(b, dict) and b.get("type") == "text")
                break
        if len(user_query) > 500:
            user_query = user_query[:500]

        verdict = _judge_ocr_quality(user_query, ocr_text)
        telemetry.log(f"OCR quality verdict: {verdict}", phase="OCR")

        if verdict == "use_ocr":
            # Strip images, add OCR text context, fall through to normal text routing
            _strip_images_from_body(body)
            _append_text_to_last_user(body, f"[OCR extracted from image]\n{ocr_text}")
            telemetry.log("OCR accepted — stripping images, routing as text", phase="ROUTE")
            # Re-route as text-only (this was the result when classify returned ocr-qa,
            # but now images are gone, so re-running classify won't hit ocr-qa again)
            try:
                routed_model, reason = classify(body)
            except Exception:
                routed_model = _TIERS["pro"]
                reason = "classify-error"
            # If still ocr-qa (edge case), fall back to vision
            if routed_model == "ocr-qa":
                routed_model = _TIERS["vision"]
                reason = "L1:ocr-fallback-qa-loop"
        else:
            routed_model = _TIERS["vision"]
            reason = "L1:ocr-rejected"

    # ── Normal L1 routing below ──────────────────────────────────────────
    if routed_model is None:
        user_query = ""
        messages = body.get("messages", []) or []
        for msg in reversed(messages):
            if msg.get("role") == "user":
                c = msg.get("content", "")
                if isinstance(c, str):
                    user_query = c
                elif isinstance(c, list):
                    user_query = " ".join(b.get("text", "") for b in c
                                          if isinstance(b, dict) and b.get("type") == "text")
                break
        if len(user_query) > 2000:
            user_query = user_query[:2000]

        policy = telemetry.load_budget_policy()
        if policy:
            budget_ctx, ratio = _build_budget_context(policy)
        else:
            remaining, credits_total, days = telemetry.get_real_credits()
            ratio = ALLOCATOR.compute_ratio(remaining, credits_total, days)
            budget_ctx = (f"Token Plan: {remaining:.0f}/{credits_total} credits "
                          f"({remaining/credits_total*100:.0f}%). ratio={ratio:.2f}.")
            telemetry.log(
                f"fallback: rem={remaining:.0f}/{credits_total} ratio={ratio:.2f}",
                phase="BUDGET"
            )

        l2_future = _classify_via_flash(user_query, budget_ctx=budget_ctx) if user_query else None
        preview = user_query[:80] if user_query else ""
        is_sub = reason == "l2-sub-agent"
        telemetry.log(f"L2 classify: \"{preview}{'...' if len(user_query)>80 else ''}\" ratio={ratio:.2f}", phase="L2")
        return None, None, "l2-pending", l2_future, ratio, is_sub

    telemetry.log(f"L1 {reason}: {model_name or 'auto'} -> {routed_model}", phase="ROUTE")
    body["model"] = routed_model
    route = ROUTES.get(routed_model) or ROUTES.get(re.sub(r'\[.*\]', '', routed_model))
    if route and route.get("provider") == "deepseek":
        if routed_model == _TIERS["flash"] and "thinking" not in body:
            body["thinking"] = {"type": "disabled"}
        sanitize.sanitize_for_deepseek(body)
        sanitize.strip_redacted_thinking_only(body)
    elif route and route.get("provider") == "anthropic":
        sanitize.strip_thinking_blocks(body)
    else:
        sanitize.strip_thinking_blocks(body)
    return route, routed_model, reason, None, None, False


# ── Model tier ladder for stuck escalation ──────────────────────────────────
_TIER_LADDER = ["flash", "pro", "max"]


def _resolve_tier_key(model_name):
    """Map a model name (e.g. 'deepseek-v4-flash') back to its tier key."""
    rev = {v: k for k, v in _TIERS.items()}
    return rev.get(model_name, "max")


def _upgrade_tier(current_tier_key):
    """Move up one tier. 'max' and 'vision' stay 'max'."""
    if current_tier_key == "vision":
        return "max"
    try:
        idx = _TIER_LADDER.index(current_tier_key)
        return _TIER_LADDER[min(idx + 1, len(_TIER_LADDER) - 1)]
    except ValueError:
        return "max"


def _resanitize_for_upgrade(body, new_route, old_route):
    """Re-apply sanitization if provider changed after model upgrade."""
    new_prov = new_route.get("provider") if new_route else None
    old_prov = old_route.get("provider") if old_route else None
    if new_prov == old_prov or new_prov is None:
        return
    if new_prov == "deepseek":
        sanitize.sanitize_for_deepseek(body)
        sanitize.strip_redacted_thinking_only(body)
    elif new_prov == "anthropic":
        sanitize.strip_thinking_blocks(body)


def _inject_escalate(body, route, model_name):
    """Upgrade model and inject escalate prompt when stuck is detected.

    Mutates body in-place. Returns (updated_route, updated_model_name).
    """
    from router import ESCAPE_PROMPT, ESCAPE_PROMPT_INJECTED_MARKER

    # 1. Check if already injected this session
    sys_field = body.get("system", "")
    if isinstance(sys_field, str) and ESCAPE_PROMPT_INJECTED_MARKER in sys_field:
        return route, model_name
    if isinstance(sys_field, list):
        combined = "\n".join(
            b.get("text", "") if isinstance(b, dict) else str(b)
            for b in sys_field
        )
        if ESCAPE_PROMPT_INJECTED_MARKER in combined:
            return route, model_name

    # 2. Upgrade model tier
    old_model = model_name
    tier_key = _resolve_tier_key(model_name)
    upgraded_key = _upgrade_tier(tier_key)
    new_model = _TIERS.get(upgraded_key)
    new_route = ROUTES.get(new_model) if new_model else None
    if new_route and new_model != model_name:
        body["model"] = new_model
        _resanitize_for_upgrade(body, new_route, route)
        route, model_name = new_route, new_model

    # 3. Inject prompt
    sep = "\n\n" if sys_field else ""
    body["system"] = f"{sys_field}{sep}{ESCAPE_PROMPT_INJECTED_MARKER}\n{ESCAPE_PROMPT}"

    telemetry.log(
        f"ESCALATE: {old_model} -> {model_name}, prompt injected",
        phase="ESCALATE"
    )
    return route, model_name


def _maybe_escalate(body, route, model_name):
    """Call detect_stuck and escalate if needed. Returns (route, model_name)."""
    if route is None:
        return route, model_name

    # Don't escalate requests with images — upgrade target would lack vision capability,
    # causing 400 errors from the upstream API and making the session permanently stuck.
    try:
        from router import _has_image
        if _has_image(body.get("messages", [])):
            return route, model_name
    except Exception:
        pass

    try:
        from router import detect_stuck
        stuck_info = detect_stuck(body.get("messages", []))
        if stuck_info is None:
            return route, model_name
        telemetry.log(
            f"STUCK detected: {stuck_info['rounds']} rounds, "
            f"{stuck_info['error_count']} errors "
            f"({stuck_info['error_pct']:.0%})",
            phase="ESCALATE"
        )
        return _inject_escalate(body, route, model_name)
    except Exception as e:
        telemetry.log(f"Escalate detection error: {e}", "ERROR", "ESCALATE")
        return route, model_name


# ── FastAPI app ────────────────────────────────────────────────────────────

app = FastAPI(openapi_url=None, docs_url=None, redoc_url=None)


@app.on_event("startup")
async def startup():
    global http_client
    http_client = httpx.AsyncClient(timeout=httpx.Timeout(300.0))


@app.on_event("shutdown")
async def shutdown():
    if http_client:
        await http_client.aclose()


@app.get("/health")
async def health():
    return JSONResponse({"status": "ok"})


@app.get("/v1/models")
async def list_models():
    models = [{"id": n, "object": "model", "created": 1, "owned_by": "proxy"} for n in ROUTES]
    return JSONResponse({"object": "list", "data": models})


@app.get("/v1/stats")
async def get_stats():
    return JSONResponse(telemetry.build_stats())


@app.get("/v1/rules")
async def get_rules():
    try:
        from router import RULES
        return JSONResponse({"rules": RULES, "count": len(RULES)})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/v1/reload")
async def reload_endpoint():
    ok, msg = reload_cfg()
    return JSONResponse({"status": "ok" if ok else "error", "message": msg},
                        status_code=200 if ok else 500)


@app.post("/v1/restart")
async def restart_self():
    """Graceful self-restart. Sends SIGTERM to self; ops-daemon auto-restores."""
    telemetry.log("Restart requested via /v1/restart — exiting", phase="SYSTEM")
    import threading
    def _die():
        time.sleep(0.5)
        os.kill(os.getpid(), signal.SIGTERM)
    threading.Thread(target=_die, daemon=True).start()
    return JSONResponse({"status": "restarting"})


@app.post("/v1/rules/debug")
async def rules_debug(request: Request):
    body = await request.json()
    messages = body.get("messages", []) or []
    from router import classify, _all_text, _has_image, _has_recent_tools, \
        _last_user_text, _is_greeting_or_ack, estimate_tokens
    text = _all_text(messages)
    total_tok = estimate_tokens(text)
    last_text = _last_user_text(messages)
    last_tok = estimate_tokens(last_text)
    routed, reason = classify(body)
    return JSONResponse({
        "route_to": routed,
        "reason": reason,
        "analysis": {
            "total_tokens": total_tok,
            "last_user_tokens": last_tok,
            "has_image": _has_image(messages),
            "has_recent_tools": _has_recent_tools(messages),
            "is_trivial": last_tok < 400 and _is_greeting_or_ack(last_text),
            "is_very_long": total_tok > 15000,
            "last_user_text_preview": last_text[:120],
        },
    })


@app.post("/v1/messages")
async def proxy_anthropic(request: Request):
    telemetry.set_req_id(uuid.uuid4().hex[:8])
    body = await request.json()
    path = urlparse(str(request.url)).path
    if path in ("/v1/messages", "/v1/chat/completions"):
        sanitize.embed_images(body)

    route, model_name, _reason = _resolve_prompt_model(body)
    route, model_name = _maybe_escalate(body, route, model_name)

    if not route:
        route, model_name, _reason = _resolve_bypass(body, request.headers)
        route, model_name = _maybe_escalate(body, route, model_name)

    if not route:
        route, model_name, _reason, l2_future, ratio, is_sub = _route_and_sanitize(body)

        if l2_future is not None:
            route, model_name = await _resolve_l2(body, l2_future, ratio, is_sub)

        route, model_name = _maybe_escalate(body, route, model_name)

    if not route:
        await telemetry.record_error(model_name or "unknown")
        return JSONResponse({"error": f"unknown model: {model_name}"}, status_code=404)

    is_stream = body.get("stream", False)
    telemetry.log(f"{model_name} /v1/messages{' (stream)' if is_stream else ''}", phase="UPSTREAM")

    work_dir = request.headers.get("x-claude-work-dir", "")
    session_id = request.headers.get("x-claude-session-id", "")

    await telemetry.record_request(model_name, _reason)

    _t0 = time.time()
    try:
        if is_stream:
            return await handle_anthropic_stream(body, route, model_name, ROUTES,
                                                  http_client, work_dir, session_id)
        else:
            return await handle_anthropic(body, route, model_name, ROUTES, http_client,
                                          work_dir, session_id)
    finally:
        await telemetry.record_latency(model_name, (time.time() - _t0) * 1000)


@app.post("/v1/chat/completions")
async def proxy_openai(request: Request):
    telemetry.set_req_id(uuid.uuid4().hex[:8])
    body = await request.json()
    sanitize.embed_images(body)

    route, model_name, _reason = _resolve_prompt_model(body)
    route, model_name = _maybe_escalate(body, route, model_name)

    if not route:
        route, model_name, _reason = _resolve_bypass(body, request.headers)
        route, model_name = _maybe_escalate(body, route, model_name)

    if not route:
        route, model_name, _reason, l2_future, ratio, is_sub = _route_and_sanitize(body)

        if l2_future is not None:
            route, model_name = await _resolve_l2(body, l2_future, ratio, is_sub)

        route, model_name = _maybe_escalate(body, route, model_name)

    if not route:
        return JSONResponse({"error": f"unknown model: {model_name}"}, status_code=404)

    is_stream = body.get("stream", False)
    model_name_display = body.get("model", model_name)
    telemetry.log(f"{model_name_display} /v1/chat/completions{' (stream)' if is_stream else ''}", phase="UPSTREAM")

    await telemetry.record_request(model_name_display, _reason)

    _t0 = time.time()
    try:
        if is_stream:
            return await handle_openai_stream(body, route, model_name_display, ROUTES, http_client)
        else:
            return await handle_openai(body, route, model_name_display, ROUTES, http_client)
    finally:
        await telemetry.record_latency(model_name_display, (time.time() - _t0) * 1000)


@app.post("/{path:path}")
async def not_found_post(path: str):
    if path not in ("v1/messages", "v1/chat/completions", "v1/rules/debug"):
        await telemetry.record_error()
        return JSONResponse({"error": f"unsupported path: /{path}"}, status_code=404)
    return JSONResponse({}, status_code=404)


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    # OpenTelemetry init — after routes registered, before uvicorn starts
    os.environ.setdefault("OTEL_SERVICE_NAME", "model-proxy")
    os.environ.setdefault("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor

    _provider = TracerProvider(resource=Resource.create({"service.name": "model-proxy"}))
    _provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(
        endpoint="http://localhost:4317", insecure=True)))
    trace.set_tracer_provider(_provider)
    HTTPXClientInstrumentor().instrument()

    config.load_dotenv()
    _init_tiers()
    global ROUTES
    ROUTES = config.load_routes()
    # Inject TIERS into router module for L1 rules
    from router import TIERS as _rt
    _rt.update(_TIERS)

    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    flags = set(a for a in sys.argv[1:] if a in ("-v", "--verbose"))
    verbose = "-v" in flags or "--verbose" in flags

    log_dir = os.path.dirname(os.path.abspath(__file__))
    token_log = r"D:\ClaudeProjects\.claudetalk\token_usage.jsonl"
    log_file = os.path.join(log_dir, "proxy.log")
    access_log = os.path.join(log_dir, "proxy_access.log")

    telemetry.init(token_log_path=token_log, log_file=log_file, access_log=access_log, verbose=verbose)

    import uvicorn
    port = int(args[0]) if args else 4000

    pid_path = os.path.join(log_dir, "proxy.pid")
    with open(pid_path, "w") as f:
        f.write(str(os.getpid()))

    @app.middleware("http")
    async def access_log_middleware(request: Request, call_next):
        start = time.time()
        response = await call_next(request)
        dt = (time.time() - start) * 1000
        with open(access_log, "a", encoding="utf-8") as af:
            af.write(f"[{time.strftime('%H:%M:%S')}] {request.method} {request.url.path} "
                     f"{response.status_code} {dt:.0f}ms\n")
        return response

    uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")


if __name__ == "__main__":
    main()

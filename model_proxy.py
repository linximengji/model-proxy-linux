"""Model proxy v3 — FastAPI app backed by proxy_lib modules."""
import json
import os
import sys
import re
import time
import hashlib
import uuid
from urllib.parse import urlparse

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(__file__))

# OpenTelemetry — initialized in main()

from proxy_lib import config, sanitize, telemetry
from proxy_lib.allocator import MultiModelAllocator


def _bypass_fail():
    return _active_bypass() is not None
from proxy_lib.handlers import (
    handle_anthropic, handle_anthropic_stream,
    handle_openai, handle_openai_stream,
)

_restart_port = 4000
_restart_args: list = []

# ── Model tiers ─────────────────────────────────────────────────────────────
# Maps tier key → actual model name (from .env TIER_* vars, fallback defaults).
# _init_tiers() called from main() after load_dotenv() overrides from env.
_TIERS: dict[str, str] = {
    "flash": "deepseek-v4-flash",
    "pro": "deepseek-v4-pro",
    "max": "qwen3.8-max",
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
    "moderate": "pro",
    "complex": "pro",
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

# Global bypass — when set, ALL requests skip routing and go to this model.
# Set via env PROXY_BYPASS=1 (uses TIERS["max"]), or PROXY_BYPASS=<model_name>.
_BYPASS_MODEL: str | None = None
_BYPASS_EXPIRY: float | None = None

# Force switch (L2) — 进入 L2 的请求无论复杂度统一走指定模型，取代 force-max。
# 通过 API POST /v1/force-switch 开/关 + 指定模型，持久化到 force_switch.json。
_FORCE_SWITCH: dict = {"enabled": False, "model": None}
_FORCE_SWITCH_TS: float | None = None  # 可选 TTL 到期时间戳（epoch seconds），None=不过期
_FORCE_SWITCH_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "force_switch.json")


def _expire_force_switch():
    """TTL 到期后自动关闭 force-switch。"""
    global _FORCE_SWITCH, _FORCE_SWITCH_TS
    if _FORCE_SWITCH["enabled"] and _FORCE_SWITCH_TS and time.time() >= _FORCE_SWITCH_TS:
        _FORCE_SWITCH = {"enabled": False, "model": None}
        _FORCE_SWITCH_TS = None
        _save_force_switch()
        telemetry.log("FORCE-SWITCH auto-disabled: TTL expired", "INFO", "ROUTE")


def _load_force_switch():
    """启动时从 force_switch.json 恢复 force-switch 状态（持久化）。"""
    global _FORCE_SWITCH, _FORCE_SWITCH_TS
    try:
        if os.path.isfile(_FORCE_SWITCH_PATH):
            with open(_FORCE_SWITCH_PATH, "r", encoding="utf-8") as f:
                d = json.load(f)
            model = d.get("model")
            ts = d.get("ts")
            if model is not None and model not in ROUTES:
                model = None
            _FORCE_SWITCH = {"enabled": bool(d.get("enabled")), "model": model}
            _FORCE_SWITCH_TS = float(ts) if ts else None
            _expire_force_switch()
            if _FORCE_SWITCH["enabled"]:
                telemetry.log(f"FORCE-SWITCH restored: {model or _TIERS['max']}", phase="ROUTE")
    except Exception as e:
        telemetry.log(f"_load_force_switch: {type(e).__name__}: {e}", "WARN", "ROUTE")


def _save_force_switch():
    """把当前 force-switch 状态写回 force_switch.json。"""
    try:
        payload = dict(_FORCE_SWITCH)
        payload["ts"] = _FORCE_SWITCH_TS
        tmp = _FORCE_SWITCH_PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
        os.replace(tmp, _FORCE_SWITCH_PATH)  # 原子替换，避免写盘中断留下损坏 JSON
    except Exception as e:
        telemetry.log(f"_save_force_switch: {type(e).__name__}: {e}", "WARN", "ROUTE")

def _active_bypass():
    global _BYPASS_MODEL, _BYPASS_EXPIRY
    if _BYPASS_EXPIRY and time.monotonic() >= _BYPASS_EXPIRY:
        if _BYPASS_MODEL:
            telemetry.log(f"BYPASS TTL expired: {_BYPASS_MODEL}, routing restored", "INFO", "ROUTE")
        _BYPASS_MODEL = None
        _BYPASS_EXPIRY = None
    return _BYPASS_MODEL


def _resolve_bypass_global():
    global _BYPASS_MODEL, _BYPASS_EXPIRY
    val = os.environ.get("PROXY_BYPASS", "").strip()
    if not val:
        _BYPASS_MODEL = None
        _BYPASS_EXPIRY = None
        return
    if val == "1":
        _BYPASS_MODEL = _TIERS["max"]
    elif val in ROUTES:
        _BYPASS_MODEL = val
    else:
        _BYPASS_MODEL = _TIERS["max"]
    _BYPASS_EXPIRY = None
    telemetry.log(f"GLOBAL BYPASS enabled: {_BYPASS_MODEL}", "INFO", "ROUTE")


def reload_cfg():
    global ROUTES
    try:
        import importlib
        import router
        importlib.reload(router)
        router.TIERS = _TIERS
        ROUTES = config.load_routes()
        load_aliases()
        telemetry.route_health.clear()
        telemetry.log("Config reloaded", phase="SYSTEM")
        return True, f"{len(ROUTES)} routes loaded"
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        telemetry.log(f"Config reload FAILED: {tb}", "ERROR", "SYSTEM")
        return False, str(e)


# ── L2 Classifier ──────────────────────────────────────────────────────────

def _local_fallback_classify(user_query: str):
    """本地启发式 L2 fallback — flash 超时或不可用时使用。

    根据 query 特征判断复杂度，不做远程调用。
    """
    from router import _has_code_indicators, estimate_tokens
    tok = estimate_tokens(user_query)

    # code intent（中文代码意图）：bug/修复/报错/代码/实现 等，即使短文本也归 code，
    # 避免被 trivial/simple 的 general 早退吞掉，导致 kimi 通道不触发
    _code_intent = any(kw in user_query for kw in ("bug", "报错", "报错信息", "错误", "exception", "error", "编译", "运行失败", "报 bug", "这个bug", "那段代码", "函数", "接口对接", "前端代码", "后端代码", "实现一个"))
    if _code_intent:
        return "moderate", "code", "medium"

    # creative（写作/文案/翻译/总结）：需在 trivial/simple 之前判定，
    # 否则短文案会被 general 兜底吞掉，导致 GLM 通道不触发
    if any(kw in user_query for kw in ("文案", "润色", "改写", "翻译", "总结", "摘要", "散文", "创作", "续写", "扩写", "压缩", "提炼要点", "写一首", "写一段", "写一篇")):
        return "moderate", "creative", "medium"

    # reasoning intent（中文推理/方案/权衡）：短方案对比问题也归 reasoning，指向 qwen-max
    if any(kw in user_query for kw in ("设计方案", "架构设计", "方案对比", "架构", "权衡", "技术选型", "方案利弊", "怎么设计", "规划一下", "架构方案", "系统设计")):
        return "complex", "reasoning", "high"

    # trivial: 极短的无代码内容
    if tok < 10 and not _has_code_indicators(user_query) and not _code_intent:
        return "trivial", "general", "low"

    # simple: 短文本，无代码指示符
    if tok < 50 and not _has_code_indicators(user_query):
        return "simple", "general", "low"

    # complex: 超长或明显架构/设计类 query
    if tok > 400 or _has_code_indicators(user_query):
        if any(kw in user_query for kw in ("设计", "架构", "architecture", "refactor", "重构", "方案", "plan", "对比", "compare")):
            return "complex", "reasoning", "high"
        if any(kw in user_query for kw in ("写", "create", "implement", "实现", "代码", "debug", "修")):
            return "moderate", "code", "medium"
        if any(kw in user_query for kw in ("文案", "翻译", "总结", "改写")):
            return "moderate", "creative", "medium"

    # long_context: 超长输入 → 长文档分析，而非直接压到 general
    if tok > 4000:
        return "complex", "long_context", "high"

    # moderate: 默认
    return "moderate", "general", "medium"


def _classify_via_flash(user_query, budget_ctx=None, timeout=8.0):
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
    """计算 moderate→qwen3.8-max 的连续分配比例。

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


# trivial/simple 低档流量的 TP 平替模型：qwen3.6-flash 能力档位匹配 DeepSeek flash，
# 走 TokenPlan 额度，帮 flash 分担流量、消耗临近过期的 TP 余额。
_TP_FLASH_FALLBACK = "qwen3.6-flash"


def _flash_tp_split(req_id: str, ratio: float, label: str = "trivial") -> str:
    """低档请求按 ratio 在 deepseek-v4-flash 与 qwen3.6-flash(TP) 间分流。

    确定性 hash 门：同一 (req_id, label) 路由稳定，不会在两次请求间抖动。
    label 区分 trivial/simple 两条路径，避免两个档位的分流互相关联。
    返回选中模型名；ratio 越低越倾向保留 DeepSeek（省 TP 保底），越高越分流到 TP。
    """
    if ratio <= 0:
        return _TIERS["flash"]
    if ratio >= 1.0:
        return _TP_FLASH_FALLBACK
    h = int(hashlib.md5(f"{req_id}:{label}-tp".encode(), usedforsecurity=False).hexdigest(), 16) % 10000
    if h < ratio * 10000:
        return _TP_FLASH_FALLBACK
    return _TIERS["flash"]


# ── Prompt-level @model routing ──────────────────────────────────────────

_PROMPT_MODEL_RE = re.compile(r'(?:^|\s)@([a-zA-Z0-9_.-]+)')
_STRIP_TAG_RE = re.compile(r'\s*@[a-zA-Z0-9_.-]+')


_ALIASES: dict[str, str] = {}

def load_aliases():
    """从 litellm_config.yaml 的 aliases 段加载 @tag 别名映射。"""
    global _ALIASES
    _ALIASES.clear()
    # 内置 tier 别名（始终有效）
    for k, v in _TIERS.items():
        _ALIASES[k] = v
    # 从 yaml 加载自定义别名
    import yaml
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "litellm_config.yaml")
    try:
        with open(config_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        for entry in (cfg.get("aliases") or []):
            if isinstance(entry, dict):
                tag = entry.get("tag", "").strip()
                model = entry.get("model", "").strip()
                if tag and model:
                    _ALIASES[tag] = model
    except Exception as e:
        telemetry.log(f"load_aliases failed: {e}", "WARN", "ROUTE")

def _get_alias(tag):
    """Resolve @tag alias -> model name. 查 _ALIASES，再回退 ROUTES exact match。"""
    if tag in _ALIASES:
        return _ALIASES[tag]
    if tag in ROUTES:
        return tag
    return None


def _fuzzy_resolve_model(tag):
    """@tag -> ROUTES key。优先 exact，再 alias，再后缀唯一匹配。"""
    if tag in ROUTES:
        return tag
    resolved = _get_alias(tag)
    if resolved:
        return resolved
    candidates = [k for k in ROUTES if k.endswith(tag)]
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
            matches = list(_PROMPT_MODEL_RE.finditer(content))
            if matches:
                tag = matches[-1].group(1)
                resolved = _fuzzy_resolve_model(tag)
                if resolved:
                    model_name = resolved
                    msg["content"] = _STRIP_TAG_RE.sub("", content).strip()
                    break
        elif isinstance(content, list):
            for block in reversed(content):
                if isinstance(block, dict) and block.get("type") == "text":
                    text = block.get("text", "")
                    matches = list(_PROMPT_MODEL_RE.finditer(text))
                    if matches:
                        tag = matches[-1].group(1)
                        resolved = _fuzzy_resolve_model(tag)
                        if resolved:
                            model_name = resolved
                            block["text"] = _STRIP_TAG_RE.sub("", text).strip()
                            break
            if model_name:
                break

    if model_name:
        body["model"] = model_name
        route = ROUTES.get(model_name)
        if not route:
            telemetry.log(f"BYPASS @model: {model_name} not in ROUTES", "INFO", "ROUTE")
            del body["model"]
            return None, None, None
        if route.get("provider") == "deepseek":
            _sanitize_deepseek(body, model_name)
        elif route.get("provider") == "anthropic":
            sanitize.sanitize_for_maas(body)
        else:
            sanitize.strip_thinking_blocks(body)
        telemetry.log(f"BYPASS @model: {model_name}", phase="ROUTE")
        return route, model_name, "prompt-bypass"
    return None, None, None


def _has_thinking_history(body):
    """Check if any assistant message in history has thinking blocks."""
    for msg in body.get("messages", []):
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content", "")
        if not isinstance(content, list):
            continue
        if any(isinstance(b, dict) and b.get("type") == "thinking" for b in content):
            return True
    return False


def _disable_thinking_if_mixed_history(body):
    """Disable thinking when conversation has mixed thinking/non-thinking
    assistant messages. DeepSeek returns 400 'content[].thinking must be passed
    back' when thinking is enabled (explicitly or by default) but some assistant
    messages in history lack thinking blocks."""
    thinking_val = body.get("thinking", {})
    if not isinstance(thinking_val, dict):
        return
    thinking_type = thinking_val.get("type") if isinstance(thinking_val, dict) else None
    thinking_is_explicitly_disabled = thinking_type == "disabled"
    if thinking_is_explicitly_disabled:
        return
    has_thinking = False
    has_non_thinking = False
    for msg in body.get("messages", []):
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content", "")
        if not isinstance(content, list):
            has_non_thinking = True
        else:
            msg_has_thinking = any(
                isinstance(b, dict) and b.get("type") == "thinking"
                for b in content
            )
            if msg_has_thinking:
                has_thinking = True
            else:
                has_non_thinking = True
    if has_thinking and has_non_thinking:
        body["thinking"] = {"type": "disabled"}
        sanitize.strip_thinking_blocks(body)
        telemetry.log("Mixed thinking/non-thinking history — disabled thinking and stripped blocks",
                      "WARN", "SANITIZE")


def _sanitize_deepseek(body, model_name):
    """Apply DeepSeek-specific sanitization based on model type.

    deepseek-chat (flash) doesn't support extended thinking on the Anthropic
    API. Assistant messages with thinking blocks from pro model responses
    cause 400 "thinking must be passed back" errors. Always strip thinking
    blocks for flash.

    deepseek-reasoner (pro) supports extended thinking — only strip
    redacted_thinking blocks and detect mixed history.
    """
    if model_name == _TIERS["flash"]:
        body["thinking"] = {"type": "disabled"}
        sanitize.strip_thinking_blocks(body)
    else:
        _disable_thinking_if_mixed_history(body)
        sanitize.strip_redacted_thinking_only(body)
    sanitize.sanitize_for_deepseek(body)


# ── Routing ────────────────────────────────────────────────────────────────

def _resolve_bypass(body, headers):
    """X-Proxy-Model header overrides routing. Returns (route, model, reason) or (None, None, None)."""
    explicit = (headers.get("x-proxy-model", "") or "").strip()
    if explicit and explicit in ROUTES:
        body["model"] = explicit
        route = ROUTES[explicit]
        if route.get("provider") == "deepseek":
            _sanitize_deepseek(body, explicit)
        elif route.get("provider") == "anthropic":
            sanitize.sanitize_for_maas(body)
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


async def _resolve_l2(body, l2_future, ratio, is_sub_agent=False, user_query=""):
    """Resolve L2 classifier output → route + sanitization.
    flash 超时或不可用时用本地规则 fallback。
    """
    complexity, task_type, budget_est = await _resolve_classifier(l2_future)
    if complexity is None:
        telemetry.log("L2: flash classify failed, using local fallback", phase="L2")
        complexity, task_type, budget_est = _local_fallback_classify(user_query)

    # Force switch — 进入 L2 的请求无论复杂度统一走指定模型（取代 force-max）
    _expire_force_switch()
    if _FORCE_SWITCH["enabled"]:
        model_name = _FORCE_SWITCH["model"] or _TIERS["max"]
        route = ROUTES.get(model_name)
        if route:
            telemetry.log(
                f"L2: force-switch {complexity}:{task_type}{' (sub-agent)' if is_sub_agent else ''} -> {model_name}",
                phase="L2"
            )
            body["model"] = model_name
            if route.get("provider") == "deepseek":
                _sanitize_deepseek(body, model_name)
            elif route.get("provider") == "anthropic":
                sanitize.sanitize_for_maas(body)
            else:
                sanitize.strip_thinking_blocks(body)
            return route, model_name
        # force 的模型不在 ROUTES（配置变更/文件被改）→ 降级走正常 L2 路由
        telemetry.log(f"L2: force-switch {model_name} not in ROUTES, degrading to L2 routing", "WARN", "L2")

    route_map = _SUB_AGENT_CLASSIFIER_ROUTE if is_sub_agent else _CLASSIFIER_ROUTE
    tier_key = route_map.get(complexity, "pro")
    model_name = _TIERS.get(tier_key, _TIERS["pro"])
    tag = "L2-sub" if is_sub_agent else "L2"
    # 主线程与 sub-agent 都由 allocator 统一分配，平等竞争 TP 配额
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
    # simple 分流：allocator 未调整（仍走 flash）时，不抢占 TP 缺额的前提下，
    # 把部分 simple 的 flash 流量按 ratio 交给 qwen3.6-flash(TP)，与 trivial 分流解耦。
    if not adjusted and model_name == _TIERS["flash"] and not is_sub_agent and complexity == "simple":
        try:
            _rem_t, _tot_t, _d_t = telemetry.get_real_credits()
            _r_t = ALLOCATOR.compute_ratio(_rem_t, _tot_t, _d_t)
            split_to = _flash_tp_split(telemetry.get_req_id(), _r_t, label="simple")
            if split_to != _TIERS["flash"] and ROUTES.get(split_to):
                model_name = split_to
                telemetry.log(f"{tag}: simple -> {model_name} (tp-simple-split ratio={_r_t:.2f})", phase="L2")
        except Exception:
            pass
    body["model"] = model_name
    route = ROUTES.get(model_name)
    if route and route.get("provider") == "deepseek":
        _sanitize_deepseek(body, model_name)
    elif route and route.get("provider") == "anthropic":
        sanitize.sanitize_for_maas(body)
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
            _sanitize_deepseek(body, model_name)
        elif route.get("provider") == "anthropic":
            sanitize.sanitize_for_maas(body)
        else:
            sanitize.strip_thinking_blocks(body)
        # tier 模型（pro/flash）不走直达，坠入 L1/L2 让路由+allocator 决定
        if model_name not in (_TIERS.get("pro",""), _TIERS.get("flash",""), _TIERS.get("max","")):
            return route, model_name, "agent-model", None, None, False, ""
        body["model"] = "auto"
        # 抹掉 model 后继续坠落 L1/L2

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
        return None, None, "l2-pending", l2_future, ratio, is_sub, user_query

    telemetry.log(f"L1 {reason}: {model_name or 'auto'} -> {routed_model}", phase="ROUTE")

    # trivial 分流：把一部分 L1:trivial 的 flash 流量按 ratio 交给 qwen3.6-flash(TP)，
    # 消耗临近过期的 TP 额度、替 flash 分担现金。仅当 trivial 且 qwen3.6-flash 可用才分。
    if reason == "L1:trivial":
        req_id = telemetry.get_req_id()
        try:
            _rem, _tot, _d = telemetry.get_real_credits()
            _r = ALLOCATOR.compute_ratio(_rem, _tot, _d)
            split_to = _flash_tp_split(req_id, _r)
            if split_to != _TIERS["flash"] and ROUTES.get(split_to):
                routed_model = split_to
                reason += f"/tp-flash-split(ratio={_r:.2f})"
        except Exception:
            pass  # 分流失败不影响原有 flash 路由

    body["model"] = routed_model
    route = ROUTES.get(routed_model) or ROUTES.get(re.sub(r'\[.*\]', '', routed_model))
    # L1 分类结果找不到路由时 fallback 到 flash（兜底，不返回 404）
    if not route:
        fallback = _TIERS["flash"]
        telemetry.log(f"L1 routed_model={routed_model!r} not in ROUTES, fallback to {fallback}", "WARN", "ROUTE")
        body["model"] = fallback
        route = ROUTES.get(fallback)
        routed_model = fallback
        reason += f"/fallback-to-{fallback}"
    if route and route.get("provider") == "deepseek":
        _sanitize_deepseek(body, routed_model)
    elif route and route.get("provider") == "anthropic":
        sanitize.sanitize_for_maas(body)
    else:
        sanitize.strip_thinking_blocks(body)
    return route, routed_model, reason, None, None, False, ""


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


def _resanitize_for_upgrade(body, new_route, old_route, new_model_name=None):
    """Re-apply sanitization if provider changed after model upgrade."""
    new_prov = new_route.get("provider") if new_route else None
    old_prov = old_route.get("provider") if old_route else None
    if new_prov == old_prov or new_prov is None:
        return
    if new_prov == "deepseek":
        _sanitize_deepseek(body, new_model_name or "")
    elif new_prov == "anthropic":
        sanitize.sanitize_for_maas(body)


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
        _resanitize_for_upgrade(body, new_route, route, new_model)
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


@app.get("/v1/token-stats")
async def get_token_stats():
    """Aggregated token usage stats from token_usage.jsonl — today/month/all/trends/models/providers/balance."""
    import os as _os
    import json as _json
    import time as _time
    records: list[dict] = []
    tup = telemetry.TOKEN_USAGE_PATH
    if _os.path.isfile(tup):
        with open(tup, "r", encoding="utf-8-sig") as _f:
            for _line in _f:
                _line = _line.strip()
                if _line:
                    try:
                        records.append(_json.loads(_line))
                    except _json.JSONDecodeError:
                        pass

    def _fmt_ratio(inp: int, out: int) -> str:
        return "∞" if out == 0 else f"{inp / out:.1f}"

    def _provider_of(model: str) -> str:
        if model.startswith("deepseek"):
            return "DeepSeek"
        if model.startswith("qwen") or model.startswith("doubao"):
            return "Qwen (Plan)"
        return "Other"

    now_str = _time.strftime("%Y-%m-%d")
    month_str = now_str[:7]
    today = [r for r in records if (r.get("ts") or "").startswith(now_str)]
    month = [r for r in records if (r.get("ts") or "").startswith(month_str)]

    def _sum_stats(recs):
        return {
            "calls": len(recs),
            "input": sum(r.get("inputTokens", 0) for r in recs),
            "output": sum(r.get("outputTokens", 0) for r in recs),
        }

    # by model (today)
    by_model_map: dict[str, dict] = {}
    for r in today:
        m = r.get("model", "unknown")
        s = by_model_map.setdefault(m, {"calls": 0, "input": 0, "output": 0})
        s["calls"] += 1
        s["input"] += r.get("inputTokens", 0)
        s["output"] += r.get("outputTokens", 0)
    models = sorted(
        [{"model": k, **v, "ratio": _fmt_ratio(v["input"], v["output"])} for k, v in by_model_map.items()],
        key=lambda x: -x["calls"]
    )

    # by provider (today)
    by_prov_map: dict[str, dict] = {}
    for r in today:
        p = _provider_of(r.get("model", ""))
        s = by_prov_map.setdefault(p, {"calls": 0, "input": 0, "output": 0})
        s["calls"] += 1
        s["input"] += r.get("inputTokens", 0)
        s["output"] += r.get("outputTokens", 0)
    providers = sorted([{"provider": k, **v} for k, v in by_prov_map.items()], key=lambda x: -x["calls"])

    # 7-day trend
    trends = []
    model_trends = []
    for i in range(6, -1, -1):
        d = _time.strftime("%Y-%m-%d", _time.gmtime(_time.time() - i * 86400))
        day_recs = [r for r in records if (r.get("ts") or "").startswith(d)]
        trends.append({
            "date": d, "calls": len(day_recs),
            "input": sum(r.get("inputTokens", 0) for r in day_recs),
            "output": sum(r.get("outputTokens", 0) for r in day_recs),
        })
        by_m = {}
        for r in day_recs:
            mm = r.get("model", "unknown")
            by_m[mm] = by_m.get(mm, 0) + 1
        model_trends.append({
            "date": d,
            "models": sorted(
                [{"model": k, "calls": v} for k, v in by_m.items()],
                key=lambda x: -x["calls"]
            ),
        })

    # balance
    balance = None
    if _os.path.isfile(telemetry.BUDGET_POLICY_PATH):
        try:
            policy = _json.load(open(telemetry.BUDGET_POLICY_PATH, encoding="utf-8"))
            balance = policy.get("token_plan") or {}
        except Exception:
            pass

    return JSONResponse({
        "today": _sum_stats(today), "month": _sum_stats(month),
        "all": _sum_stats(records), "models": models, "providers": providers,
        "trends": trends, "modelTrends": model_trends, "balance": balance,
    })


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


@app.get("/v1/bypass")
async def get_bypass():
    _active_bypass()
    remaining = int(max(0, _BYPASS_EXPIRY - time.monotonic())) if _BYPASS_EXPIRY else None
    return JSONResponse({"bypass": _BYPASS_MODEL,
                         "failures": telemetry._bypass_health["failures"],
                         "ttl_remaining": remaining})


@app.post("/v1/bypass")
async def set_bypass(request: Request):
    global _BYPASS_MODEL, _BYPASS_EXPIRY
    try:
        data = await request.json()
    except Exception:
        data = {}
    model = (data.get("model") or "").strip()
    if not model or model in ("off", "0", "none", "null"):
        _BYPASS_MODEL = None
        _BYPASS_EXPIRY = None
        telemetry.log("GLOBAL BYPASS disabled via API", "INFO", "ROUTE")
        return JSONResponse({"bypass": None})
    if model == "1":
        model = _TIERS["max"]
    model = _get_alias(model) or model
    if model not in ROUTES:
        return JSONResponse({"error": f"unknown model: {model}"}, status_code=400)
    _BYPASS_MODEL = model
    telemetry.reset_bypass_health()
    ttl = data.get("ttl")
    if ttl is not None:
        try:
            ttl = int(ttl)
            if ttl > 0:
                _BYPASS_EXPIRY = time.monotonic() + ttl
        except (ValueError, TypeError):
            pass
    else:
        _BYPASS_EXPIRY = None
    telemetry.log(f"GLOBAL BYPASS set via API: {model}", "INFO", "ROUTE")
    return JSONResponse({"bypass": _BYPASS_MODEL})


@app.get("/v1/force-switch")
async def get_force_switch():
    _expire_force_switch()
    ts = _FORCE_SWITCH_TS
    ttl_remaining = int(max(0, ts - time.time())) if ts else None
    return JSONResponse({
        "enabled": _FORCE_SWITCH["enabled"],
        "model": _FORCE_SWITCH["model"],
        "ttl_remaining": ttl_remaining,
    })


@app.post("/v1/force-switch")
async def set_force_switch(request: Request):
    global _FORCE_SWITCH, _FORCE_SWITCH_TS
    try:
        data = await request.json()
    except Exception:
        data = {}
    enabled = data.get("enabled", True)
    if isinstance(enabled, str):
        enabled = enabled.lower() in ("1", "true", "yes", "on")
    enabled = bool(enabled)
    if not enabled:
        _FORCE_SWITCH = {"enabled": False, "model": None}
        _FORCE_SWITCH_TS = None
        _save_force_switch()
        telemetry.log("FORCE-SWITCH disabled", phase="ROUTE")
        return JSONResponse({"enabled": False, "model": None, "ttl_remaining": None})

    model = (data.get("model") or "").strip() or _TIERS["max"]
    model = model.lstrip("@")
    model = _get_alias(model) or model
    if model not in ROUTES:
        return JSONResponse({"error": f"unknown model: {model}"}, status_code=400)

    _FORCE_SWITCH = {"enabled": True, "model": model}
    ttl = data.get("ttl")
    if enabled and ttl is not None:
        try:
            _FORCE_SWITCH_TS = time.time() + int(ttl) if int(ttl) > 0 else None
        except (ValueError, TypeError):
            _FORCE_SWITCH_TS = None
    else:
        _FORCE_SWITCH_TS = None
    _save_force_switch()
    telemetry.log(f"FORCE-SWITCH {'enabled' if enabled else 'disabled'} -> {model}", phase="ROUTE")
    return JSONResponse({
        "enabled": _FORCE_SWITCH["enabled"],
        "model": _FORCE_SWITCH["model"],
        "ttl_remaining": int(max(0, _FORCE_SWITCH_TS - time.time())) if _FORCE_SWITCH_TS else None,
    })


@app.post("/v1/restart")
async def restart_self():
    """Graceful restart: spawn new process, then shutdown current uvicorn.
    New process inherits same port and args. No downtime — new process starts
    before old one exits, and port binding happens after old process releases it
    (~1-2s window, acceptable for internal proxy)."""
    import subprocess as _sp
    telemetry.log("Restart requested via /v1/restart", phase="SYSTEM")

    port = _restart_port
    extra_args = _restart_args
    py = sys.executable
    script = os.path.abspath(__file__)

    # Spawn replacement
    cmd = [py, script, str(port)] + extra_args
    _sp.Popen(cmd, creationflags=0x08000000 if sys.platform == "win32" else 0)  # CREATE_NO_WINDOW

    # Shutdown uvicorn after 0.5s (let response flush)
    def _shutdown():
        import asyncio as _asyncio
        time.sleep(0.5)
        try:
            _loop = _asyncio.new_event_loop()
            _loop.run_until_complete(telemetry.shutdown())
        except Exception:
            pass
        os._exit(0)

    import threading
    threading.Thread(target=_shutdown, daemon=True).start()
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

    route = model_name = _reason = None

    # Global bypass check
    effective = _active_bypass()
    if effective:
        body["model"] = effective
        route = ROUTES.get(effective)
        if route:
            if route.get("provider") == "deepseek":
                _sanitize_deepseek(body, effective)
            elif route.get("provider") == "anthropic":
                sanitize.sanitize_for_maas(body)
            else:
                sanitize.strip_thinking_blocks(body)
            model_name = effective
            _reason = "global-bypass"

    if not route:
        route, model_name, _reason = _resolve_prompt_model(body)
        route, model_name = _maybe_escalate(body, route, model_name)

    if not route:
        route, model_name, _reason = _resolve_bypass(body, request.headers)
        route, model_name = _maybe_escalate(body, route, model_name)

    if not route:
        route, model_name, _reason, l2_future, ratio, is_sub, l2_user_query = _route_and_sanitize(body)

        if l2_future is not None:
            route, model_name = await _resolve_l2(body, l2_future, ratio, is_sub, l2_user_query)

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
        if route["provider"] in ("openai",):
            h = handle_openai_stream if is_stream else handle_openai
        else:
            h = handle_anthropic_stream if is_stream else handle_anthropic
        return await h(body, route, model_name, ROUTES, http_client,
                       work_dir, session_id, _reason=_reason,
                       is_bypass=_bypass_fail())
    finally:
        await telemetry.record_latency(model_name, (time.time() - _t0) * 1000)


@app.post("/v1/chat/completions")
async def proxy_openai(request: Request):
    telemetry.set_req_id(uuid.uuid4().hex[:8])
    body = await request.json()
    sanitize.embed_images(body)

    route = model_name = _reason = None

    # Global bypass check
    effective = _active_bypass()
    if effective:
        body["model"] = effective
        route = ROUTES.get(effective)
        if route:
            if route.get("provider") == "deepseek":
                _sanitize_deepseek(body, effective)
            elif route.get("provider") == "anthropic":
                sanitize.sanitize_for_maas(body)
            else:
                sanitize.strip_thinking_blocks(body)
            model_name = effective
            _reason = "global-bypass"

    if not route:
        route, model_name, _reason = _resolve_prompt_model(body)
        route, model_name = _maybe_escalate(body, route, model_name)

    if not route:
        route, model_name, _reason = _resolve_bypass(body, request.headers)
        route, model_name = _maybe_escalate(body, route, model_name)

    if not route:
        route, model_name, _reason, l2_future, ratio, is_sub, l2_user_query = _route_and_sanitize(body)

        if l2_future is not None:
            route, model_name = await _resolve_l2(body, l2_future, ratio, is_sub, l2_user_query)

        route, model_name = _maybe_escalate(body, route, model_name)

    if not route:
        return JSONResponse({"error": f"unknown model: {model_name}"}, status_code=404)
    is_stream = body.get("stream", False)
    model_name_display = body.get("model", model_name)
    telemetry.log(f"{model_name_display} /v1/chat/completions{' (stream)' if is_stream else ''}", phase="UPSTREAM")

    await telemetry.record_request(model_name_display, _reason)

    work_dir = request.headers.get("x-claude-work-dir", "")
    session_id = request.headers.get("x-claude-session-id", "")

    _t0 = time.time()
    try:
        # /v1/chat/completions 收到的是 OpenAI 格式 body。
        # deepseek provider 的 API base 是 OpenAI 兼容端点，直接透传 OpenAI 格式（handle_openai 自动处理）。
        # anthropic provider 走 handle_anthropic，但其 body 需是真 Anthropic 格式——
        # 原生 OpenAI 格式到这里无法直接转（convert 模块无 Anthropic 请求转换），保持原逻辑由 handle_anthropic 处理。
        if route["provider"] in ("openai", "deepseek"):
            h = handle_openai_stream if is_stream else handle_openai
        else:
            h = handle_anthropic_stream if is_stream else handle_anthropic
        return await h(body, route, model_name_display, ROUTES, http_client,
                       work_dir, session_id, _reason=_reason,
                       is_bypass=_bypass_fail())
    finally:
        await telemetry.record_latency(model_name_display, (time.time() - _t0) * 1000)


@app.post("/v1/embeddings")
async def proxy_embeddings(request: Request):
    """Forward OpenAI-compatible embedding request to DashScope text-embedding-v3.

    从 git 44f527d (07-16) 恢复——07-26 大重写时被误删，导致 digital-twin 的
    generate_embedding 恒返 None、向量检索在生产静默失效。恢复后需旁路测试。
    """
    body = await request.json()
    model = body.get("model", "text-embedding-v3")
    inp = body.get("input", "")

    dashscope_key = os.environ.get("DASHSCOPE_API_KEY", "")
    if not dashscope_key:
        return JSONResponse({"error": "DASHSCOPE_API_KEY not configured"}, status_code=500)

    payload = {
        "model": model,
        "input": inp,
        "encoding_format": "float",
    }
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                "https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings",
                json=payload,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {dashscope_key}",
                },
            )
        if resp.status_code != 200:
            detail = resp.text[:500]
            telemetry.log(f"Embedding API error {resp.status_code}: {detail}", "ERROR", "EMBED")
            return JSONResponse({"error": f"DashScope {resp.status_code}: {detail}"}, status_code=resp.status_code)
        return JSONResponse(resp.json())
    except httpx.TimeoutException:
        telemetry.log("Embedding API timeout", "ERROR", "EMBED")
        return JSONResponse({"error": "DashScope embedding request timed out"}, status_code=504)


@app.post("/{path:path}")
async def not_found_post(path: str):
    if path not in ("v1/messages", "v1/chat/completions", "v1/rules/debug", "v1/embeddings"):
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
    load_aliases()

    _resolve_bypass_global()

    async def _bypass_disable():
        global _BYPASS_MODEL, _BYPASS_EXPIRY
        if _BYPASS_MODEL:
            telemetry.log(
                f"GLOBAL BYPASS auto-disabled: {_BYPASS_MODEL} failed consecutively, routing restored",
                "INFO", "ROUTE"
            )
            _BYPASS_MODEL = None
            _BYPASS_EXPIRY = None
    telemetry.set_bypass_disable_hook(_bypass_disable)

    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    flags = set(a for a in sys.argv[1:] if a in ("-v", "--verbose"))
    verbose = "-v" in flags or "--verbose" in flags

    log_dir = os.path.dirname(os.path.abspath(__file__))
    token_log = os.path.join(log_dir, "token_usage.jsonl")
    log_file = os.path.join(log_dir, "proxy.log")
    access_log = os.path.join(log_dir, "proxy_access.log")

    telemetry.init(token_log_path=token_log, log_file=log_file, access_log=access_log, verbose=verbose)
    _load_force_switch()

    import uvicorn
    port = int(args[0]) if args else 4000

    # Store for /v1/restart
    global _restart_port, _restart_args
    _restart_port = port
    _restart_args = [a for a in sys.argv[1:] if a.startswith("-")]

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

    for attempt in range(3):
        try:
            uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
            break
        except OSError as e:
            if "address already in use" in str(e).lower() and attempt < 2:
                telemetry.log(f"Port {port} in use, retrying in 2s (attempt {attempt+1})", "WARN", "SYSTEM")
                time.sleep(2)
            else:
                raise


if __name__ == "__main__":
    main()

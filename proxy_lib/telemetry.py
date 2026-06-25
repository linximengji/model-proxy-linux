"""Telemetry — token recording, stats, health tracking, logging. Mutable module state."""
import json
import os
import time
import asyncio
import glob as glob_m
import contextvars
from datetime import datetime, date, timezone, timedelta

START_TIME = time.time()

stats = {
    "requests": 0,
    "by_model": {},
    "by_reason": {},
    "errors": 0,
    "errors_by_model": {},
    "latency_ms": {},
    "tokens": {},
}

route_health: dict = {}

# 错误类型 → (阈值, cooldown秒数)
HEALTH_CONFIG: dict[str, dict] = {
    "rate_limit":       {"threshold": 3, "cooldown": 60},      # 429 — 瞬时，快速恢复
    "quota_exhausted":  {"threshold": 1, "cooldown": 86400},   # 402 — 一天内不重试
    "server_error":     {"threshold": 2, "cooldown": 600},     # 502/503 — 服务挂了
    "connection":       {"threshold": 3, "cooldown": 120},     # 超时/DNS/Connect
}

def _classify_error(status_code=None, error_type=None):
    if error_type:
        return error_type
    if status_code == 429:
        return "rate_limit"
    if status_code == 402:
        return "quota_exhausted"
    if status_code in (502, 503):
        return "server_error"
    if status_code and 500 <= status_code < 600:
        return "server_error"
    return None  # 4xx non-402/429 不计入降级

_stats_lock = asyncio.Lock()

VERBOSE = False
LOG_FILE = None
ACCESS_LOG = None
TOKEN_LOG = None

_req_id_var = contextvars.ContextVar('_req_id', default=None)


def set_req_id(rid):
    _req_id_var.set(rid)


def get_req_id():
    return _req_id_var.get() or "--------"


def init(token_log_path=None, log_file=None, access_log=None, verbose=False):
    global TOKEN_LOG, LOG_FILE, ACCESS_LOG, VERBOSE
    TOKEN_LOG = token_log_path
    LOG_FILE = log_file
    ACCESS_LOG = access_log
    VERBOSE = verbose
    _rotate_logs()
    _rotate_token_log()


def _rotate_logs():
    today = date.today()
    for log_path in [LOG_FILE, ACCESS_LOG]:
        if not log_path or not os.path.isfile(log_path):
            continue
        if os.path.getsize(log_path) == 0:
            continue
        mtime = datetime.fromtimestamp(os.path.getmtime(log_path)).date()
        if mtime >= today:
            continue
        rotated = _rotated_name(log_path, mtime)
        if not os.path.exists(rotated):
            os.rename(log_path, rotated)
    _cleanup_rotated(LOG_FILE, keep_days=3)
    _cleanup_rotated(ACCESS_LOG, keep_days=3)


def _rotated_name(log_path, file_date):
    base_dir = os.path.dirname(log_path)
    return os.path.join(base_dir, f"proxy.{file_date.isoformat()}.log")


def _cleanup_rotated(log_path, keep_days):
    if not log_path:
        return
    base_dir = os.path.dirname(log_path)
    pattern = os.path.join(base_dir, "proxy.*.log")
    cutoff = date.today() - timedelta(days=keep_days)
    for f in glob_m.glob(pattern):
        try:
            fname = os.path.basename(f)
            parts = fname.replace("proxy.", "").replace(".log", "").split("-")
            if len(parts) == 3:
                fdate = date(int(parts[0]), int(parts[1]), int(parts[2]))
                if fdate < cutoff:
                    os.remove(f)
        except (ValueError, OSError):
            pass


def _rotate_token_log():
    if not TOKEN_LOG or not os.path.isfile(TOKEN_LOG):
        return
    if os.path.getsize(TOKEN_LOG) == 0:
        return
    mtime = datetime.fromtimestamp(os.path.getmtime(TOKEN_LOG))
    now = datetime.now()
    if mtime.year == now.year and mtime.month == now.month:
        return
    base_dir = os.path.dirname(TOKEN_LOG)
    archive_name = os.path.join(base_dir, f"token_usage.{mtime.strftime('%Y-%m')}.jsonl")
    if not os.path.exists(archive_name):
        os.rename(TOKEN_LOG, archive_name)


def log(msg, level="INFO", phase=None):
    ts = time.strftime("%H:%M:%S")
    rid = get_req_id()
    phase_str = f"[{phase}] " if phase else ""
    line = f"[{ts}] [{rid}] [{level}] {phase_str}{msg}"
    if level in ("ERROR", "WARN") or VERBOSE:
        print(line, flush=True)
    if LOG_FILE and (level in ("ERROR", "WARN") or VERBOSE):
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def write_token_usage(model, input_tokens, output_tokens, cache_read=0, work_dir=None, session_id=None):
    if not TOKEN_LOG:
        return
    try:
        record = {
            "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"),
            "model": model,
            "inputTokens": input_tokens,
            "outputTokens": output_tokens,
            "cacheReadTokens": cache_read,
        }
        if work_dir:
            record["workDir"] = work_dir
        if session_id:
            record["session_id"] = session_id
        with open(TOKEN_LOG, "a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:
        log(f"write_token_usage: {e}", "ERROR")


def build_stats():
    uptime = time.time() - START_TIME
    result = {
        "uptime_seconds": round(uptime, 1),
        "requests_total": stats["requests"],
        "errors_total": stats["errors"],
        "by_model": dict(stats["by_model"]),
        "by_reason": dict(stats["by_reason"]),
        "errors_by_model": dict(stats["errors_by_model"]),
        "latency_ms": {},
        "tokens": dict(stats["tokens"]),
    }
    for m, d in stats["latency_ms"].items():
        if d["count"] > 0:
            result["latency_ms"][m] = {
                "min": round(d["min"], 1),
                "max": round(d["max"], 1),
                "avg": round(d["sum"] / d["count"], 1),
                "count": d["count"],
            }
    return result


async def record_request(model_name, reason=None):
    async with _stats_lock:
        stats["requests"] += 1
        stats["by_model"][model_name] = stats["by_model"].get(model_name, 0) + 1
        if reason:
            stats["by_reason"][reason] = stats["by_reason"].get(reason, 0) + 1


async def record_error(model_name="unknown"):
    async with _stats_lock:
        stats["errors"] += 1
        stats["errors_by_model"][model_name] = stats["errors_by_model"].get(model_name, 0) + 1


async def record_tokens(model_name, input_tokens, output_tokens, cache_read=0, work_dir=None, session_id=None):
    async with _stats_lock:
        t = stats["tokens"].setdefault(model_name, {"input": 0, "output": 0})
        t["input"] += input_tokens
        t["output"] += output_tokens
    write_token_usage(model_name, input_tokens, output_tokens, cache_read, work_dir, session_id)


async def record_latency(model_name, lat_ms):
    async with _stats_lock:
        d = stats["latency_ms"].setdefault(model_name, {"min": 1e9, "max": 0, "sum": 0, "count": 0})
        if lat_ms < d["min"]:
            d["min"] = lat_ms
        if lat_ms > d["max"]:
            d["max"] = lat_ms
        d["sum"] += lat_ms
        d["count"] += 1


async def record_failure(model_name, status_code=None, error_type=None):
    err_kind = _classify_error(status_code, error_type)
    async with _stats_lock:
        h = route_health.setdefault(model_name, {"failures": 0, "last_failure": 0, "by_type": {}})
        h["failures"] += 1
        h["last_failure"] = time.time()
        if err_kind:
            bt = h["by_type"].setdefault(err_kind, {"count": 0, "last": 0})
            bt["count"] += 1
            bt["last"] = time.time()


async def record_success(model_name):
    async with _stats_lock:
        h = route_health.get(model_name)
        if h:
            h["failures"] = 0
            h["by_type"] = {}


def is_degraded(model_name):
    h = route_health.get(model_name)
    if not h:
        return False
    now = time.time()
    by_type = h.get("by_type", {})
    for err_type, info in list(by_type.items()):
        cfg = HEALTH_CONFIG.get(err_type)
        if not cfg:
            continue
        if info["count"] >= cfg["threshold"]:
            if now - info["last"] <= cfg["cooldown"]:
                return True
            info["count"] = 0  # cooldown 过期，清零
    return False


async def reset_stats():
    global START_TIME
    async with _stats_lock:
        START_TIME = time.time()
        for k in stats:
            stats[k] = {} if isinstance(stats[k], dict) else 0
        route_health.clear()


# ── Budget policy (Token Plan aware routing) ─────────────────────────────────

BUDGET_POLICY_PATH = r"D:\ClaudeProjects\.claudetalk\routing_policy.json"
TOKEN_USAGE_PATH = r"D:\ClaudeProjects\.claudetalk\token_usage.jsonl"


def load_budget_policy():
    try:
        with open(BUDGET_POLICY_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


# 60-second cache for real-time credits computation
_credits_cache: dict = {"rem": None, "total": None, "days": None, "ts": 0}


_days_cache: dict = {"days": None, "total": None, "ts": 0}


def _read_days_from_file():
    """Read days_to_expiry and credits_total from file fresh every call.
    Separated from the 60s compute cache so manual edits to routing_policy.json
    (e.g. days_to_expiry=0 when plan is about to expire) are picked up immediately,
    not overwritten by stale cached values in write_routing_policy.
    """
    now = time.time()
    if now - _days_cache["ts"] < 60:
        return _days_cache["total"], _days_cache["days"]

    total = 25000
    days = 0
    policy = load_budget_policy()
    if policy:
        tp = policy.get("token_plan", {})
        total = tp.get("credits_total", total)
        days = tp.get("days_to_expiry", 0)

    _days_cache["total"] = total
    _days_cache["days"] = days
    _days_cache["ts"] = now
    return total, days


def get_real_credits():
    """Compute real-time Token Plan remaining from token_usage.jsonl.

    60s cache for expensive qwen-credit computation.
    days_to_expiry/credits_total read separately (cheap) for immediate manual edits.
    Returns (credits_remaining, credits_total, days_to_expiry).
    """
    now = time.time()
    total, days = _read_days_from_file()

    if now - _credits_cache["ts"] < 60:
        return _credits_cache["rem"], total, days

    # Compute actual used credits from token_usage.jsonl
    used, _, _ = _compute_qwen_credits_all()
    remaining = max(0, total - used)

    _credits_cache["rem"] = remaining
    _credits_cache["ts"] = now

    return remaining, total, days


def _compute_qwen_credits_all():
    """Full-scan of token_usage.jsonl for qwen model credits.

    Returns (total_credits_used, record_count, per_model: dict).
    """
    if not os.path.isfile(TOKEN_USAGE_PATH):
        return 0, 0, {}
    total_effective = 0
    count = 0
    per_model = {}
    try:
        with open(TOKEN_USAGE_PATH, "r", encoding="utf-8-sig", errors="replace") as f:
            for line in f:
                line = line.strip().rstrip("\r")
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                model = rec.get("model", "")
                mk = _qwen_model_key(model)
                if mk is None:
                    continue
                mult = QWEN_MULTIPLIERS.get(mk, 1.0)
                eff = (rec.get("inputTokens", 0) + rec.get("outputTokens", 0)) * mult
                total_effective += eff
                count += 1
                per_model[mk] = per_model.get(mk, 0) + eff
    except OSError:
        return 0, 0, {}
    return round(total_effective / EFFECTIVE_TOKENS_PER_CREDIT, 2), count, per_model


def write_routing_policy(remaining, total, days):
    """Back-write correct credits to routing_policy.json so dashboard reads right values."""
    try:
        data = {
            "token_plan": {
                "credits_remaining": round(remaining, 2),
                "credits_total": total,
                "days_to_expiry": days,
                "last_updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            }
        }
        with open(BUDGET_POLICY_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        log(f"write_routing_policy: {e}", "ERROR")


def compute_proxy_burn(hours=24):
    """Read token_usage.jsonl, sum input+output tokens in last N hours.
    Returns (token_total, request_count) or (0, 0) on error."""
    if not os.path.isfile(TOKEN_USAGE_PATH):
        return 0, 0
    cutoff = time.time() - hours * 3600
    total = 0
    count = 0
    try:
        with open(TOKEN_USAGE_PATH, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                ts = rec.get("ts", "")
                if ts:
                    try:
                        t = datetime.fromisoformat(ts).timestamp()
                    except (ValueError, TypeError):
                        continue
                    if t < cutoff:
                        continue
                total += rec.get("inputTokens", 0) + rec.get("outputTokens", 0)
                count += 1
    except OSError:
        return 0, 0
    return total, count


# ── Token Plan credit estimation ────────────────────────────────────────────
# qwen model → effective-token multiplier (qwen3.6-plus = 1x base)
# Derived from pricing ratio: 0.010/0.002 = 5x (qwen3.7-max input vs plus)
QWEN_MULTIPLIERS: dict[str, float] = {
    "qwen3.6-plus": 1.0,
    "qwen3.6-maas": 1.0,
    "qwen3-coder-plus": 3.0,
    "qwen3.7-max": 5.0,
    "qwen3.7-max-vision": 5.0,
}
# Calibrated: 23,465.94 computed credits ≈ 42,522.50 real credits (1.812×)
# 10,000 / 1.812 ≈ 5519
EFFECTIVE_TOKENS_PER_CREDIT = 5519


def _qwen_model_key(model: str) -> str | None:
    """Extract qwen model name from model field (handles composite names like 'a,b').
    Returns the model key if it's a qwen model, None otherwise.
    Searches ALL comma-separated parts — qwen may be the first or last element
    in a fallback chain (e.g. 'qwen3.7-max,deepseek-v4-flash' or 'deepseek-v4-flash,qwen3.7-max').
    """
    for name in model.split(","):
        name = name.strip()
        if "qwen" not in name.lower():
            continue
        for known in QWEN_MULTIPLIERS:
            if known in name:
                return known
    return None


def compute_qwen_since(iso_cutoff):
    """Estimate Token Plan credits consumed since a given timestamp.
    Applies model-specific multipliers and empirical conversion rate.
    Returns estimated credits (float, 0 if no data or error).
    """
    if not os.path.isfile(TOKEN_USAGE_PATH) or not iso_cutoff:
        return 0
    try:
        cutoff = datetime.fromisoformat(iso_cutoff).timestamp()
    except (ValueError, TypeError):
        return 0
    total_effective = 0  # base-tokens adjusted by multiplier
    try:
        with open(TOKEN_USAGE_PATH, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                ts = rec.get("ts", "")
                if not ts:
                    continue
                try:
                    if datetime.fromisoformat(ts).timestamp() < cutoff:
                        continue
                except (ValueError, TypeError):
                    continue
                model = rec.get("model", "")
                mk = _qwen_model_key(model)
                if mk is None:
                    continue
                mult = QWEN_MULTIPLIERS.get(mk, 1.0)
                total_effective += (rec.get("inputTokens", 0) + rec.get("outputTokens", 0)) * mult
    except OSError:
        return 0
    return round(total_effective / EFFECTIVE_TOKENS_PER_CREDIT, 2)


def compute_qwen_total_credits():
    """扫描 token_usage.jsonl 全量数据，计算 Qwen 模型消耗的总 credits。
    无时间过滤——反映 Token Plan 全生命周期消耗。
    返回 (credits_used: float, credits_total: int) 或 (0, 25000)。
    """
    used, _, _ = _compute_qwen_credits_all()
    policy = load_budget_policy()
    total = 25000
    if policy:
        total = policy.get("token_plan", {}).get("credits_total", total)
    return used, total

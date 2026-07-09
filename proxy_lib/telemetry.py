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

BUDGET_POLICY_PATH = "/home/ubuntu/projects/.claudetalk/routing_policy.json"
TOKEN_USAGE_PATH = "/home/ubuntu/projects/.claudetalk/token_usage.jsonl"


def load_budget_policy():
    try:
        with open(BUDGET_POLICY_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


# 60-second cache for real-time credits computation
_credits_cache: dict = {"rem": None, "total": None, "days": None, "ts": 0}


_days_cache: dict = {"days": None, "total": None, "start_ts": None, "ts": 0}


def _read_days_from_file():
    """Read credits_total and plan_start_ts from file fresh every call.

    days_to_expiry is auto-computed from plan_start_ts + plan_duration_days,
    not read directly — ensures it decays naturally with wall-clock time.

    60s cache, Separated from compute cache so manual edits (e.g. plan_start_ts reset)
    are picked up within a minute.
    """
    now_ts = time.time()
    if now_ts - _days_cache["ts"] < 60:
        return _days_cache["total"], _days_cache["days"], _days_cache["start_ts"]

    total = 25000
    duration_days = 0
    start_ts = None
    policy = load_budget_policy()
    if policy:
        tp = policy.get("token_plan", {})
        total = tp.get("credits_total", total)
        # plan_duration_days is the canonical field; fall back to days_to_expiry for legacy
        duration_days = tp.get("plan_duration_days", tp.get("days_to_expiry", 0))
        start_ts = tp.get("plan_start_ts")

    # Auto-decay from plan_start_ts + duration_days
    if start_ts:
        try:
            plan_start = datetime.fromisoformat(start_ts)
            if plan_start.tzinfo is None:
                plan_start = plan_start.replace(tzinfo=timezone.utc)
            elapsed = (datetime.now(timezone.utc) - plan_start).days
            days = max(0, duration_days - elapsed)
        except (ValueError, TypeError):
            days = duration_days
    else:
        days = duration_days

    _days_cache["total"] = total
    _days_cache["days"] = days
    _days_cache["start_ts"] = start_ts
    _days_cache["ts"] = now_ts
    return total, days, start_ts


def get_real_credits():
    """Compute real-time Token Plan remaining from token_usage.jsonl.

    If the user manually set manual_baseline + manual_updated_at, the manual value
    serves as the baseline and only credits consumed SINCE that timestamp are subtracted.
    Otherwise computes from plan_start_ts.

    60s cache for expensive qwen-credit computation.
    Returns (credits_remaining, credits_total, days_to_expiry).
    """
    now = time.time()
    total, days, start_ts = _read_days_from_file()

    if now - _credits_cache["ts"] < 60:
        return _credits_cache["rem"], total, days

    # Check for manual baseline
    policy = load_budget_policy()
    manual_baseline = None
    manual_ts = None
    if policy:
        tp = policy.get("token_plan", {})
        manual_baseline = tp.get("manual_baseline")
        manual_ts = tp.get("manual_updated_at")
        last_proxy_at = tp.get("last_updated_at")
        if not (manual_ts and last_proxy_at and manual_ts > last_proxy_at):
            manual_baseline = None

    if manual_baseline is not None and manual_ts:
        # Compute credits consumed since manual baseline timestamp
        used_since, _, _ = _compute_tp_credits_all(plan_start_ts=manual_ts)
        remaining = max(0, manual_baseline - used_since)
    else:
        # Full scan from plan_start_ts
        used, _, _ = _compute_tp_credits_all(plan_start_ts=start_ts)
        remaining = max(0, total - used)

    _credits_cache["rem"] = remaining
    _credits_cache["ts"] = now

    return remaining, total, days


def _compute_tp_credits_all(plan_start_ts=None):
    """Full-scan of token_usage.jsonl for TP-mapped model credits.

    Uses flat rate per 1K total tokens per model (TP_PRICING_FLAT).
    DeepSeek models are excluded — they use native DeepSeek API keys, not TP.

    Args:
        plan_start_ts: ISO timestamp string. Records with ts < plan_start_ts are skipped.
    Returns (total_credits_used, record_count, per_model: dict).
    """
    if not os.path.isfile(TOKEN_USAGE_PATH):
        return 0, 0, {}
    total = 0.0
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
                if "costUSD" in rec:
                    continue
                if plan_start_ts and rec.get("ts", "") < plan_start_ts:
                    continue
                model = rec.get("model", "")
                if model.startswith("deepseek-") or model.startswith("doubao-"):
                    continue
                km = _tp_model_key(model)
                rate = TP_PRICING_FLAT.get(km)
                if not rate:
                    continue
                inp = rec.get("inputTokens", 0)
                out = rec.get("outputTokens", 0)
                credits = _calc_tp_credits(inp, out, rate)
                total += credits
                count += 1
                per_model[model] = per_model.get(model, 0) + credits
    except OSError:
        return 0, 0, {}
    return round(total, 2), count, per_model


def write_routing_policy(remaining, total, days):
    """Back-write correct credits to routing_policy.json so dashboard reads right values.
    Preserves plan_start_ts, plan_duration_days, and respects manual_updated_at edits.

    When manual_updated_at > last_updated_at (manual baseline active):
    - writes the newly computed remaining (decremented from manual baseline)
    - does NOT update last_updated_at — this keeps the manual baseline active for subsequent
      proxy requests, allowing credits to accumulate on top of the manual value
    - only when the user removes manual_updated_at (or it becomes stale) does proxy revert
      to full-scan from plan_start_ts
    """
    try:
        existing = load_budget_policy()
        existing_tp = (existing or {}).get("token_plan", {})
        plan_start_ts = existing_tp.get("plan_start_ts")
        duration_days = existing_tp.get("plan_duration_days", existing_tp.get("days_to_expiry", 0))
        manual_at = existing_tp.get("manual_updated_at")
        last_proxy_at = existing_tp.get("last_updated_at")

        now_str = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        # Manual baseline active: write back decremented value but keep last_updated_at
        # at old_proxy_at so manual_ts > last_updated_at remains true. This keeps the
        # manual baseline active across all subsequent proxy requests.
        if manual_at and last_proxy_at and manual_at > last_proxy_at:
            final_remaining = round(remaining, 2)
            final_updated_at = last_proxy_at  # freeze — don't advance past manual_ts
        else:
            final_remaining = round(remaining, 2)
            final_updated_at = now_str

        data = {
            "token_plan": {
                "credits_remaining": final_remaining,
                "credits_total": total,
                "plan_duration_days": duration_days,
                "days_to_expiry": days,
                "plan_start_ts": plan_start_ts,
                "manual_baseline": existing_tp.get("manual_baseline"),
                "manual_updated_at": manual_at,
                "last_updated_at": final_updated_at,
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
                if "costUSD" in rec:
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


# ── Token Plan credit calculation ───────────────────────────────────────────
# Flat rate per 1K total tokens (不分 input/output), per-model calibration
# Calibrated 2026-07-07 against portal billing for New Token Plan (6/26~):
#   kimi-k2.7-code: 7,629,786 tokens → 8,523.46 credits → 1.1171/1K  (2026-07-07)
#   qwen3.7-max:      214,084 tokens → 11,602.10 credits → 54.19/1K (local data incomplete)
#   glm-5.2:          No local records → 1,567.35 credits (cannot calibrate)
# Models without portal data use nearest-match pricing.
TP_PRICING_FLAT: dict[str, float] = {
    # Calibrated from portal (2026-07-07)
    "kimi-k2.7-code": 1.1171,
    # Uncalibrated — estimated similar-to-kimi rates
    "kimi-k2.6":      1.16,
    "glm-5.2":        1.16,
    "glm-5.1":        1.16,
    "MiniMax-M2.5":   1.16,
    # Calibrated from portal (portal: 11602.10 credits / 214084 tokens → 54.19/1K;
    # local has 240983 tokens in same period → effective rate = 11602.10 / (240983/1000) = 48.14)
    "qwen3.7-max":       48.14,
    "qwen3.7-max-vision": 48.14,
    # Estimated — placeholder rates (not yet seen in portal)
    "qwen3.7-plus":      34.0,
    "qwen3-coder-plus":  34.0,
    "qwen3.6-plus":      34.0,
    "qwen3.6-maas":      34.0,
    "qwen3.6-flash":     34.0,
}

def _tp_model_key(model: str) -> str | None:
    """Resolve a model name to a TP_PRICING_FLAT key.

    Scans for substring match against TP_PRICING_FLAT keys.
    """
    for name in model.split(","):
        name = name.strip()
        for km in TP_PRICING_FLAT:
            if km in name:
                return km
    return None


def _calc_tp_credits(inp: int, out: int, rate: float) -> float:
    """(input + output) / 1000 * flat_rate per 1K tokens"""
    return ((inp + out) / 1000) * rate


def compute_tp_credits_since(iso_cutoff):
    """Compute TP credits consumed since a given timestamp.
    Uses flat rate per 1K total tokens per model.
    Returns credits (float, 0 if no data or error).
    """
    if not os.path.isfile(TOKEN_USAGE_PATH) or not iso_cutoff:
        return 0
    try:
        cutoff = datetime.fromisoformat(iso_cutoff).timestamp()
    except (ValueError, TypeError):
        return 0
    total = 0.0
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
                if "costUSD" in rec:
                    continue
                ts = rec.get("ts", "")
                if not ts:
                    continue
                try:
                    if datetime.fromisoformat(ts).timestamp() < cutoff:
                        continue
                except (ValueError, TypeError):
                    continue
                km = _tp_model_key(rec.get("model", ""))
                rate = TP_PRICING_FLAT.get(km)
                if not rate:
                    continue
                total += _calc_tp_credits(rec.get("inputTokens", 0), rec.get("outputTokens", 0), rate)
    except OSError:
        return 0
    return round(total, 2)


def compute_tp_total_credits():
    """全量扫描，计算所有 TP-mapped 模型的总 credits 消耗。
    返回 (credits_used: float, credits_total: int) 或 (0, 25000)。
    """
    used, _, _ = _compute_tp_credits_all()
    policy = load_budget_policy()
    total = 25000
    if policy:
        total = policy.get("token_plan", {}).get("credits_total", total)
    return used, total

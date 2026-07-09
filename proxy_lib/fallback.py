"""Fallback chain construction and request-with-fallback execution."""
import httpx
from proxy_lib import telemetry

log = telemetry.log


def build_fallback_chain(route, model_name, routes):
    """Build ordered list of (route_dict, model_name) pairs through fallback chain."""
    models_to_try = [(route, model_name)]
    seen = {model_name}
    cur_route, cur_name = route, model_name
    while True:
        fb_name = cur_route.get("fallback")
        if not fb_name or fb_name not in routes or fb_name in seen:
            break
        seen.add(fb_name)
        cur_route = routes[fb_name]
        cur_name = fb_name
        models_to_try.append((cur_route, cur_name))

    if len(models_to_try) > 1 and telemetry.is_degraded(models_to_try[0][1]):
        degraded_entry = models_to_try.pop(0)
        log(f"{degraded_entry[1]} degraded, skipping to fallback chain", phase="FALLBACK")
    return models_to_try


def is_quota_exhausted(http_err):
    if http_err.response.status_code not in (402, 429):
        return False
    try:
        body = http_err.response.json()
    except Exception:
        body = {}
    msg = str(body).lower()
    for kw in ("quota", "exhausted", "insufficient", "limit exceeded",
               "rate limit", "余额", "额度", "超限", "已达上限"):
        if kw in msg:
            return True
    return False


async def request_with_fallback(route, model_name, routes, http_client, build_req_kwargs, timeout=180):
    models_to_try = build_fallback_chain(route, model_name, routes)
    last_err = None
    for i, (r, m) in enumerate(models_to_try):
        try:
            kwargs = build_req_kwargs(r, m)
            kwargs["timeout"] = timeout
            resp = await http_client.request(**kwargs)
            await telemetry.record_success(m)
            return resp, m
        except httpx.HTTPStatusError as e:
            code = e.response.status_code
            await telemetry.record_failure(m, status_code=code)
            last_err = e
            if i == 0 and is_quota_exhausted(e):
                qb_name = r.get("quota_backup")
                if qb_name and qb_name in routes and qb_name not in {mm for _, mm in models_to_try}:
                    models_to_try.insert(1, (routes[qb_name], qb_name))
                    log(f"<- {m} quota exhausted, switching to {qb_name}", phase="FALLBACK")
                    continue
            if i < len(models_to_try) - 1:
                log(f"<- {m} ({code}), fallback to {models_to_try[i+1][1]}", phase="FALLBACK")
            else:
                log(f"<- {m} ({code}), no more fallback", "WARN", "FALLBACK")
            continue
        except (httpx.RequestError, httpx.TimeoutException) as e:
            await telemetry.record_failure(m, error_type="connection")
            last_err = e
            code = type(e).__name__
            if i < len(models_to_try) - 1:
                log(f"<- {m} ({code}), fallback to {models_to_try[i+1][1]}", phase="FALLBACK")
            else:
                log(f"<- {m} ({code}), no more fallback", "WARN", "FALLBACK")
            continue
    raise last_err

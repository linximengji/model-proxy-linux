"""Repro: confirm DeepSeek flash rejects thinking blocks, and the fix.

Runs 3 probes against the REAL DeepSeek Anthropic endpoint:
  A) flash + thinking blocks + thinking enabled  -> expect 400 (the bug)
  B) flash + thinking stripped + thinking disabled -> expect 200 (the fix)
  C) pro   + thinking blocks + thinking enabled   -> expect 200 (pro supports)
"""
import os, json, httpx

def load_env(path="/home/ubuntu/projects/proxy/.env"):
    d = {}
    if not os.path.isfile(path):
        return d
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        d[k.strip()] = v.strip().strip('"').strip("'")
    return d

env = load_env()
KEY = env.get("DEEPSEEK_API_KEY") or os.environ.get("DEEPSEEK_API_KEY")
URL = "https://api.deepseek.com/anthropic/v1/messages"
assert KEY, "DEEPSEEK_API_KEY not found"

# history: one assistant turn WITH thinking, one WITHOUT (mixed)
MIXED_MSGS = [
    {"role": "user", "content": "hi"},
    {"role": "assistant", "content": [
        {"type": "thinking", "thinking": "thinking about greeting"},
        {"type": "text", "text": "Hello!"},
    ]},
    {"role": "user", "content": "ok"},
    {"role": "assistant", "content": [{"type": "text", "text": "Sure."}]},
    {"role": "user", "content": "say hi back in one word"},
]

def probe(label, model, msgs, thinking):
    body = {"model": model, "max_tokens": 200, "stream": False,
            "messages": msgs}
    if thinking is not None:
        body["thinking"] = thinking
    try:
        r = httpx.post(URL, json=body, timeout=60,
                       headers={"Content-Type": "application/json",
                                "Authorization": f"Bearer {KEY}"})
        try:
            d = r.json()
            if r.status_code != 200:
                print(f"[{label}] HTTP {r.status_code} ERROR: {json.dumps(d.get('error', d), ensure_ascii=False)[:250]}")
            else:
                txt = "".join(b.get("text","") for b in d.get("content",[])
                              if isinstance(b,dict) and b.get("type")=="text")
                print(f"[{label}] HTTP 200 OK reply={txt[:60]!r} model={d.get('model')}")
        except Exception:
            print(f"[{label}] HTTP {r.status_code} raw={r.text[:200]}")
    except Exception as e:
        print(f"[{label}] EXC {type(e).__name__}: {e}")

import copy
probe("A flash+thinking_enabled", "deepseek-v4-flash", copy.deepcopy(MIXED_MSGS),
      {"type": "enabled", "budget_tokens": 1024})
probe("B flash+stripped+disabled", "deepseek-v4-flash", copy.deepcopy(MIXED_MSGS),
      {"type": "disabled"})
probe("C pro+thinking_enabled",   "deepseek-v4-pro",  copy.deepcopy(MIXED_MSGS),
      {"type": "enabled", "budget_tokens": 1024})

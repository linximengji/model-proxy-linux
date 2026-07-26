"""Proxy routing simulation v2 — replay L2 outputs through allocator."""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from proxy_lib.allocator import MultiModelAllocator

ALLOCATOR = MultiModelAllocator()

SCENARIOS = [
    {"name": "current (23272/72500, 3d)", "remaining": 23272, "total": 72500, "days": 3, "force_ratio": None},
    {"name": "current-map, ratio=0.6", "remaining": 23272, "total": 72500, "days": 3, "force_ratio": 0.6},
    {"name": "+moderate:general→glm, ratio=0.35", "remaining": 23272, "total": 72500, "days": 3, "force_ratio": None,
     "extra_map": {("moderate", "general"): "glm-5.2"}},
    {"name": "+moderate:general→glm, ratio=0.6", "remaining": 23272, "total": 72500, "days": 3, "force_ratio": 0.6,
     "extra_map": {("moderate", "general"): "glm-5.2"}},
]

CLASSIFICATIONS = [
    ("trivial", "general", 23),
    ("simple", "general", 35),
    ("simple", "code", 9),
    ("simple", "reasoning", 4),
    ("moderate", "code", 132),
    ("moderate", "reasoning", 123),
    ("moderate", "general", 42),
    ("complex", "reasoning", 109),
    ("complex", "code", 12),
]

CLASSIFIER_ROUTE = {"trivial": "flash", "simple": "flash", "moderate": "pro", "complex": "max"}
TIER_NAMES = {"flash": "deepseek-v4-flash", "pro": "deepseek-v4-pro", "max": "qwen3.8-max-preview"}

BASE_MAPPING: dict[tuple[str, str], str] = {
    ("moderate", "code"): "kimi-k2.7-code",
    ("complex", "reasoning"): "qwen3.8-max-preview",
    ("complex", "code"): "kimi-k2.7-code",
}
# lower-priority mappings not tested often:
BASE_MAPPING.setdefault(("moderate", "creative"), "glm-5.2")
BASE_MAPPING.setdefault(("moderate", "long_context"), "qwen3.6-flash")
BASE_MAPPING.setdefault(("complex", "creative"), "glm-5.2")
BASE_MAPPING.setdefault(("complex", "long_context"), "qwen3.7-plus")


def simulate(sc):
    mapping = dict(BASE_MAPPING)
    if sc.get("extra_map"):
        mapping.update(sc["extra_map"])
    
    ratio = sc["force_ratio"] if sc["force_ratio"] is not None else \
        ALLOCATOR.compute_ratio(sc["remaining"], sc["total"], sc["days"])
    
    print(f"\n{'='*70}")
    print(f"Scenario: {sc['name']}")
    print(f"  credits={sc['remaining']}/{sc['total']}, days={sc['days']}, ratio={ratio:.3f} ({ratio*100:.0f}%)")
    print(f"{'='*70}")
    print(f"{'Cx':12s} {'Task':12s} {'N':5s} {'Base':10s} {'Map':25s} {'Gate':8s} {'→ TP':7s} {'→ DS':7s}")
    print(f"{'-'*12} {'-'*12} {'-'*5} {'-'*10} {'-'*25} {'-'*8} {'-'*7} {'-'*7}")
    
    req_counter = [0]
    totals = {"tp_calls": 0, "ds_calls": 0, "tp_input_M": 0, "ds_input_M": 0}
    
    for cx, tt, count in CLASSIFICATIONS:
        base_tier = CLASSIFIER_ROUTE.get(cx, "pro")
        base_model = TIER_NAMES.get(base_tier, "deepseek-v4-pro")
        alloc_target = mapping.get((cx, tt))
        
        tp_dest = 0
        ds_dest = 0
        
        for _ in range(count):
            req_counter[0] += 1
            req_id = f"sim-{req_counter[0]}"
            
            if alloc_target and cx in ("moderate", "complex"):
                selected = ALLOCATOR.select(cx, tt, req_id, ratio)
            else:
                selected = None
            
            if selected:
                final = selected
                tp_dest += 1
                totals["tp_calls"] += 1
            elif cx == "complex" and base_tier == "max":
                # complex base is qwen3.8-max-preview — already TP
                final = base_model
                tp_dest += 1
                totals["tp_calls"] += 1
            else:
                final = base_model
                ds_dest += 1
                totals["ds_calls"] += 1
        
        gate_display = f"{tp_dest}/{count}" if (alloc_target and cx in ("moderate", "complex")) else "n/a"
        print(f"{cx:12s} {tt:12s} {count:5d} {base_model:10s} {alloc_target or '(none)':25s} {gate_display:8s} {tp_dest:7d} {ds_dest:7d}")
    
    all_calls = totals["tp_calls"] + totals["ds_calls"]
    tp_pct = totals["tp_calls"] / max(all_calls, 1) * 100
    print(f"\n  → TP: {totals['tp_calls']} ({tp_pct:.0f}%) | DS: {totals['ds_calls']} ({100-tp_pct:.0f}%) | Total L2: {all_calls}")


if __name__ == "__main__":
    for sc in SCENARIOS:
        simulate(sc)
    
    # Also test: what if we control complex:reasoning base tier?
    print("\n" + "="*70)
    print("BONUS: if complex:reasoning base tier was pro (not max → qwen3.8-max-preview)")
    print("="*70)
    # Hack: set CLASSIFIER_ROUTE["complex"] = "pro" temporarily
    old = CLASSIFIER_ROUTE["complex"]
    CLASSIFIER_ROUTE["complex"] = "pro"
    simulate({"name": "complex→pro base, alloc→ qwen3.8-max-preview (gate=0.3)",
              "remaining": 23272, "total": 72500, "days": 3, "force_ratio": None, "extra_map": None})
    CLASSIFIER_ROUTE["complex"] = old

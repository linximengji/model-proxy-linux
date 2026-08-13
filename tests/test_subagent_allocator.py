"""Verify sub-agents now flow through the token-plan allocator."""

import sys, os, asyncio
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import model_proxy


class _FakeResp:
    def __init__(self, text, ok=True):
        self.status_code = 200 if ok else 500
        self._t = text

    def json(self):
        return {"content": [{"type": "text", "text": self._t}]}


async def _fut(text, ok=True):
    return _FakeResp(text, ok)


def _resolve(text, is_sub_agent, ok=True):
    async def go():
        body = {"model": "auto"}
        await model_proxy._resolve_l2(body, _fut(text, ok), 1.0,
                                      is_sub_agent=is_sub_agent, user_query="x")
        return body
    return asyncio.run(go())


# 临时把 user_query 设成触发 local fallback（moderate/code）的兜底场景
def _resolve_fallback_fails_classify(is_sub_agent):
    async def go():
        body = {"model": "auto"}
        # flash 返回 500 → 解析失败 → local fallback 兜底，用含 code 关键词的 query
        # 触发 local classify 返回 moderate/code，sub-agent 应仍走 allocator
        await model_proxy._resolve_l2(body, _fut("zzz", ok=False), 1.0,
                                      is_sub_agent=is_sub_agent,
                                      user_query="实现一个缓存模块并写好测试")
        return body
    return asyncio.run(go())


def expect(name, body, want):
    got = body.get("model")
    assert got == want, f"FAIL {name}: got {got!r}, want {want!r}"
    print(f"  ok  {name} -> {got}")


def main():
    print("=== sub-agent uses allocator ===")
    expect("sub-agent moderate:code -> TP kimi",
           _resolve("moderate code medium", True), "kimi-k2.7-code")

    print("=== main thread unchanged ===")
    expect("main moderate:code -> TP kimi",
           _resolve("moderate code medium", False), "kimi-k2.7-code")

    print("=== trivial/simple never burn TP ===")
    expect("sub-agent trivial:general -> flash",
           _resolve("trivial general low", True), model_proxy._TIERS["flash"])
    expect("sub-agent simple:general -> flash",
           _resolve("simple general low", True), model_proxy._TIERS["flash"])

    print("=== flash-fail fallback still routes sub-agent via allocator ===")
    expect("sub-agent flash-fail -> local fallback moderate/code -> kimi",
           _resolve_fallback_fails_classify(True), "kimi-k2.7-code")

    print("\nALL PASS")


if __name__ == "__main__":
    main()

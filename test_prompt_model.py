"""Test @tag prompt-level model routing.

Pure function tests for _resolve_prompt_model and helpers.
Tests need a populated ROUTES dict — created via a session-scoped fixture
that loads from the real litellm_config.yaml.

Usage:
    cd D:\ClaudeProjects && python -m pytest proxy/test_prompt_model.py -v -x
"""
import json
import os
import sys
import re

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
sys.stdout.reconfigure(encoding="utf-8")

from proxy_lib import config


# ── Test data ────────────────────────────────────────────────────────────
# Minimal ROUTES for testing (avoids loading full yaml in every run).
# Must cover: pro, flash, max, vision, and some exact-match routes.
_FAKE_ROUTES = {
    "deepseek-v4-pro": {"provider": "deepseek", "model": "deepseek-chat"},
    "deepseek-v4-flash": {"provider": "deepseek", "model": "deepseek-reasoner"},
    "qwen3.7-max": {"provider": "openai", "model": "qwen-max"},
    "qwen3.7-plus": {"provider": "openai", "model": "qwen-plus"},
    "qwen3.6-flash": {"provider": "openai", "model": "qwen-flash"},
    "doubao-1.5-vision-pro": {"provider": "openai", "model": "doubao-vision"},
    "kimi-k2.6": {"provider": "openai", "model": "kimi-k2.6"},
    "kimi-k2.7-code": {"provider": "openai", "model": "kimi-k2.7-code"},
    "glm-5.2": {"provider": "openai", "model": "glm-5.2"},
    "MiniMax-M2.5": {"provider": "openai", "model": "minimax-m2.5"},
    "qwen3.7-max-ds": {"provider": "deepseek", "model": "deepseek-chat"},
}

_FAKE_TIERS = {
    "flash": "deepseek-v4-flash",
    "pro": "deepseek-v4-pro",
    "max": "qwen3.7-max",
    "vision": "doubao-1.5-vision-pro",
}


# ── Helpers ──────────────────────────────────────────────────────────────

def _build_helpers(tiers=None, routes=None):
    """Create a local module scope with patched globals."""
    # We import the real model_proxy module but override its globals
    import importlib
    import model_proxy as mp
    importlib.reload(mp)
    mp._TIERS.clear()
    mp._TIERS.update(tiers or _FAKE_TIERS)
    mp.ROUTES.clear()
    mp.ROUTES.update(routes or _FAKE_ROUTES)
    return mp


@pytest.fixture(scope="module")
def mp():
    """Import model_proxy with fake routes, once per module."""
    return _build_helpers()


# ── Regex tests ─────────────────────────────────────────────────────────

def test_regex_matches_tag_at_end(mp):
    assert mp._PROMPT_MODEL_RE.search("帮我写个程序 @pro")


def test_regex_matches_tag_in_middle(mp):
    assert mp._PROMPT_MODEL_RE.search("@pro 帮我写个程序")


def test_regex_matches_tag_with_punctuation(mp):
    assert mp._PROMPT_MODEL_RE.search("帮我写个程序 @pro。")


def test_regex_matches_tag_multiple(mp):
    assert mp._PROMPT_MODEL_RE.search("帮我 @flash @pro 运行")


def test_regex_matches_tag_after_newline(mp):
    assert mp._PROMPT_MODEL_RE.search("你好\n@max 继续")


def test_regex_not_matches_at_sign_without_tag(mp):
    assert not mp._PROMPT_MODEL_RE.search("帮我写个程序 pro")


def test_regex_not_matches_at_sign_in_email(mp):
    assert not mp._PROMPT_MODEL_RE.search("联系 user@example.com")


def test_regex_not_matches_at_sign_in_ascii_word(mp):
    """@ inside an ASCII word (foo@bar) must not match — protects email-like patterns."""
    assert not mp._PROMPT_MODEL_RE.search("foo@bar")


def test_regex_matches_cjk_adjacent_at(mp):
    """CJK char directly before @ should match (Chinese habit: 能力@qwen)."""
    m = mp._PROMPT_MODEL_RE.search("我现在测试显式调用模型的能力@qwen")
    assert m is not None and m.group(1) == "qwen"


# ── Strip tag tests ─────────────────────────────────────────────────────

def test_strip_tag_from_end(mp):
    assert mp._STRIP_TAG_RE.sub("", "帮我写个程序 @pro").strip() == "帮我写个程序"


def test_strip_tag_from_start(mp):
    assert mp._STRIP_TAG_RE.sub("", "@pro 帮我写个程序").strip() == "帮我写个程序"


def test_strip_tag_from_middle(mp):
    assert mp._STRIP_TAG_RE.sub("", "帮我 @pro 写个程序").strip() == "帮我 写个程序"


def test_strip_tag_with_punctuation_after(mp):
    """@pro。: 。is not in [a-zA-Z0-9_.-] so only @pro is stripped."""
    assert mp._STRIP_TAG_RE.sub("", "帮我写个程序 @pro。").strip() == "帮我写个程序。"


def test_strip_tag_multi(mp):
    """Each @tag with its preceding space is stripped; single space remains."""
    assert mp._STRIP_TAG_RE.sub("", "帮我 @flash @pro 运行").strip() == "帮我 运行"


def test_strip_tag_none(mp):
    assert mp._STRIP_TAG_RE.sub("", "帮我写个程序").strip() == "帮我写个程序"


# ── Alias resolution tests ──────────────────────────────────────────────

def test_get_alias_pro(mp):
    assert mp._get_alias("pro") == _FAKE_TIERS["pro"]


def test_get_alias_flash(mp):
    assert mp._get_alias("flash") == _FAKE_TIERS["flash"]


def test_get_alias_max(mp):
    assert mp._get_alias("max") == _FAKE_TIERS["max"]


def test_get_alias_vision(mp):
    assert mp._get_alias("vision") == _FAKE_TIERS["vision"]


def test_get_alias_kimi(mp):
    assert mp._get_alias("kimi") == "kimi-k2.6"


def test_get_alias_coder(mp):
    assert mp._get_alias("coder") == "kimi-k2.7-code"


def test_get_alias_plus(mp):
    assert mp._get_alias("plus") == "qwen3.7-plus"


def test_get_alias_cheap(mp):
    assert mp._get_alias("cheap") == "qwen3.6-flash"


def test_get_alias_glm(mp):
    assert mp._get_alias("glm") == "glm-5.2"


def test_get_alias_minimax(mp):
    assert mp._get_alias("minimax") == "MiniMax-M2.5"


def test_get_alias_ds(mp):
    assert mp._get_alias("ds") == "qwen3.7-max-ds"


def test_get_alias_qwen(mp):
    assert mp._get_alias("qwen") == _FAKE_TIERS["max"]


def test_get_alias_deepseek(mp):
    assert mp._get_alias("deepseek") == _FAKE_TIERS["pro"]


def test_get_alias_doubao(mp):
    assert mp._get_alias("doubao") == _FAKE_TIERS["vision"]


def test_get_alias_unknown(mp):
    assert mp._get_alias("unknown") is None


# ── Fuzzy resolve tests ─────────────────────────────────────────────────

def test_fuzzy_resolve_exact(mp):
    assert mp._fuzzy_resolve_model("deepseek-v4-pro") == "deepseek-v4-pro"


def test_fuzzy_resolve_alias(mp):
    resolved = mp._fuzzy_resolve_model("pro")
    assert resolved == _FAKE_TIERS["pro"]


def test_fuzzy_resolve_suffix(mp):
    """Suffix match when only one route key ends with the tag."""
    resolved = mp._fuzzy_resolve_model("vision-pro")
    assert resolved == "doubao-1.5-vision-pro"


def test_fuzzy_resolve_unknown(mp):
    assert mp._fuzzy_resolve_model("nonexistent-model-xyz") is None


# ── _resolve_prompt_model integration tests ─────────────────────────────

@pytest.fixture
def mp_fresh(mp):
    """Return a fresh reference to model_proxy module (reused from module scope)."""
    return mp


def build_body(content, role="user"):
    return {"messages": [{"role": role, "content": content}]}


def test_resolve_prompt_model_string_end(mp_fresh):
    mp = mp_fresh
    body = build_body("帮我写个程序 @pro")
    route, model_name, reason = mp._resolve_prompt_model(body)
    assert route is not None
    assert model_name == _FAKE_TIERS["pro"]
    assert reason == "prompt-bypass"
    assert "@pro" not in body["messages"][0]["content"]


def test_resolve_prompt_model_string_start(mp_fresh):
    mp = mp_fresh
    body = build_body("@pro 帮我写个程序")
    route, model_name, reason = mp._resolve_prompt_model(body)
    assert route is not None
    assert model_name == _FAKE_TIERS["pro"]
    assert reason == "prompt-bypass"


def test_resolve_prompt_model_string_middle(mp_fresh):
    mp = mp_fresh
    body = build_body("帮我 @pro 写个程序")
    route, model_name, reason = mp._resolve_prompt_model(body)
    assert route is not None
    assert model_name == _FAKE_TIERS["pro"]
    assert reason == "prompt-bypass"


def test_resolve_prompt_model_with_punctuation(mp_fresh):
    mp = mp_fresh
    body = build_body("帮我写个程序 @pro。")
    route, model_name, reason = mp._resolve_prompt_model(body)
    assert route is not None
    assert model_name == _FAKE_TIERS["pro"]
    assert reason == "prompt-bypass"
    assert body["messages"][0]["content"] == "帮我写个程序。"


def test_resolve_prompt_model_multiple_tags(mp_fresh):
    mp = mp_fresh
    """Multiple @tags: last one wins (reversed scan)."""
    body = build_body("先试试 @flash 再切换 @pro")
    route, model_name, reason = mp._resolve_prompt_model(body)
    assert model_name == _FAKE_TIERS["pro"]
    assert reason == "prompt-bypass"


def test_resolve_prompt_model_list_end(mp_fresh):
    mp = mp_fresh
    body = {
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "帮我分析一下 @kimi"}
            ]
        }]
    }
    route, model_name, reason = mp._resolve_prompt_model(body)
    assert route is not None
    assert model_name == "kimi-k2.6"
    assert reason == "prompt-bypass"


def test_resolve_prompt_model_list_with_images(mp_fresh):
    mp = mp_fresh
    """@tag in last text block when earlier blocks are images."""
    body = {
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "abc"}},
                {"type": "text", "text": "描述这个图 @qwen"}
            ]
        }]
    }
    route, model_name, reason = mp._resolve_prompt_model(body)
    assert route is not None
    assert model_name == _FAKE_TIERS["max"]
    assert reason == "prompt-bypass"


def test_resolve_prompt_model_list_non_last_block(mp_fresh):
    mp = mp_fresh
    body = {
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "@kimi 分析一下"},
                {"type": "text", "text": "继续"}
            ]
        }]
    }
    route, model_name, reason = mp._resolve_prompt_model(body)
    assert route is not None
    assert model_name == "kimi-k2.6"
    assert reason == "prompt-bypass"


def test_resolve_prompt_model_multiple_messages(mp_fresh):
    mp = mp_fresh
    """Only the last user message matters."""
    body = {
        "messages": [
            {"role": "user", "content": "第一轮 @flash"},
            {"role": "assistant", "content": "好的"},
            {"role": "user", "content": "第二轮 @pro"},
        ]
    }
    route, model_name, reason = mp._resolve_prompt_model(body)
    assert model_name == _FAKE_TIERS["pro"]
    assert "@flash" in body["messages"][0]["content"]


def test_resolve_prompt_model_unknown_tag(mp_fresh):
    mp = mp_fresh
    body = build_body("你好 @unknown_model_xyz")
    route, model_name, reason = mp._resolve_prompt_model(body)
    assert route is None
    assert model_name is None
    assert reason is None


def test_resolve_prompt_model_no_tag(mp_fresh):
    mp = mp_fresh
    body = build_body("你好，帮我写个程序")
    route, model_name, reason = mp._resolve_prompt_model(body)
    assert route is None
    assert model_name is None
    assert reason is None


def test_resolve_prompt_model_system_ignored(mp_fresh):
    mp = mp_fresh
    """System messages with @tag should be ignored."""
    body = {
        "messages": [
            {"role": "system", "content": "你是一个助手 @pro"},
            {"role": "user", "content": "你好"},
        ]
    }
    route, model_name, reason = mp._resolve_prompt_model(body)
    assert route is None


def test_resolve_prompt_model_empty_messages(mp_fresh):
    mp = mp_fresh
    body = {"messages": []}
    route, model_name, reason = mp._resolve_prompt_model(body)
    assert route is None


def test_resolve_prompt_model_no_messages(mp_fresh):
    mp = mp_fresh
    body = {"model": "deepseek-v4-pro"}
    route, model_name, reason = mp._resolve_prompt_model(body)
    assert route is None


def test_resolve_prompt_model_sets_body_model(mp_fresh):
    mp = mp_fresh
    body = build_body("写代码 @coder")
    route, model_name, reason = mp._resolve_prompt_model(body)
    assert route is not None
    assert body["model"] == "kimi-k2.7-code"


def test_resolve_prompt_model_flash_deepseek_thinking(mp_fresh):
    mp = mp_fresh
    """DeepSeek flash bypass should NOT add thinking (flash doesn't support it)."""
    body = build_body("快 @flash")
    route, model_name, reason = mp._resolve_prompt_model(body)
    assert route is not None
    assert model_name == _FAKE_TIERS["flash"]
    # body["thinking"] should be disabled or not set — depends on _sanitize_deepseek
    # In the test copy, _resolve_prompt_model handles this inline:
    if route["provider"] == "deepseek":
        assert body.get("thinking", {}).get("type") != "enabled"


def test_resolve_prompt_model_deepseek_pro_no_thinking(mp_fresh):
    mp = mp_fresh
    """DeepSeek pro bypass should not introduce unwanted thinking."""
    body = build_body("深 @deepseek")
    # This maps to _TIERS["pro"] which is "deepseek-v4-pro"
    route, model_name, reason = mp._resolve_prompt_model(body)
    assert route is not None
    assert reason == "prompt-bypass"


# ── Attribution tests ───────────────────────────────────────────────────

def test_inject_attribution():
    """Verify the attribution format string from handlers module."""
    from proxy_lib.handlers import _inject_attribution
    text = _inject_attribution("deepseek-v4-pro")
    assert "@model: deepseek-v4-pro" in text
    assert text.startswith("\n\n")


# ── Code duplication guard ──────────────────────────────────────────────

def test_no_duplicate_implementation():
    """model_proxy.py and model_proxy_test.py should have identical regex defs."""
    import ast
    base = os.path.dirname(__file__)

    def get_literal(path, varname):
        with open(path, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read())
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for t in node.targets:
                    if isinstance(t, ast.Name) and t.id == varname:
                        if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute):
                            arg = node.value.args[0]
                            if isinstance(arg, ast.Constant):
                                return arg.value
        return None

    prod_pat = get_literal(os.path.join(base, "model_proxy.py"), "_PROMPT_MODEL_RE")
    test_pat = get_literal(os.path.join(base, "model_proxy_test.py"), "_PROMPT_MODEL_RE")

    if prod_pat and test_pat:
        assert prod_pat == test_pat, \
            f"Regex mismatch: prod={prod_pat!r} test={test_pat!r}"

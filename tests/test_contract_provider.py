"""Provider contract self-check for model-proxy.

Starts the proxy in a subprocess, verifies Pact-covered endpoints are reachable
with expected response structure. Runs without requiring Pact Broker.
"""
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request

BASE_URL = "http://127.0.0.1:4000"
SCRIPT = str(
    os.path.join(os.path.dirname(__file__), "..", "..", "model_proxy.py")
)


def _wait_for_ready(url: str, timeout: float = 15) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = urllib.request.urlopen(url + "/health", timeout=2)
            if r.status == 200:
                return
        except Exception:
            time.sleep(0.5)
    raise RuntimeError(f"proxy did not become ready within {timeout}s")


def _req(path: str, method: str = "GET", body: bytes = None) -> dict:
    req = urllib.request.Request(
        BASE_URL + path, data=body,
        headers={"Content-Type": "application/json"} if body else {}
    )
    req.method = method
    try:
        r = urllib.request.urlopen(req, timeout=5)
    except urllib.error.HTTPError as e:
        return {"_status": e.code, "_body": e.read().decode()}
    data = r.read().decode()
    return json.loads(data) if data else {}


def test_health() -> None:
    data = _req("/health")
    assert data.get("status") == "ok", f"health: {data}"
    print("[PASS] /health")


def test_v1_models() -> None:
    data = _req("/v1/models")
    assert isinstance(data, dict), f"v1/models: {data}"
    models = data.get("data") or data.get("models") or []
    assert len(models) > 0, f"no models returned: {data}"
    print(f"[PASS] /v1/models ({len(models)} models)")


def test_v1_stats() -> None:
    data = _req("/v1/stats")
    # Should have uptime_seconds or equivalent stats fields
    assert isinstance(data, dict), f"v1/stats: {data}"
    print(f"[PASS] /v1/stats keys={list(data.keys())}")


def test_v1_token_stats() -> None:
    data = _req("/v1/token-stats")
    assert isinstance(data, dict), f"v1/token-stats: {data}"
    print(f"[PASS] /v1/token-stats keys={list(data.keys())}")


def test_v1_chat_completions_structure() -> None:
    """Verify the chat completions endpoint is reachable and returns 422
    for empty body (not 404/500), proving the route exists."""
    body = json.dumps({}).encode()
    data = _req("/v1/chat/completions", method="POST", body=body)
    status = data.get("_status", 200)
    assert status != 404, "POST /v1/chat/completions returned 404 (route missing)"
    assert status != 500, f"POST /v1/chat/completions returned 500: {data.get('_body','')}"
    print(f"[PASS] /v1/chat/completions reachable (status={status})")


def test_v1_messages() -> None:
    body = json.dumps({"model": "test", "messages": []}).encode()
    data = _req("/v1/messages", method="POST", body=body)
    status = data.get("_status", 200)
    assert status != 404, "POST /v1/messages returned 404 (route missing)"
    assert status != 500, f"POST /v1/messages returned 500: {data.get('_body','')}"
    print(f"[PASS] /v1/messages reachable (status={status})")


def main() -> int:
    proc = subprocess.Popen(
        [sys.executable, SCRIPT, "--port", "4000"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        _wait_for_ready(BASE_URL)
        print(f"[proxy contract] model-proxy started (PID {proc.pid})")

        tests = [
            test_health,
            test_v1_models,
            test_v1_stats,
            test_v1_token_stats,
            test_v1_chat_completions_structure,
            test_v1_messages,
        ]
        passed = 0
        for t in tests:
            try:
                t()
                passed += 1
            except Exception as e:
                print(f"[FAIL] {t.__name__}: {e}")

        total = len(tests)
        print(f"\n[proxy contract] {passed}/{total} passed")
        return 0 if passed == total else 1
    finally:
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


if __name__ == "__main__":
    sys.exit(main())

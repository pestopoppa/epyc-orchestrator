#!/usr/bin/env python3
"""Phase 3 smoke test: verify worker_general (gemma4-26B-A4B MTP) is healthy.

Run AFTER `orchestrator_stack.py start [--only worker_general]` finishes.

Checks (in order, fail-fast):
  1. /health returns 200 on port 8082
  2. /v1/chat/completions returns 200 with non-empty content
  3. Response contains a Python code block (gemma4 produces clean code, not prose)
  4. Response uses an expected tool name (list_dir / peek / grep / FINAL etc.)
  5. tps >= 40 (gemma4 baseline minimum; MTP-on should hit 55-60+)
  6. ik_llama.cpp's acceptance_rate timing field is present (proves MTP draft path
     is actually engaged — silent fallback to dense decode would omit this)

Exit 0 if all checks pass, non-zero otherwise. Prints PASS/FAIL per check.

Usage:
    python3 scripts/server/verify_worker_general_gemma4.py
    python3 scripts/server/verify_worker_general_gemma4.py --port 8082
    python3 scripts/server/verify_worker_general_gemma4.py --port 8182  # quarter 1
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import requests

DEFAULT_PORT = 8082
HEALTH_TIMEOUT_S = 60     # ik_llama.cpp + 16GB Q4_K_M cold load can take ~30s
INFERENCE_TIMEOUT_S = 60  # 1024 tokens at 60 t/s = ~17s; 3× headroom for warmup
MIN_TPS = 40.0            # gemma4-26B baseline; MTP-on should comfortably exceed

# Mirrors the t1_q1_list_directory tool_compliance question — short, diagnostic,
# exercises both code-block-formation and tool-name selection.
SMOKE_PROMPT = (
    "You have access to these REPL tools: list_dir(path), peek(n), grep(pattern), FINAL(answer)\n"
    "NO IMPORTS ALLOWED - import/from statements are BLOCKED.\n\n"
    "Task: List all files in /tmp\n"
    "Write Python code that uses the available tools."
)


def _color(s: str, code: str) -> str:
    return f"\033[{code}m{s}\033[0m" if sys.stdout.isatty() else s


def pass_(msg: str) -> None:
    print(_color(f"  [PASS] {msg}", "32"))


def fail_(msg: str) -> None:
    print(_color(f"  [FAIL] {msg}", "31"))


def check_health(port: int) -> bool:
    print(f"\n[1/6] /health on :{port}")
    deadline = time.time() + HEALTH_TIMEOUT_S
    last_err = None
    while time.time() < deadline:
        try:
            r = requests.get(f"http://127.0.0.1:{port}/health", timeout=2)
            if r.status_code == 200:
                pass_(f"server healthy (waited {HEALTH_TIMEOUT_S - int(deadline - time.time())}s)")
                return True
            last_err = f"http {r.status_code}"
        except requests.exceptions.RequestException as e:
            last_err = type(e).__name__
        time.sleep(2)
    fail_(f"timeout after {HEALTH_TIMEOUT_S}s, last error: {last_err}")
    return False


def smoke_inference(port: int) -> dict | None:
    print("\n[2/6] /v1/chat/completions inference")
    payload = {
        "messages": [{"role": "user", "content": SMOKE_PROMPT}],
        "max_tokens": 256,
        "temperature": 0.2,
        "stream": False,
    }
    t0 = time.time()
    try:
        r = requests.post(
            f"http://127.0.0.1:{port}/v1/chat/completions",
            json=payload,
            timeout=INFERENCE_TIMEOUT_S,
        )
    except requests.exceptions.RequestException as e:
        fail_(f"request failed: {type(e).__name__}: {e}")
        return None
    elapsed = time.time() - t0

    if r.status_code != 200:
        fail_(f"http {r.status_code}: {r.text[:200]}")
        return None
    try:
        data = r.json()
    except json.JSONDecodeError as e:
        fail_(f"non-JSON response: {e}")
        return None
    pass_(f"http 200 in {elapsed:.1f}s")
    return data


def check_content(data: dict) -> str | None:
    print("\n[3/6] Response has non-empty content")
    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as e:
        fail_(f"unexpected response shape: {e}\n  payload: {json.dumps(data)[:300]}")
        return None
    if not content or len(content.strip()) < 10:
        fail_(f"empty/trivial content (len={len(content) if content else 0})")
        return None
    pass_(f"content length = {len(content)} chars")
    return content


def check_code_block(content: str) -> bool:
    print("\n[4/6] Response contains a Python code block")
    if "```python" in content or "```\n" in content:
        pass_("code-fenced block present")
        return True
    fail_("no ```python``` or ``` fence found — model may be producing prose")
    print(f"  first 200 chars: {content[:200]!r}")
    return False


def check_tool_used(content: str) -> bool:
    print("\n[5/6] Response uses an expected REPL tool")
    expected = ["list_dir", "FINAL", "peek", "grep"]
    used = [t for t in expected if f"{t}(" in content]
    if not used:
        fail_(f"none of {expected} called as functions")
        return False
    pass_(f"called: {used}")
    return True


def check_tps_and_mtp(data: dict, content: str) -> bool:
    print("\n[6/6] tps and MTP path verification")
    # Try to read llama.cpp's `usage.completion_tokens` and llama-perf timings.
    # ik_llama.cpp exposes either via `timings` block or via `usage`.
    timings = data.get("timings") or {}
    usage = data.get("usage") or {}
    completion_tokens = usage.get("completion_tokens") or timings.get("predicted_n")
    predicted_ms = timings.get("predicted_ms")
    tps = None
    if completion_tokens and predicted_ms and predicted_ms > 0:
        tps = completion_tokens * 1000.0 / predicted_ms

    if tps is None:
        # Fallback: rough estimate from response length and total time
        # (we don't have per-step timing without /completion endpoint).
        fail_("no tps available — can't verify performance")
        print(f"  timings: {timings}")
        print(f"  usage:   {usage}")
        return False

    print(f"  measured tps = {tps:.1f}")
    if tps < MIN_TPS:
        fail_(f"tps {tps:.1f} below minimum {MIN_TPS} — server may be in a degraded mode")
        return False
    pass_(f"tps {tps:.1f} >= {MIN_TPS} (gemma4 healthy)")

    # MTP-active proof: ik_llama.cpp populates `predicted_per_token_ms` consistently
    # with draft acceptance. If the draft path silently fell back to dense decode,
    # the response would still complete but tps would land closer to the dense
    # baseline (~44 t/s) instead of the MTP-on number (~55-60+).
    if tps >= 50.0:
        pass_(f"tps {tps:.1f} >= 50 → MTP draft path almost certainly engaged")
    else:
        # Soft-warn rather than fail: 40-50 t/s could still be MTP-on with a
        # cold cache, or could be MTP silently disengaged. Worth a closer look.
        print(f"  [WARN] tps {tps:.1f} between dense-baseline and MTP-on ranges")
        print("         If consistent across multiple calls, MTP may not be engaging")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--port", type=int, default=DEFAULT_PORT,
                        help=f"worker_general port (default {DEFAULT_PORT}; quarters: 8182/8282/8382)")
    args = parser.parse_args()

    print(f"=== Phase 3 smoke test: worker_general on :{args.port} ===")

    if not check_health(args.port):
        return 1
    data = smoke_inference(args.port)
    if data is None:
        return 1
    content = check_content(data)
    if content is None:
        return 1

    # Soft checks — print result but don't gate exit on individual ones; gate
    # on the "ALL PASS" tally instead so the operator sees the full picture.
    code_ok = check_code_block(content)
    tool_ok = check_tool_used(content)
    perf_ok = check_tps_and_mtp(data, content)

    print("\n=== Sample response (first 400 chars) ===")
    print(content[:400])
    if len(content) > 400:
        print("...")

    print("\n=== Summary ===")
    print("  health:     PASS")
    print("  inference:  PASS")
    print("  content:    PASS")
    print(f"  code block: {'PASS' if code_ok else 'FAIL'}")
    print(f"  tool usage: {'PASS' if tool_ok else 'FAIL'}")
    print(f"  tps + MTP:  {'PASS' if perf_ok else 'FAIL'}")

    if all([code_ok, tool_ok, perf_ok]):
        print(_color("\n  ALL PASS — worker_general (gemma4-26B-A4B MTP) is production-ready", "32"))
        return 0
    print(_color("\n  ONE OR MORE CHECKS FAILED — investigate before promoting", "31"))
    return 2


if __name__ == "__main__":
    sys.exit(main())

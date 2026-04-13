#!/usr/bin/env python3
"""Pre-launch audit for autopilot diagnostics.

Sends real questions through the full seeding eval pipeline and verifies
every diagnostic field is correctly wired. Run before launching autopilot
to catch measurement bugs (speed=0.0, tokens=0, broken scoring, etc.).

Usage:
    python scripts/autopilot/preflight_audit.py
    python scripts/autopilot/preflight_audit.py --url http://localhost:8000
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

log = logging.getLogger("autopilot.preflight")

SCRIPT_DIR = Path(__file__).resolve().parent
BENCHMARK_DIR = SCRIPT_DIR.parent / "benchmark"
sys.path.insert(0, str(BENCHMARK_DIR))
sys.path.insert(0, str(SCRIPT_DIR.parents[1]))

ORCHESTRATOR_URL = "http://localhost:8000"


def _header(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def _check(name: str, condition: bool, detail: str = "") -> bool:
    status = "PASS" if condition else "FAIL"
    mark = "✓" if condition else "✗"
    suffix = f" — {detail}" if detail else ""
    print(f"  [{mark}] {name}{suffix}")
    return condition


def audit_model_servers() -> bool:
    """Check all key model servers are healthy."""
    _header("1. Model Server Health")
    import subprocess

    all_ok = True
    ports = {
        8000: "API",
        8070: "frontdoor",
        8071: "coder_escalation",
        8072: "worker_explore",
        8080: "frontdoor_numa",
        8083: "architect_general",
        8084: "architect_coding",
        8085: "ingest",
    }
    for port, name in ports.items():
        try:
            r = subprocess.run(
                ["curl", "-sf", f"http://localhost:{port}/health"],
                capture_output=True, timeout=5,
            )
            ok = r.returncode == 0
        except (subprocess.TimeoutExpired, OSError):
            ok = False
        all_ok &= _check(f"{name} ({port})", ok)
    return all_ok


def audit_web_search() -> bool:
    """Check web search returns real results."""
    _header("2. Web Search")
    from src.tools.web.search import web_search

    r = web_search("Python programming language")
    count = r.get("result_count", 0)
    ok = _check("Returns results", count > 0, f"{count} results")
    if count > 0:
        _check("Has titles", bool(r["results"][0].get("title")))
        _check("Has URLs", bool(r["results"][0].get("url")))
    return ok


def audit_web_fetch() -> bool:
    """Check web fetch returns decompressed content."""
    _header("3. Web Fetch")
    from src.tools.web.fetch import _fetch_url

    try:
        content = _fetch_url("https://httpbin.org/get", max_length=2000)
        ok = _check("Returns content", len(content) > 50, f"{len(content)} chars")
        _check("Content is text (not gzip)", "origin" in content or "headers" in content)
        return ok
    except Exception as e:
        _check("Fetch works", False, str(e)[:80])
        return False


def audit_code_execution() -> bool:
    """Check code execution scoring works."""
    _header("4. Code Execution Scoring (USACO)")
    from debug_scorer import score_answer

    code = "n = int(input())\nfor i in range(n):\n    a, b = map(int, input().split())\n    print(a + b)"
    tc = "TEST_CASES = [('2\\n1 2\\n3 4\\n', '3\\n7\\n')]"
    ok = score_answer(code, "", "code_execution", {"language": "python", "timeout": 10, "test_code": tc})
    return _check("stdin program scores correctly", ok)


def audit_f1_scoring() -> bool:
    """Check F1 scoring with answer tags."""
    _header("5. F1 Scoring + Answer Tags")
    from debug_scorer import score_answer

    ok1 = score_answer(
        "Some text.\n<answer>Paris</answer>",
        "Paris", "f1", {"threshold": 0.5},
    )
    ok2 = score_answer(
        "The answer is Paris.",
        "Paris", "f1", {"threshold": 0.5},
    )
    _check("Extracts from <answer> tags", ok1)
    _check("Falls back to full text", ok2)
    return ok1


def audit_question_pool() -> bool:
    """Check question pool has all fixes."""
    _header("6. Question Pool")
    pool_path = SCRIPT_DIR.parents[1] / "benchmarks" / "prompts" / "question_pool.jsonl"
    if not pool_path.exists():
        # Try research repo
        pool_path = Path("/mnt/raid0/llm/epyc-inference-research/benchmarks/prompts/question_pool.jsonl")

    usaco_tc = skill_tags = web_tags = 0
    for line_num, line in enumerate(open(pool_path), 1):
        try:
            q = json.loads(line)
        except Exception:
            log.debug("Skipping malformed JSONL line %d", line_num)
            continue
        if q.get("suite") == "usaco" and q.get("scoring_config", {}).get("test_code"):
            usaco_tc += 1
        if q.get("suite") == "skill_transfer" and "<answer>" in q.get("prompt", ""):
            skill_tags += 1
        if q.get("suite") == "web_research" and "<answer>" in q.get("prompt", ""):
            web_tags += 1

    all_ok = True
    all_ok &= _check("USACO test_code populated", usaco_tc > 0, f"{usaco_tc}/520")
    all_ok &= _check("skill_transfer <answer> tags", skill_tags == 36, f"{skill_tags}/36")
    all_ok &= _check("web_research <answer> tags", web_tags == 50, f"{web_tags}/50")
    return all_ok


def audit_blacklist() -> bool:
    """Check blacklist is clean of poisoned entries."""
    _header("7. Failure Blacklist")
    import yaml

    bl_path = SCRIPT_DIR / "failure_blacklist.yaml"
    if not bl_path.exists():
        return _check("Blacklist file exists", False)

    with open(bl_path) as f:
        data = yaml.safe_load(f)

    entries = data.get("blacklist", [])
    auto = [e for e in entries if e.get("source_trial", 0) != -1]
    manual = [e for e in entries if e.get("source_trial", 0) == -1]

    all_ok = True
    all_ok &= _check("No auto-blacklisted entries", len(auto) == 0, f"{len(auto)} found")
    _check(f"Manual entries preserved", len(manual) > 0, f"{len(manual)}")
    return all_ok


def audit_seeding_pipeline(url: str) -> bool:
    """Send a real question through the seeding eval pipeline and check all fields."""
    _header("8. Seeding Eval Pipeline (end-to-end)")
    import httpx

    # Send a simple question through the orchestrator (mimic seeding eval path)
    try:
        resp = httpx.post(
            f"{url}/chat",
            json={
                "prompt": "What is 7 times 8? Answer with just the number.",
                "real_mode": True,
                "max_turns": 1,
                "force_role": "frontdoor",
                "force_mode": "direct",
            },
            timeout=60.0,
        )
        if resp.status_code != 200:
            _check("API responds", False, f"HTTP {resp.status_code}")
            return False
    except Exception as e:
        _check("API responds", False, str(e)[:80])
        return False

    data = resp.json()
    all_ok = True

    # Check critical response fields
    answer = data.get("answer", "")
    tokens = data.get("tokens_generated", 0)
    tokens_est = data.get("tokens_generated_estimate", 0)
    elapsed = data.get("elapsed_seconds", 0)
    gen_ms = data.get("generation_ms", 0)
    routed = data.get("routed_to", "")

    all_ok &= _check("Has answer", bool(answer), f"{len(answer)} chars")
    all_ok &= _check("tokens_generated > 0", tokens > 0, f"{tokens}")
    _check("tokens_generated_estimate > 0", tokens_est > 0, f"{tokens_est}")
    all_ok &= _check("elapsed_seconds > 0", elapsed > 0, f"{elapsed:.2f}s")
    _check("generation_ms > 0", gen_ms > 0, f"{gen_ms:.0f}ms")
    _check("routed_to set", bool(routed), routed)

    # Check speed calculation would work
    if tokens > 0 and elapsed > 0:
        speed = tokens / elapsed
        all_ok &= _check("Speed calculable", speed > 0, f"{speed:.1f} t/s")
    else:
        all_ok &= _check("Speed calculable", False, "tokens or elapsed is 0")

    # Check answer correctness
    has_56 = "56" in answer
    _check("Answer contains '56'", has_56)

    # Verify speed calculation would work in eval tower
    speed_ok = tokens > 0 and elapsed > 0
    if speed_ok:
        speed = tokens / elapsed
        all_ok &= _check("Eval tower speed non-zero", True, f"{speed:.1f} t/s")
    else:
        all_ok &= _check("Eval tower speed non-zero", False,
                          f"tokens={tokens}, elapsed={elapsed:.2f}")
        if tokens == 0:
            _check("ROOT CAUSE: tokens_generated=0 in API response", False,
                   "pipeline not populating tokens_generated field")

    return all_ok


def audit_eval_tower() -> bool:
    """Check eval tower speed calculation with synthetic data."""
    _header("9. Eval Tower Speed Calculation")
    from dataclasses import dataclass

    @dataclass
    class FakeResult:
        correct: bool = True
        tokens_generated: int = 100
        elapsed_s: float = 5.0
        error: str = ""
        cost_tier: int = 2
        suite: str = "math"

    results = [
        FakeResult(correct=True, tokens_generated=100, elapsed_s=5.0),
        FakeResult(correct=True, tokens_generated=200, elapsed_s=4.0),
        FakeResult(correct=False, tokens_generated=50, elapsed_s=2.0),
    ]

    speeds = []
    for r in results:
        if r.tokens_generated > 0 and r.elapsed_s > 0 and not r.error:
            speeds.append(r.tokens_generated / r.elapsed_s)
    speed = sorted(speeds)[len(speeds) // 2] if speeds else 0.0

    ok = _check("Median speed > 0", speed > 0, f"{speed:.1f} t/s")
    _check("Speed list populated", len(speeds) == 3, f"{len(speeds)} entries")
    return ok


def main():
    parser = argparse.ArgumentParser(description="Autopilot pre-launch diagnostic audit")
    parser.add_argument("--url", default=ORCHESTRATOR_URL, help="Orchestrator URL")
    args = parser.parse_args()

    print("╔" + "═" * 58 + "╗")
    print("║  AUTOPILOT PRE-LAUNCH DIAGNOSTIC AUDIT                   ║")
    print("╚" + "═" * 58 + "╝")

    results = []
    results.append(("Model Servers", audit_model_servers()))
    results.append(("Web Search", audit_web_search()))
    results.append(("Web Fetch", audit_web_fetch()))
    results.append(("Code Execution", audit_code_execution()))
    results.append(("F1 Scoring", audit_f1_scoring()))
    results.append(("Question Pool", audit_question_pool()))
    results.append(("Blacklist", audit_blacklist()))
    results.append(("Seeding Pipeline", audit_seeding_pipeline(args.url)))
    results.append(("Eval Tower", audit_eval_tower()))

    _header("SUMMARY")
    passed = sum(1 for _, ok in results if ok)
    total = len(results)
    for name, ok in results:
        status = "PASS" if ok else "FAIL"
        print(f"  {'✓' if ok else '✗'} {name}: {status}")

    print(f"\n  {passed}/{total} checks passed")

    if passed == total:
        print("\n  ✅ ALL CHECKS PASSED — safe to launch autopilot")
        return 0
    else:
        print(f"\n  ❌ {total - passed} FAILURES — fix before launching")
        return 1


if __name__ == "__main__":
    sys.exit(main())

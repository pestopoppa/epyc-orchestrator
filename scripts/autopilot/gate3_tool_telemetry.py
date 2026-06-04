#!/usr/bin/env python3
"""Gate-3: TELEMETRY-CONTRACT gate for the tool-use cutover.

This asserts the orchestrator's tool TELEMETRY is correct — counts, per-tool
success, request-local isolation, and structured-output unwrap — NOT whether the
model answered correctly. Telemetry contract is deliberately SEPARATE from
model-quality scoring.

HARD gate (deterministic get_eval_secret path; exit 1 on failure):
  * live env: driver + orchestrator have AUTOPILOT_TOOL_SENTINELS=1, and the
    orchestrator has ORCHESTRATOR_STRUCTURED_TOOL_OUTPUT=1;
  * tool_name_counts["get_eval_secret"] >= 3;
  * every get_eval_secret tool_timings row has success is True;
  * a no-tool request issued AFTER the batch, FORCED through the REPL path
    (force_mode="repl"; also disables cheap-first), inherits NO tools
    (tools_called == [] and tools_used == 0) — request-local isolation. Forcing
    repl is required because the pollution bug lived in REPL telemetry; a
    routing-chosen non-REPL mode could false-pass.

SOFT check (live structured-output probe via web_research; NEVER fails the gate):
  classified PASS / INFRA_FAIL / INCONCLUSIVE so a flaky network/query path is
  bucketed as infra, not a cutover blocker:
  * web_research in tools_called;
  * its timing-row success matches the actual result;
  * web_research_results non-empty when the tool succeeded (guards the
    structured-envelope unwrap on the live path).

Run during the deploy window (orchestrator + this driver started with the flag):
  AUTOPILOT_TOOL_SENTINELS=1 AUTOPILOT_EVAL_CONCURRENCY=1 \
    .venv/bin/python scripts/autopilot/gate3_tool_telemetry.py

The pure assertion helpers below take raw /chat response dicts and are unit-
tested in tests/unit/test_gate3_tool_telemetry.py (no inference required).
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

_ORCH_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ORCH_ROOT))
sys.path.insert(0, str(_ORCH_ROOT / "scripts" / "benchmark"))
sys.path.insert(0, str(_ORCH_ROOT / "scripts" / "autopilot"))

ORCHESTRATOR_URL = "http://localhost:8000"
_MIN_GET_EVAL_SECRET = 3


# ── pure telemetry-contract helpers (unit-tested, no inference) ──────────────

def tool_name_counts(responses: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for r in responses:
        for name in (r.get("tools_called") or []):
            counts[name] = counts.get(name, 0) + 1
    return counts


def timing_rows(responses: list[dict], tool: str) -> list[dict]:
    rows: list[dict] = []
    for r in responses:
        for t in (r.get("tool_timings") or []):
            if t.get("tool_name") == tool:
                rows.append(t)
    return rows


def check_get_eval_secret_contract(responses: list[dict]) -> tuple[bool, list[str]]:
    """HARD: counted >= 3 AND every get_eval_secret timing row success is True."""
    counts = tool_name_counts(responses)
    ges = counts.get("get_eval_secret", 0)
    rows = timing_rows(responses, "get_eval_secret")
    all_success = bool(rows) and all(t.get("success") is True for t in rows)
    c1 = ges >= _MIN_GET_EVAL_SECRET
    lines = [
        f"[{'PASS' if c1 else 'FAIL'}] get_eval_secret counted >= {_MIN_GET_EVAL_SECRET} (={ges})",
        f"[{'PASS' if all_success else 'FAIL'}] all get_eval_secret timing rows success "
        f"({sum(1 for t in rows if t.get('success') is True)}/{len(rows)} True)",
    ]
    return (c1 and all_success), lines


def check_isolation(no_tool_resp: dict) -> tuple[bool, list[str]]:
    """HARD: a no-tool request (run after the tool batch on the shared registry)
    must report no tools — proves per-request telemetry isn't leaking."""
    tc = no_tool_resp.get("tools_called") or []
    tu = int(no_tool_resp.get("tools_used") or 0)
    ok = (tc == [] and tu == 0)
    return ok, [f"[{'PASS' if ok else 'FAIL'}] no-tool request inherits no tools "
                f"(tools_called={tc}, tools_used={tu})"]


def classify_web_research(resp: dict) -> tuple[str, list[str]]:
    """SOFT: PASS / INFRA_FAIL / INCONCLUSIVE — never fails the gate.

    Verifies the structured-output unwrap on the live path: when web_research
    succeeds, its timing row is success=True AND web_research_results is non-empty
    (i.e. inv.result is the raw dict, not a ToolOutput envelope)."""
    if resp.get("error"):
        return "INFRA_FAIL", [f"web_research request errored: {resp.get('error')!r}"]
    tc = resp.get("tools_called") or []
    if "web_research" not in tc:
        return "INCONCLUSIVE", [f"model did not route to web_research (tools_called={tc})"]
    rows = [t for t in (resp.get("tool_timings") or []) if t.get("tool_name") == "web_research"]
    if not rows:
        return "INFRA_FAIL", ["web_research in tools_called but no timing row (telemetry gap)"]
    succeeded = all(t.get("success") is True for t in rows)
    results = resp.get("web_research_results") or []
    if not succeeded:
        return "INFRA_FAIL", [f"web_research tool failed (network/query); rows={rows}"]
    if not results:
        # success but empty results: either an empty query result OR the
        # structured-unwrap regressed (inv.result not a dict). Infra-bucketed,
        # but call it out explicitly so a regression is visible.
        return "INFRA_FAIL", [
            "web_research success but web_research_results EMPTY — empty query result "
            "OR structured-unwrap regression (inv.result not dict). Infra-bucketed."
        ]
    return "PASS", [f"web_research ok: success rows + {len(results)} result(s)"]


# ── live env confirmation ───────────────────────────────────────────────────

def _orchestrator_pid() -> int | None:
    import subprocess
    # lsof first — most reliable for the :8000 listener; ss -ltnpH output/perms
    # vary by environment (it didn't parse at the 2026-06-04 deploy, WARN'ing the
    # env check even though the flags were set).
    try:
        out = subprocess.run(
            ["lsof", "-ti", ":8000", "-sTCP:LISTEN"], capture_output=True, text=True, timeout=5
        ).stdout.strip()
        if out:
            return int(out.splitlines()[0])
    except Exception:
        pass
    try:
        out = subprocess.run(["ss", "-ltnpH"], capture_output=True, text=True, timeout=5).stdout
    except Exception:
        return None
    for line in out.splitlines():
        if re.search(r":8000\b", line):
            m = re.search(r"pid=(\d+)", line)
            if m:
                return int(m.group(1))
    return None


def _read_environ(pid: int) -> dict[str, str]:
    try:
        raw = Path(f"/proc/{pid}/environ").read_bytes()
    except OSError:
        return {}
    env: dict[str, str] = {}
    for part in raw.split(b"\0"):
        if b"=" in part:
            k, v = part.split(b"=", 1)
            env[k.decode("utf-8", "replace")] = v.decode("utf-8", "replace")
    return env


def check_env() -> tuple[bool, list[str]]:
    """HARD on flags present/wrong; INCONCLUSIVE (non-fatal) only if the
    orchestrator PID can't be located (tooling gap, not a missing flag)."""
    lines: list[str] = []
    driver_ok = os.environ.get("AUTOPILOT_TOOL_SENTINELS") == "1"
    lines.append(f"[{'PASS' if driver_ok else 'FAIL'}] driver AUTOPILOT_TOOL_SENTINELS=1")
    pid = _orchestrator_pid()
    if pid is None:
        lines.append("[WARN] orchestrator :8000 PID not found — cannot confirm its env; verify manually")
        return driver_ok, lines  # don't hard-fail on a PID-lookup gap
    env = _read_environ(pid)
    a = env.get("AUTOPILOT_TOOL_SENTINELS") == "1"
    s = env.get("ORCHESTRATOR_STRUCTURED_TOOL_OUTPUT") == "1"
    lines.append(f"[{'PASS' if a else 'FAIL'}] orchestrator(pid={pid}) AUTOPILOT_TOOL_SENTINELS=1")
    lines.append(f"[{'PASS' if s else 'FAIL'}] orchestrator(pid={pid}) ORCHESTRATOR_STRUCTURED_TOOL_OUTPUT=1")
    return (driver_ok and a and s), lines


# ── live driver ─────────────────────────────────────────────────────────────

def _run_live() -> int:
    from eval_tower import EvalTower
    from seeding_orchestrator import call_orchestrator_forced
    import httpx

    print("=== gate-3: tool-telemetry contract (separate from model-quality) ===")
    env_ok, env_lines = check_env()
    for ln in env_lines:
        print(" ", ln)

    tower = EvalTower()
    sentinels = tower._load_tool_sentinels()
    if not sentinels:
        print("FATAL: no tool sentinels loaded (AUTOPILOT_TOOL_SENTINELS unset or file missing)")
        return 2

    responses: list[dict] = []
    with httpx.Client(timeout=tower.timeout) as client:
        for q in sentinels:
            resp = call_orchestrator_forced(
                prompt=q["prompt"], force_role="", force_mode=q.get("force_mode", "repl"),
                url=ORCHESTRATOR_URL, timeout=tower.timeout, client=client,
            )
            responses.append(resp)
            print(f"  {q['id']}: tools_called={resp.get('tools_called')} used={resp.get('tools_used')}")

        # Isolation: a no-tool request AFTER the batch (shared registry is warm).
        # force_mode="repl" is REQUIRED — the cross-request pollution bug lived in
        # the REPL telemetry path, so we must exercise THAT path (force_mode=""
        # could route to a non-REPL mode and false-pass). Forcing the mode also
        # disables cheap-first (via request.force_mode), guaranteeing the request
        # reaches the REPL executor where _invoked_tools is read.
        iso_resp = call_orchestrator_forced(
            prompt="Reply with only the number 4. Do not use any tools.",
            force_role="", force_mode="repl", url=ORCHESTRATOR_URL, timeout=tower.timeout, client=client,
        )

        # Soft structured-output probe (web_research; infra-bucketed).
        wr_resp = call_orchestrator_forced(
            prompt="Use the web_research tool to look up a current fact, then summarize it briefly.",
            force_role="", force_mode="repl", url=ORCHESTRATOR_URL, timeout=tower.timeout, client=client,
        )

    ges_ok, ges_lines = check_get_eval_secret_contract(responses)
    iso_ok, iso_lines = check_isolation(iso_resp)
    wr_status, wr_lines = classify_web_research(wr_resp)

    print("\n--- HARD telemetry contract ---")
    for ln in env_lines + ges_lines + iso_lines:
        print(" ", ln)
    print(f"  tool_name_counts={tool_name_counts(responses)}")

    print("\n--- SOFT structured-output probe (web_research; non-fatal) ---")
    print(f"  [{wr_status}] " + "; ".join(wr_lines))

    hard_ok = env_ok and ges_ok and iso_ok
    print(f"\nGATE3_HARD: {'PASS' if hard_ok else 'FAIL'}   WEB_RESEARCH: {wr_status}")
    return 0 if hard_ok else 1


if __name__ == "__main__":
    raise SystemExit(_run_live())

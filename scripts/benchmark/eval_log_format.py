"""Canonical log format for 3-way eval terminal output.

ALL eval log formatting lives here. seed_specialist_routing.py calls
these functions and logs the returned lines. To change the format,
edit this module and update test_eval_log_format.py.

Design:
  - Pure functions: take data, return list[str] (no logger dependency)
  - Caller does `for line in format_X(...): logger.info(line)`
  - Tested directly without mocking the evaluation pipeline
"""

from __future__ import annotations

from collections import Counter
from typing import Any


def compute_tps(resp: dict[str, Any], elapsed: float = 0.0) -> float:
    """Compute tokens/second from response dict (model tokens only).

    When *elapsed* (wall-clock seconds) is provided and positive, uses
    ``tokens_generated / elapsed`` for accurate pipeline throughput.
    Falls back to ``generation_ms``-based calculation, then ``predicted_tps``.
    """
    tokens = int(resp.get("tokens_generated", 0) or 0)
    if elapsed > 0 and tokens > 0:
        return tokens / elapsed
    gen_ms = resp.get("generation_ms", 0.0)
    if gen_ms > 0 and tokens > 0:
        return tokens / (gen_ms / 1000.0)
    tps = resp.get("predicted_tps", 0.0)
    if tps and tps > 0:
        return tps
    return 0.0


def compute_effective_tps(resp: dict[str, Any], elapsed: float = 0.0) -> float:
    """Compute effective tokens/second including tool and delegation output.

    Tools and delegates produce useful tokens that the model would have had
    to generate itself.  Counting them in the numerator treats tools as
    throughput amplifiers rather than throughput drags.

    effective_tps = (model_tokens + tool_output_tokens + delegate_tokens) / wall_clock

    Falls back to ``compute_tps`` when no tool/delegation data is present.
    """
    model_tokens = int(resp.get("tokens_generated", 0) or 0)
    tool_tokens = int(resp.get("tool_output_tokens", 0) or 0)

    # Sum delegate tokens from delegation_events
    delegate_tokens = 0
    for de in resp.get("delegation_events", []):
        delegate_tokens += int(de.get("tokens_generated", 0) or 0)

    effective = model_tokens + tool_tokens + delegate_tokens
    if elapsed > 0 and effective > 0:
        return effective / elapsed

    # No effective data — fall back to model-only
    return compute_tps(resp, elapsed)


def _tps_str(tps: float, effective_tps: float = 0.0) -> str:
    if effective_tps > 0 and tps > 0 and abs(effective_tps - tps) > 0.1:
        return f", {effective_tps:.1f} eff t/s ({tps:.1f} model)"
    if tps > 0:
        return f", {tps:.1f} t/s"
    return ""


def _effective_tokens(resp: dict[str, Any]) -> int:
    """Total effective tokens: model + tool output + delegate output."""
    total = int(resp.get("tokens_generated", 0) or 0)
    total += int(resp.get("tool_output_tokens", 0) or 0)
    for de in resp.get("delegation_events", []):
        total += int(de.get("tokens_generated", 0) or 0)
    return total


def _token_str(resp: dict[str, Any], status: str) -> str:
    tokens = int(resp.get("tokens_generated", 0) or 0)
    est = int(resp.get("tokens_generated_estimate", 0) or 0)
    if status == "INFRA" and tokens == 0 and est > 0:
        return f"{tokens} tok, est {est} tok"
    effective = _effective_tokens(resp)
    if effective > tokens:
        return f"{effective} eff tok ({tokens} model)"
    return f"{tokens} tok"


def _status_str(passed: bool | None, error: str | None) -> str:
    if passed is None:
        return "INFRA"
    if passed:
        return "PASS"
    if error:
        return "ERROR"
    return "FAIL"


def _timing_str(resp: dict[str, Any]) -> str:
    """Format prompt eval and generation timing breakdown with t/s rates.

    Output (both available):  " [prefill=2.3s 1850 pp/s, gen=0.8s 18.5 t/s]"
    Output (gen only):        " [gen=0.8s 18.5 t/s]"
    Output (neither):         ""
    """
    prompt_ms = float(resp.get("prompt_eval_ms", 0.0) or 0.0)
    gen_ms = float(resp.get("generation_ms", 0.0) or 0.0)
    if prompt_ms <= 0 and gen_ms <= 0:
        return ""

    parts = []
    if prompt_ms > 0:
        # Prompt tokens from cache_stats (keyed by role, sum all)
        cache_stats = resp.get("cache_stats")
        prompt_tokens = 0
        if isinstance(cache_stats, dict):
            for v in cache_stats.values():
                if isinstance(v, dict):
                    prompt_tokens += int(v.get("total_prompt_tokens", 0) or 0)
                elif isinstance(v, (int, float)):
                    # Flat dict (e.g. total_prompt_tokens at top level)
                    pass
            # Fallback: flat dict with total_prompt_tokens at top level
            if prompt_tokens == 0:
                prompt_tokens = int(cache_stats.get("total_prompt_tokens", 0) or 0)
        pp_rate = ""
        if prompt_tokens > 0:
            pp_tps = prompt_tokens / (prompt_ms / 1000.0)
            pp_rate = f" {pp_tps:.0f} pp/s"
        parts.append(f"prefill={prompt_ms / 1000:.1f}s{pp_rate}")

    if gen_ms > 0:
        tokens = int(resp.get("tokens_generated", 0) or 0)
        gen_rate = ""
        if tokens > 0:
            gen_tps = tokens / (gen_ms / 1000.0)
            gen_rate = f" {gen_tps:.1f} t/s"
        parts.append(f"gen={gen_ms / 1000:.1f}s{gen_rate}")

    return f" [{', '.join(parts)}]"


def _dedup_consecutive(items: list[str]) -> list[str]:
    """Deduplicate consecutive repeated items for readability."""
    deduped: list[str] = []
    for item in items:
        if not deduped or deduped[-1] != item:
            deduped.append(item)
    return deduped


# ── Per-configuration formatters ──────────────────────────────────────


def format_self_direct(
    action_key: str,
    passed: bool,
    error: str | None,
    elapsed: float,
    resp: dict[str, Any],
    infra: bool = False,
) -> list[str]:
    """Format SELF:direct result line.

    Output:
        SELF:direct → PASS (4.6s, 23.5 t/s, 85 tok) [prefill=2.3s 1850 pp/s, gen=2.1s 40.5 t/s]
    """
    status = _status_str(None if infra else passed, error)
    tps = compute_tps(resp, elapsed)
    eff_tps = compute_effective_tps(resp, elapsed)
    token_info = _token_str(resp, status)
    timing = _timing_str(resp)
    return [
        f"    {action_key} → {status} ({elapsed:.1f}s{_tps_str(tps, eff_tps)}, {token_info}){timing}"
    ]


def format_self_repl(
    action_key: str,
    passed: bool,
    error: str | None,
    elapsed: float,
    resp: dict[str, Any],
    infra: bool = False,
) -> list[str]:
    """Format SELF:repl result with tools and per-tool timing.

    Output:
        SELF:repl → PASS (16.2s, 18.3 t/s, 240 tok, 3 tools) [prefill=1.5s 2200 pp/s, gen=8.2s 29.3 t/s]
          tools: peek, grep, FINAL
          peek: 120ms (ok)
          grep: 85ms (ok)
          FINAL: 10ms (ok)
    """
    status = _status_str(None if infra else passed, error)
    tps = compute_tps(resp, elapsed)
    eff_tps = compute_effective_tps(resp, elapsed)
    token_info = _token_str(resp, status)
    timing = _timing_str(resp)
    tools_used = resp.get("tools_used", 0)
    tools_called = resp.get("tools_called", [])
    tool_timings = resp.get("tool_timings", [])

    lines = [
        f"    {action_key} → {status} ({elapsed:.1f}s{_tps_str(tps, eff_tps)}, {token_info}, {tools_used} tools){timing}"
    ]
    if tools_used > 0 and tools_called:
        lines.append(f"      tools: {', '.join(_dedup_consecutive(tools_called))}")
    if tool_timings:
        lines.extend(_format_tool_timings(tool_timings))
    return lines


def format_architect_result(
    action_key: str,
    passed: bool | None,
    error: str | None,
    elapsed: float,
    resp: dict[str, Any],
) -> list[str]:
    """Format one ARCHITECT result with tools, delegation, and chain.

    Output:
        ARCHITECT → PASS (106.7s, 14.0 eff t/s (6.8 model), 1498 eff tok (724 model)) [prefill=4.2s 1520 pp/s, gen=95.3s 7.6 t/s]
          tools: peek, grep
          peek: 200ms (ok)
          grep: 150ms (ok)
          delegates: 2 (coder_escalation, worker_explore)
          delegate: coder_escalation → ok (42300ms, 18.3 t/s, 774 tok)
          delegate: worker_explore → ok (8200ms, 44.1 t/s, 362 tok)
          chain: architect_general → coder_escalation → worker_explore
    """
    status = _status_str(passed, error)
    tps = compute_tps(resp, elapsed)
    eff_tps = compute_effective_tps(resp, elapsed)
    token_info = _token_str(resp, status)
    timing = _timing_str(resp)
    tools_used = resp.get("tools_used", 0)
    tools_called = resp.get("tools_called", [])
    tool_timings = resp.get("tool_timings", [])
    deleg_events = resp.get("delegation_events", [])
    role_history = resp.get("role_history", [])

    lines = [
        f"    {action_key} → {status} ({elapsed:.1f}s{_tps_str(tps, eff_tps)}, {token_info}){timing}"
    ]

    # Tool list
    if tools_used > 0 and tools_called:
        lines.append(f"      tools: {', '.join(_dedup_consecutive(tools_called))}")

    # Per-tool timing
    if tool_timings:
        lines.extend(_format_tool_timings(tool_timings))

    # Delegation events
    if deleg_events:
        lines.extend(_format_delegation_events(deleg_events))

    # Role chain
    if len(role_history) > 1:
        lines.append(f"      chain: {' → '.join(role_history)}")

    return lines


def format_reward_skip(action_key: str, reason: str = "INFRA_SKIP") -> list[str]:
    """Format a reward skip line.

    Output:
        SELF:direct -> INFRA_SKIP (not injecting reward)
    """
    return [f"    {action_key} -> {reason} (not injecting reward)"]


def format_all_infra_skip(action_key: str) -> list[str]:
    """Format when all architect results are infra errors.

    Output:
        ARCHITECT -> ALL INFRA_SKIP
    """
    return [f"    {action_key} -> ALL INFRA_SKIP"]


# ── Shared sub-formatters ─────────────────────────────────────────────


def _format_tool_timings(tool_timings: list[dict]) -> list[str]:
    """Format per-tool timing lines.

    Output:
        peek: 120ms (ok)
        grep: 85ms (fail)
    """
    lines = []
    for tt in tool_timings:
        name = tt.get("tool_name", "?")
        ms = tt.get("elapsed_ms", 0.0)
        ok = "ok" if tt.get("success", True) else "fail"
        lines.append(f"      {name}: {ms:.0f}ms ({ok})")
    return lines


def _format_delegation_events(events: list[dict]) -> list[str]:
    """Format delegation event lines with optional summary.

    Output (multiple delegates):
        delegates: 3 (2x worker_explore, coder_escalation)
        delegate: worker_explore → ok (8200ms, 44.1 t/s, 362 tok)
        delegate: worker_explore → ok (7800ms, 42.3 t/s, 340 tok)
        delegate: coder_escalation → ok (31500ms, 22.7 t/s, 715 tok)

    Output (single delegate):
        delegate: coder_escalation → ok (31500ms, 22.7 t/s, 715 tok)
    """
    lines: list[str] = []

    # Summary line for multiple delegates
    if len(events) > 1:
        role_counts = Counter(de.get("to_role", "?") for de in events)
        parts = []
        for role, count in role_counts.most_common():
            parts.append(f"{count}x {role}" if count > 1 else role)
        lines.append(f"      delegates: {len(events)} ({', '.join(parts)})")

    # Individual delegate lines
    for de in events:
        to_role = de.get("to_role", "?")
        ms = de.get("elapsed_ms", 0.0)
        tok = de.get("tokens_generated", 0)
        success = de.get("success")
        tps = (tok / (ms / 1000.0)) if ms > 0 and tok > 0 else 0.0
        ok = "ok" if success else ("fail" if success is False else "?")
        lines.append(
            f"      delegate: {to_role} → {ok} ({ms:.0f}ms{_tps_str(tps)}, {tok} tok)"
        )

    return lines

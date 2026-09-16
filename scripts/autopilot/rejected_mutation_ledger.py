"""AP-53: harness-written ledger of rejected prompt/code mutations.

WHY
---
A zero-compute measurement over the current-run journal (trials 0-1505,
2026-05-26 .. 2026-08-09) found 133 trials (9.7% of all trials, 31.3% of trials
with a usable config key) re-running a concrete config that an earlier trial had
already rejected. The median gap between the first rejection and the re-proposal
was 91 trials, far outside the 12-trial journal window the planner sees. Only 8
of the 133 were blacklisted at dispatch. The mutation proposer had no durable
memory of WHAT was rejected, only a short prose failure window.

WHAT
----
Every reject path in ``actions.py`` appends one record here:
``{target, mutation_type, unified diff, per-suite deltas, rejecting gate,
timestamp}``. The HARNESS writes it from values it already holds (the mutation
object, the eval result, and the gate that refused it). No LLM text is ever
recorded as a finding. ``render_for_prompt`` feeds the most recent rejections
for the same target back into the mutation prompt through
``_build_mutation_context``.

The diff is stored truncated with a SHA-256 of the full diff, so a re-proposal
of an identical change is detectable (``diff_sha256``) even when the prompt shows
only a prefix.

Append-only JSONL beside the journal. Writes are flock'd and fsync'd. Every
public entry point fails open: a ledger failure never changes a trial.
"""

from __future__ import annotations

import difflib
import fcntl
import hashlib
import json
import logging
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

log = logging.getLogger("autopilot")

LEDGER_FILENAME = "autopilot_rejected_mutations.jsonl"
LEDGER_SCHEMA_VERSION = 1
MAX_STORED_DIFF_CHARS = 8000
DEFAULT_PROMPT_DIFF_CHARS = 1200


def ledger_path(journal_dir: Path) -> Path:
    return Path(journal_dir) / LEDGER_FILENAME


def unified_diff(original: str, mutated: str, *, target: str = "") -> str:
    return "".join(
        difflib.unified_diff(
            (original or "").splitlines(keepends=True),
            (mutated or "").splitlines(keepends=True),
            fromfile=f"a/{target}",
            tofile=f"b/{target}",
            n=2,
        )
    )


def _finite(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def per_suite_deltas(
    candidate: Mapping[str, Any] | None,
    baseline: Mapping[str, Any] | None,
) -> dict[str, float]:
    """Suite-wise candidate minus baseline, over suites both sides measured."""
    out: dict[str, float] = {}
    for suite, value in sorted((candidate or {}).items()):
        cand = _finite(value)
        base = _finite((baseline or {}).get(suite))
        if cand is None or base is None:
            continue
        out[str(suite)] = round(cand - base, 6)
    return out


def build_record(
    *,
    target: str,
    mutation_type: str,
    artifact_kind: str,
    rejecting_gate: str,
    original_content: str = "",
    mutated_content: str = "",
    diff_text: str = "",
    description: str = "",
    gate_detail: str = "",
    per_suite_quality: Mapping[str, Any] | None = None,
    baseline_per_suite_quality: Mapping[str, Any] | None = None,
    quality: Any = None,
    tier: Any = None,
    trial_id: int | None = None,
    timestamp: str = "",
) -> dict[str, Any]:
    diff = diff_text or unified_diff(original_content, mutated_content, target=target)
    return {
        "schema_version": LEDGER_SCHEMA_VERSION,
        "writer": "harness",
        "timestamp": timestamp or datetime.now(timezone.utc).isoformat(),
        "trial_id": trial_id,
        "target": str(target or ""),
        "mutation_type": str(mutation_type or ""),
        "artifact_kind": str(artifact_kind or ""),
        "rejecting_gate": str(rejecting_gate or ""),
        "gate_detail": str(gate_detail or "")[:500],
        "description": str(description or "")[:300],
        "diff_sha256": hashlib.sha256(diff.encode("utf-8")).hexdigest(),
        "diff_chars": len(diff),
        "unified_diff": diff[:MAX_STORED_DIFF_CHARS],
        "diff_truncated": len(diff) > MAX_STORED_DIFF_CHARS,
        "quality": _finite(quality),
        "tier": tier if isinstance(tier, int) else None,
        "per_suite_deltas": per_suite_deltas(per_suite_quality, baseline_per_suite_quality),
        "per_suite_deltas_basis": (
            "candidate_minus_tier_baseline" if baseline_per_suite_quality else "unavailable"
        ),
    }


def append_record(journal_dir: Path, record: Mapping[str, Any]) -> bool:
    """Durably append one record. Returns False (and logs) instead of raising."""
    try:
        path = ledger_path(journal_dir)
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(dict(record), sort_keys=True, default=str, allow_nan=False) + "\n"
        with open(path, "a") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                f.write(line)
                f.flush()
                os.fsync(f.fileno())
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        return True
    except Exception as exc:  # noqa: BLE001 - the ledger must never fail a trial
        log.warning("AP-53 rejected-mutation ledger append failed: %s", exc)
        return False


def load_records(journal_dir: Path) -> list[dict[str, Any]]:
    """Read every well-formed record; a torn/malformed line is skipped."""
    path = ledger_path(journal_dir)
    out: list[dict[str, Any]] = []
    try:
        with open(path) as f:
            for raw in f:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    row = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if isinstance(row, dict):
                    out.append(row)
    except OSError:
        return []
    return out


def recent_for_target(
    journal_dir: Path,
    target: str,
    *,
    n: int = 3,
) -> list[dict[str, Any]]:
    rows = [r for r in load_records(journal_dir) if r.get("target") == target]
    return rows[-n:] if n > 0 else []


def render_for_prompt(
    journal_dir: Path,
    target: str,
    *,
    n: int = 3,
    max_diff_chars: int = DEFAULT_PROMPT_DIFF_CHARS,
) -> str:
    """Render the last ``n`` rejections of ``target`` for the mutation prompt."""
    try:
        rows = recent_for_target(journal_dir, target, n=n)
        if not rows:
            return ""
        all_rows = load_records(journal_dir)
        repeats: dict[str, int] = {}
        for row in all_rows:
            if row.get("target") == target and row.get("diff_sha256"):
                repeats[row["diff_sha256"]] = repeats.get(row["diff_sha256"], 0) + 1
        lines = [
            f"## Previously Rejected Mutations of `{target}` (harness ledger, newest last)",
            "These exact changes were applied, evaluated and reverted. Do not re-propose "
            "them; if you revisit the idea, state what differs and why it should now pass.",
        ]
        for row in rows:
            deltas = row.get("per_suite_deltas") or {}
            worst = sorted(deltas.items(), key=lambda kv: kv[1])[:4]
            delta_text = ", ".join(f"{k} {v:+.3f}" for k, v in worst) or "n/a"
            seen = repeats.get(row.get("diff_sha256", ""), 1)
            trial = row.get("trial_id")
            lines.append(
                f"- {row.get('timestamp', '')[:19]} trial #{trial if trial is not None else '?'} "
                f"{row.get('mutation_type', '')}: rejected by `{row.get('rejecting_gate', '')}`"
                + (f" ({row.get('gate_detail')})" if row.get("gate_detail") else "")
                + f"; per-suite delta vs baseline: {delta_text}"
                + (f"; this exact diff was rejected {seen}x" if seen > 1 else "")
            )
            diff = str(row.get("unified_diff") or "")
            if diff:
                clipped = diff[:max_diff_chars]
                suffix = "\n... (diff truncated)" if len(diff) > max_diff_chars or row.get("diff_truncated") else ""
                lines.append("```diff\n" + clipped + suffix + "\n```")
        return "\n".join(lines)
    except Exception as exc:  # noqa: BLE001 - prompt context must never block dispatch
        log.debug("AP-53 ledger render failed: %s", exc)
        return ""


# ── planner-facing view: previously rejected CONCRETE configs ─────────────────
#
# The AP-53 measurement found structural_experiment actions account for 105 of the
# 133 re-proposals; they never pass through PromptForge, so the planner prompt needs
# the same memory. This view is a pure fold over the durable journal — every reject
# path already journals a row — so it is harness-derived, never LLM-written.

_OBSERVATIONAL_ACTION_TYPES = frozenset({
    "seed_batch", "deep_eval", "consult_gate_probe", "rollback",
    "train_routing_models", "distill_knowledge", "distill_skillbank",
})
_DELIBERATE_REPLAY_KEYS = (
    "seq_candidate_replay", "seq_promotion_fresh_eval", "multitier_validation",
    "seq_baseline_reference_draw",
)
_NEUTRAL_DEFICIENCIES = frozenset({"mad_noise", "reproduction_confirmed", "seq_stale_reference"})


def _hard_rejection(entry: Any) -> str:
    """Terminal rejection class of a journal entry, or '' when not a hard reject."""
    deficiency = str(getattr(entry, "deficiency_category", "") or "")
    corrupted = str(getattr(entry, "bug_corrupted_by", "") or "")
    if deficiency == "seq_refuted" or corrupted == "seq_refuted":
        return "sequential_refuted"
    if corrupted or deficiency in _NEUTRAL_DEFICIENCIES or deficiency == "seq_accumulating":
        return ""
    if str(getattr(entry, "keep_revert_decision", "") or "") == "excluded":
        return ""
    if str(getattr(entry, "outcome_status", "ok") or "ok") == "invalid":
        return "invalid"
    analysis = str(getattr(entry, "failure_analysis", "") or "")
    if analysis.startswith("VIOLATIONS") or deficiency == "consecutive_failures":
        return "safety_gate"
    return ""


def _concrete_action(entry: Any) -> dict[str, Any] | None:
    action = getattr(entry, "config_snapshot", None)
    if not isinstance(action, dict) or not action.get("type"):
        return None
    if action["type"] in _OBSERVATIONAL_ACTION_TYPES:
        return None
    if action["type"] == "numeric_trial":
        # NumericSwarm owns numeric surfaces and already records failed samples
        # (``swarm.mark_failed``); continuous Optuna draws almost never repeat
        # exactly and would crowd the planner block.
        return None
    details = getattr(entry, "eval_details", None)
    if isinstance(details, dict) and any(details.get(k) for k in _DELIBERATE_REPLAY_KEYS):
        return None  # a deliberate re-measurement is not a re-proposal
    return action


def _first_violation(text: str) -> str:
    for line in text.splitlines():
        line = line.strip().lstrip("-").strip()
        if not line or line.rstrip(":").upper() in {"VIOLATIONS", "DEGRADED SUITES"}:
            continue
        return line[:140]
    return ""


def rejected_configs_from_entries(entries: list[Any]) -> list[dict[str, Any]]:
    """Fold journal entries into still-standing hard rejections per config identity.

    A later ``frontier`` trial of the same config clears the record: the config has
    since been accepted, so it is no longer a known-bad proposal.
    """
    from src.autopilot_core.action_identity import canonical_action, config_fingerprint

    records: dict[str, dict[str, Any]] = {}
    for entry in entries:
        action = _concrete_action(entry)
        if action is None:
            continue
        key = config_fingerprint(action)
        reason = _hard_rejection(entry)
        trial_id = getattr(entry, "trial_id", None)
        if not reason:
            if str(getattr(entry, "pareto_status", "") or "") == "frontier":
                records.pop(key, None)
            continue
        rec = records.setdefault(key, {
            "config_fingerprint": key,
            "action_type": str(action.get("type")),
            "config": canonical_action(action),
            "rejections": 0,
            "first_trial_id": trial_id,
        })
        rec["rejections"] += 1
        rec["last_trial_id"] = trial_id
        rec["last_reason"] = reason
        rec["last_detail"] = _first_violation(str(getattr(entry, "failure_analysis", "") or ""))
    return sorted(records.values(), key=lambda r: (r.get("last_trial_id") or -1))


def render_rejected_configs_for_planner(entries: list[Any], *, limit: int = 10) -> str:
    try:
        rows = rejected_configs_from_entries(entries)[-limit:]
        if not rows:
            return ""
        lines = [
            "## Previously Rejected Configs (harness journal fold, newest last)",
            "Each config below (action identity, narrative fields dropped) was dispatched "
            "and hard-rejected; the recent-trial "
            "summary does not show configs. Re-proposing one wastes a trial unless you "
            "name what has changed since.",
        ]
        for row in rows:
            config = json.dumps(
                {k: v for k, v in row["config"].items() if k != "type"},
                sort_keys=True, default=str,
            )
            if len(config) > 160:
                config = config[:157] + "..."
            times = f" x{row['rejections']}" if row["rejections"] > 1 else ""
            detail = f" — {row['last_detail']}" if row.get("last_detail") else ""
            lines.append(
                f"- trial #{row['last_trial_id']}{times} {row['action_type']} {config}: "
                f"{row['last_reason']}{detail}"
            )
        return "\n".join(lines)
    except Exception as exc:  # noqa: BLE001 - prompt context must never block planning
        log.debug("AP-53 rejected-config render failed: %s", exc)
        return ""

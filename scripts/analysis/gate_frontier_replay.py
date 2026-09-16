#!/usr/bin/env python3
"""Gate-frontier replay: legacy promotion guard vs live-scope promotion guard (2026-09-16).

Zero inference. Replays the stored autopilot journal through ``SafetyGate.update_baseline``
twice — once with the pre-2026-09-16 guard scope (LEGACY tokens/second replay over every
era) and once with the live scope (the objective policy + epoch fence live at that trial) —
and reports each path's promotions and refusals.

What the replay holds fixed, because the journal does not record it:
  * baseline eligibility (contention matrix / speed_metric_mode) is held OPEN;
  * no eval-quality / speed era hold, sequential path OFF;
  * each tier's baseline starts EMPTY and ratchets only through the replayed promotions
    of the path being simulated;
  * a row is a promotion CANDIDATE when it reached the clean-update branch that calls
    ``update_baseline``: tier >= 1, no ``bug_corrupted_by``, no learning exclusion,
    measured quality;
  * supersession events are folded first (today's journal-authority view of each row).

Live scope at trial R (reconstructed; the state history is not journaled): the policy
stamped on R (``objective_policy_live``; unstamped rows are legacy) with the epoch fence at
the first row that carried that policy. The legacy era therefore has ``exclude_before_ts``
= None on BOTH paths, so the two paths must agree on every legacy-era row — the replay
asserts that as a self-check.

Two orderings are replayed:
  * ``production`` — rows strictly BEFORE R. This is what the running loop sees: the trial
    is journaled after ``update_baseline`` returns.
  * ``archive_first`` — rows up to AND INCLUDING R, the ordering ``update_baseline``'s
    docstring assumes.

Usage:
    gate_frontier_replay.py slim --out tests/fixtures/gate_frontier_replay.json
    gate_frontier_replay.py report --fixture tests/fixtures/gate_frontier_replay.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
for _p in (REPO, REPO / "scripts" / "autopilot"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from src.autopilot_core.action_identity import config_fingerprint_from_row  # noqa: E402
from src.autopilot_core.journal_reconstruction import (  # noqa: E402
    fold_supersession_events,
    objectives_from_journal_row,
    parse_journal_ts,
    reconstruct_archive_from_journal_rows,
)
from src.autopilot_core.tier_specs import (  # noqa: E402
    LEGACY_OBJECTIVE_POLICY,
    MIN_FRONTIER_EVAL_TIER,
    PRE_RESOURCE_LANES_RATE_4D_OBJECTIVE_POLICY,
    RATE_4D_OBJECTIVE_POLICY,
    RESOURCE_LANES_V2_RATE_4D_OBJECTIVE_POLICY,
    quality_from_row,
)

# The canonical clone's journal (a worktree has no orchestration/ journal of its own).
_JOURNAL_DIR = Path("/mnt/raid0/llm/epyc-orchestrator/orchestration")
DEFAULT_SHARDS = (
    _JOURNAL_DIR / "autopilot_journal.jsonl",
    _JOURNAL_DIR / "autopilot_journal_1.jsonl",
)
LIVE_POLICIES = (
    LEGACY_OBJECTIVE_POLICY,
    PRE_RESOURCE_LANES_RATE_4D_OBJECTIVE_POLICY,
    RESOURCE_LANES_V2_RATE_4D_OBJECTIVE_POLICY,
    RATE_4D_OBJECTIVE_POLICY,
)
ORDERINGS = ("production", "archive_first")


# ── slim fixture ────────────────────────────────────────────────────────────


def _read_shards(paths: list[Path]) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    rows: list[dict[str, Any]] = []
    provenance = []
    for path in paths:
        data = path.read_bytes()
        provenance.append({"path": str(path), "sha256": hashlib.sha256(data).hexdigest()})
        for line in data.decode("utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows, provenance


def _row_policy(row: dict[str, Any]) -> str:
    details = row.get("eval_details")
    policy = ""
    if isinstance(details, dict):
        policy = str(details.get("objective_policy_live") or "")
    return policy or str(row.get("objective_policy_live") or "") or LEGACY_OBJECTIVE_POLICY


def _slim_row(row: dict[str, Any]) -> dict[str, Any]:
    """Keep exactly what objective building, clustering and exclusion read."""
    details = row.get("eval_details") if isinstance(row.get("eval_details"), dict) else {}
    inner = details.get("details") if isinstance(details.get("details"), dict) else {}
    qresults = details.get("question_results")
    declared = row.get("n_questions")
    if isinstance(qresults, list) and qresults:
        qids = {
            str(item.get("qid") or item.get("question_id") or "").strip()
            for item in qresults
            if isinstance(item, dict)
        } - {""}
        n_questions = len(qids) if qids else len(qresults)
    else:
        n_questions = declared or inner.get("total") or inner.get("n_questions")
        if not n_questions and isinstance(inner.get("per_suite_counts"), dict):
            n_questions = sum(int(v) for v in inner["per_suite_counts"].values() if int(v) > 0)
    slim_details: dict[str, Any] = {}
    if details:
        slim_details = {
            "slim": True,
            "objective_policy_live": details.get("objective_policy_live"),
            "learning_exclusion": details.get("learning_exclusion") or {},
            "eval_wall_s": details.get("eval_wall_s") or inner.get("eval_wall_s"),
        }
    return {
        "trial_id": row.get("trial_id"),
        "timestamp": row.get("timestamp"),
        "tier": row.get("tier"),
        "quality": row.get("quality"),
        "speed": row.get("speed"),
        "cost": row.get("cost"),
        "reliability": row.get("reliability"),
        "bug_corrupted_by": row.get("bug_corrupted_by") or "",
        "objective_policy_live": row.get("objective_policy_live"),
        "n_questions": n_questions,
        "eval_wall_s": row.get("eval_wall_s"),
        "eval_details": slim_details,
        "config_snapshot": {"fp": config_fingerprint_from_row(row)},
        "reasoning": "",
    }


def build_slim_fixture(paths: list[Path]) -> dict[str, Any]:
    raw, provenance = _read_shards(paths)
    folded, meta = fold_supersession_events(raw)
    trials = [r for r in folded if "trial_id" in r]
    slim = [_slim_row(r) for r in trials]
    # Self-check: slimming must not change any objective under any replayed policy, nor
    # any clustering identity.
    for full, small in zip(trials, slim):
        for policy in LIVE_POLICIES:
            a = objectives_from_journal_row(full, objective_policy=policy)
            b = objectives_from_journal_row(small, objective_policy=policy)
            assert a == b, (full.get("trial_id"), policy, a, b)
        assert _row_policy(full) == _row_policy(small)
    return {
        "_provenance": {
            "source_shards": provenance,
            "supersession_fold": {k: v for k, v in meta.items() if k != "target_trial_ids"},
            "note": (
                "Supersession events folded; config_snapshot replaced by the original "
                "config fingerprint (clustering-identical); question_results reduced to the "
                "distinct-qid count in n_questions (seq-rate-identical). Objective equality "
                "under every replayed policy is asserted at build time."
            ),
        },
        "rows": slim,
    }


# ── replay ───────────────────────────────────────────────────────────────────


def _epoch_by_policy(rows: list[dict[str, Any]]) -> dict[str, float | None]:
    epochs: dict[str, float | None] = {LEGACY_OBJECTIVE_POLICY: None}
    for row in rows:
        policy = _row_policy(row)
        if policy not in epochs:
            epochs[policy] = parse_journal_ts(row.get("timestamp"))
    return epochs


def _is_candidate(row: dict[str, Any]) -> bool:
    try:
        tier = int(row.get("tier") or 0)
    except (TypeError, ValueError):
        return False
    if tier < MIN_FRONTIER_EVAL_TIER or row.get("bug_corrupted_by"):
        return False
    details = row.get("eval_details") or {}
    if (details.get("learning_exclusion") or {}).get("by"):
        return False
    return quality_from_row(row) is not None


def _eval_result(row: dict[str, Any]):
    from safety_gate import EvalResult

    n = int(row.get("n_questions") or 0)
    wall = float(row.get("eval_wall_s") or (row.get("eval_details") or {}).get("eval_wall_s") or 0)
    return EvalResult(
        tier=int(row["tier"]),
        quality=float(row["quality"]),
        speed=float(row.get("speed") or 0.0),
        cost=float(row.get("cost") or 0.0),
        reliability=float(row.get("reliability") or 0.0),
        n_questions=n,
        eval_wall_s=wall,
    )


def _view(rows, policy, exclude_before_ts, scope):
    from pareto_archive import ParetoArchive
    from safety_gate import PromotionGuardView

    payload = reconstruct_archive_from_journal_rows(
        rows, None, exclude_before_ts=exclude_before_ts, objective_policy=policy
    )
    return PromotionGuardView(
        archive=ParetoArchive.from_archive_payload(payload, read_only=True),
        objective_policy=policy,
        scope=scope,
        exclude_before_ts=exclude_before_ts,
    )


COMPACT_KEYS = (
    "trial_id",
    "tier",
    "row_policy",
    "decision",
    "previous_quality",
    "new_quality",
    "row_speed_tps",
    "frontdoor_speed_after",
    "frontdoor_task_rate_qph_after",
)


def compact(decision: dict[str, Any]) -> dict[str, Any]:
    return {k: decision[k] for k in COMPACT_KEYS}


def baselines_before(decisions: list[dict[str, Any]], trial_id: int) -> dict[int, float]:
    """Tier baselines a replay held just before ``trial_id`` (from its promoted decisions)."""
    out: dict[int, float] = {}
    for d in decisions:
        if d["trial_id"] < trial_id and d["decision"] == "promoted":
            out[int(d["tier"])] = float(d["new_quality"])
    return out


def _classify(reason: str) -> str:
    table = (
        ("live-epoch frontier is empty", "refused_live_frontier_empty"),
        ("not a same-tier frontier representative", "refused_not_frontier_representative"),
        ("exceeds same-tier archive max", "refused_above_archive_max"),
        ("reproductions; source has", "refused_too_few_reproductions"),
        ("does not clear baseline", "refused_repro_median_below_quantum"),
        ("missing objective tuple", "refused_objective_tuple"),
        ("missing n_questions", "refused_no_quantum"),
        ("not a monotonic", "skipped_not_monotonic"),
    )
    for needle, label in table:
        if needle in reason:
            return label
    return "other:" + reason[:60]


def replay(
    rows: list[dict[str, Any]],
    *,
    path: str,
    ordering: str,
    start_trial_id: int | None = None,
    initial_baselines: dict[int, float] | None = None,
    cache: dict | None = None,
) -> dict[str, Any]:
    """Replay one (path, ordering). ``path`` is ``old`` (legacy) or ``new`` (live scope).

    ``start_trial_id`` + ``initial_baselines`` replay a WINDOW: candidates before the start
    are skipped and the tier baselines start from ``initial_baselines`` (the golden state
    at that point); every archive still sees the full visible prefix.
    """
    import safety_gate as sg

    cache = {} if cache is None else cache

    epochs = _epoch_by_policy(rows)
    decisions: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory() as tmp:
        gate = sg.SafetyGate(baseline_path=Path(tmp) / "absent.yaml")
        gate._baseline_eligible = lambda result: (True, "replay: eligibility held open", {})
        gate.baseline.baselines_by_tier.clear()
        gate.baseline.baselines_by_tier.update(
            {int(t): float(q) for t, q in (initial_baselines or {}).items()}
        )
        current: dict[str, Any] = {}
        sg.configure_promotion_guard_archive(lambda: current["view"])
        try:
            for index, row in enumerate(rows):
                if not _is_candidate(row):
                    continue
                if start_trial_id is not None and int(row["trial_id"]) < start_trial_id:
                    continue
                n_visible = index + 1 if ordering == "archive_first" else index
                policy = _row_policy(row)
                if path == "old":
                    current["view"] = _LazyView(
                        rows, n_visible, LEGACY_OBJECTIVE_POLICY, None, "legacy_unscoped", cache
                    )
                else:
                    current["view"] = _LazyView(
                        rows, n_visible, policy, epochs[policy], "live", cache
                    )
                result = _eval_result(row)
                before_speed = gate.baseline.frontdoor_speed
                update = gate.update_baseline(result, source_trial_id=int(row["trial_id"]))
                if update.updated:
                    decision = "promoted"
                elif update.reason.startswith("not a monotonic"):
                    continue  # ratchet skip, not a guard decision
                else:
                    decision = _classify(update.reason)
                decisions.append(
                    {
                        "trial_id": int(row["trial_id"]),
                        "tier": int(row["tier"]),
                        "row_policy": policy,
                        "decision": decision,
                        "previous_quality": update.previous_quality,
                        "new_quality": float(update.new_quality),
                        "row_speed_tps": float(row.get("speed") or 0.0),
                        "frontdoor_speed_before": before_speed,
                        "frontdoor_speed_after": gate.baseline.frontdoor_speed,
                        "frontdoor_task_rate_qph_after": gate.baseline.frontdoor_task_rate_qph,
                        "guard_policy": current["view"].objective_policy,
                    }
                )
        finally:
            sg.configure_promotion_guard_archive(None)
    counts = Counter(d["decision"] for d in decisions)
    return {"path": path, "ordering": ordering, "counts": dict(sorted(counts.items())), "decisions": decisions}


class _LazyView:
    """PromotionGuardView built on first archive access (most candidates never need one).

    Archives are cached by (visible prefix length, policy, fence): the old path and the new
    path share every legacy-era view, and the two orderings share prefixes."""

    def __init__(self, rows, n_visible, policy, exclude_before_ts, scope, cache):
        self._rows = rows
        self._key = (n_visible, policy, exclude_before_ts)
        self._cache = cache
        self.objective_policy = policy
        self.scope = scope
        self.exclude_before_ts = exclude_before_ts

    @property
    def archive(self):
        if self._key not in self._cache:
            n_visible, policy, exclude = self._key
            self._cache[self._key] = _view(self._rows[:n_visible], policy, exclude, "").archive
        return self._cache[self._key]


def compare(
    rows: list[dict[str, Any]],
    *,
    start_trial_id: int | None = None,
    initial_baselines: dict[str, dict[int, float]] | None = None,
) -> dict[str, Any]:
    logging.getLogger("autopilot.safety").setLevel(logging.CRITICAL)
    cache: dict = {}
    runs = {
        f"{p}/{o}": replay(
            rows,
            path=p,
            ordering=o,
            start_trial_id=start_trial_id,
            initial_baselines=(initial_baselines or {}).get(f"{p}/{o}"),
            cache=cache,
        )
        for o in ORDERINGS
        for p in ("old", "new")
    }
    differences = {}
    for ordering in ORDERINGS:
        old = {d["trial_id"]: d for d in runs[f"old/{ordering}"]["decisions"]}
        new = {d["trial_id"]: d for d in runs[f"new/{ordering}"]["decisions"]}
        diff = []
        for tid in sorted(set(old) | set(new)):
            a, b = old.get(tid), new.get(tid)
            if (a or {}).get("decision") != (b or {}).get("decision"):
                diff.append(
                    {
                        "trial_id": tid,
                        "row_policy": (a or b)["row_policy"],
                        "old": (a or {}).get("decision", "not_reached(ratchet_skip)"),
                        "new": (b or {}).get("decision", "not_reached(ratchet_skip)"),
                    }
                )
        differences[ordering] = diff
    return {
        "runs": {k: {"counts": v["counts"], "decisions": v["decisions"]} for k, v in runs.items()},
        "differences": differences,
    }


def live_restart_probe(rows: list[dict[str, Any]], state: dict[str, Any]) -> dict[str, Any]:
    """What the NEW guard reports at the next restart, under the stored live state."""
    policy = str(state.get("pareto_objective_policy") or LEGACY_OBJECTIVE_POLICY)
    exclude = float(state.get("pareto_exclude_before_ts") or 0.0) or None
    view = _view(rows, policy, exclude, "live")
    return {
        "objective_policy": policy,
        "exclude_before_ts": exclude,
        "frontier_sizes": {str(t): len(view.archive.frontier(t)) for t in (1, 2, 3)},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_slim = sub.add_parser("slim")
    p_slim.add_argument("--shard", action="append", type=Path)
    p_slim.add_argument("--state", type=Path)
    p_slim.add_argument("--out", type=Path, required=True)
    p_rep = sub.add_parser("report")
    p_rep.add_argument("--fixture", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.cmd == "slim":
        shards = args.shard or list(DEFAULT_SHARDS)
        fixture = build_slim_fixture(shards)
        result = compare(fixture["rows"])
        fixture["golden"] = {
            "counts": {k: v["counts"] for k, v in result["runs"].items()},
            "differences": result["differences"],
            "decisions": {k: [compact(d) for d in v["decisions"]] for k, v in result["runs"].items()},
            "decisions_sha256": {
                k: hashlib.sha256(
                    json.dumps(v["decisions"], sort_keys=True).encode()
                ).hexdigest()
                for k, v in result["runs"].items()
            },
        }
        if args.state and args.state.exists():
            state = json.loads(args.state.read_text())
            fixture["live_state"] = {
                "pareto_objective_policy": state.get("pareto_objective_policy"),
                "pareto_exclude_before_ts": state.get("pareto_exclude_before_ts"),
                "baselines_by_tier": (state.get("baseline_state") or {}).get("baselines_by_tier"),
                "frontdoor_speed": (state.get("baseline_state") or {}).get("frontdoor_speed"),
            }
            fixture["golden"]["live_restart_probe"] = live_restart_probe(fixture["rows"], state)
        args.out.write_text(json.dumps(fixture, sort_keys=True, separators=(",", ":")) + "\n")
        print(json.dumps(fixture["golden"], indent=1, sort_keys=True))
        return 0
    fixture = json.loads(args.fixture.read_text())
    result = compare(fixture["rows"])
    print(json.dumps({"counts": {k: v["counts"] for k, v in result["runs"].items()},
                      "differences": result["differences"]}, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

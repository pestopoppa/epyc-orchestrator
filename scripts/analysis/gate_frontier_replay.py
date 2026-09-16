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

Three paths are replayed (see ``replay``): ``old`` (legacy scope), ``live`` (live scope,
commit e401549d) and ``live_c`` (live scope + operator decisions (c) and (b), 2026-09-16).
``forward_simulation`` replays the tasks/hour trials as if they arrived after the live fence.

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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
for _p in (REPO, REPO / "scripts" / "autopilot"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from src.autopilot_core.action_identity import (  # noqa: E402
    CONFIG_IDENTIFYING_ACTION_FIELDS,
    CONTENT_IDENTIFIED_ACTION_TYPES,
    INFRA_REGIME_DIGEST_KEY,
    SERVED_CONTENT_KEY,
    action_from_journal_row,
    config_fingerprint_from_row,
    row_config_identity,
)
from src.autopilot_core.journal_reconstruction import (  # noqa: E402
    fold_supersession_events,
    objectives_from_journal_row,
    parse_journal_ts,
    reconstruct_archive_from_journal_rows,
)
from src.autopilot_core.learning_exclusions import (  # noqa: E402
    FRONTIER_ADMISSION_KEY,
    FRONTIER_ADMISSION_REPRESENTATIVE,
)
from src.autopilot_core.live_reproductions import live_reproductions  # noqa: E402
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


def _slim_action(row: dict[str, Any]) -> dict[str, Any]:
    """Clustering- and identity-preserving stand-in for the action dict.

    The original fingerprint is embedded, so distinct configs stay distinct; an action that
    identifies a served config keeps its type and a non-empty delta field, so
    ``row_config_identity`` is None exactly when it was None on the full row."""
    fp = config_fingerprint_from_row(row)
    if row_config_identity(row) is None:
        return {"fp": fp}
    action = action_from_journal_row(row)
    action_type = str(action.get("type"))
    if action_type in CONTENT_IDENTIFIED_ACTION_TYPES:
        # Identity comes from eval_details.served_content, which the slim row keeps.
        return {"type": action_type, "fp": fp}
    field = CONFIG_IDENTIFYING_ACTION_FIELDS[action_type]
    return {"type": action_type, field: {"fp": fp}}


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
            "infra_comparability": details.get("infra_comparability") or "",
        }
        for key in (SERVED_CONTENT_KEY, INFRA_REGIME_DIGEST_KEY):
            if details.get(key):
                slim_details[key] = details[key]
    return {
        "trial_id": row.get("trial_id"),
        "action_type": row.get("action_type")
        or str((action_from_journal_row(row) or {}).get("type") or ""),
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
        "comparability": row.get("comparability") or {},
        "config_snapshot": _slim_action(row),
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
        assert (row_config_identity(full) is None) == (row_config_identity(small) is None)
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


def _is_candidate(row: dict[str, Any], *, include_within_noise: bool = False) -> bool:
    """Rows whose trial reached ``update_baseline``.

    The pre-(c) loop called it for clean trials only. The (c) loop also calls it for a
    trusted within-noise reproduction (``mad_noise`` / ``reproduction_confirmed``) whose
    live objectives were measured."""
    try:
        tier = int(row.get("tier") or 0)
    except (TypeError, ValueError):
        return False
    if tier < MIN_FRONTIER_EVAL_TIER:
        return False
    details = row.get("eval_details") or {}
    excluded_by = str((details.get("learning_exclusion") or {}).get("by") or "")
    bug = str(row.get("bug_corrupted_by") or "")
    if excluded_by or bug:
        if not include_within_noise or excluded_by not in {"mad_noise", "reproduction_confirmed"}:
            return False
        if bug and bug != "mad_noise":
            return False
        if not _row_objectives_measured(row):
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


def stamp_clean_representatives(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """What-if (c): every promotion candidate row carries the frontier-representative stamp,
    as the new trial loop writes it. The stored journal itself is never modified."""
    out = []
    for row in rows:
        if (
            _is_candidate(row)
            and _row_objectives_measured(row)
            and row_config_identity(row) is not None
        ):
            row = dict(row)
            details = dict(row.get("eval_details") or {})
            details[FRONTIER_ADMISSION_KEY] = FRONTIER_ADMISSION_REPRESENTATIVE
            row["eval_details"] = details
        out.append(row)
    return out


def _row_objectives_measured(row: dict[str, Any]) -> bool:
    policy = _row_policy(row)
    return objectives_from_journal_row(row, objective_policy=policy) is not None


COMPACT_KEYS = (
    "trial_id",
    "action_type",
    "tier",
    "row_policy",
    "decision",
    "previous_quality",
    "new_quality",
    "row_speed_tps",
    "frontdoor_speed_after",
    "frontdoor_task_rate_qph_after",
    "promotion_rule",
    "frontier_size_after",
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
        ("carries no served-config identity", "refused_no_config_identity"),
        ("empty-frontier rule (b): the candidate's journal row", "refused_b_no_candidate_row"),
        ("empty-frontier rule (b): candidate config", "refused_b_too_few_reproductions"),
        ("empty-frontier rule (b): reproduced median", "refused_b_median_below_quantum"),
        ("promotion guard archive unavailable", "refused_guard_unavailable"),
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


PATHS = ("old", "live", "live_c")


def replay(
    rows: list[dict[str, Any]],
    *,
    path: str,
    start_trial_id: int | None = None,
    initial_baselines: dict[int, float] | None = None,
    epochs: dict[str, float | None] | None = None,
    cache: dict | None = None,
) -> dict[str, Any]:
    """Replay one path over ``rows`` (production ordering: the guard sees rows BEFORE R).

    * ``old``    — legacy t/s replay over every era, no candidate row (pre-2026-09-16).
    * ``live``   — live policy + epoch fence, no candidate row (commit e401549d).
    * ``live_c`` — live scope plus decision (c): candidate rows are stamped as frontier
      representatives and the candidate's own row is handed to the guard; the empty-frontier
      rule (b) counts comparable live-regime reproductions.

    ``start_trial_id`` + ``initial_baselines`` replay a WINDOW: candidates before the start
    are skipped and the tier baselines start from ``initial_baselines``; every archive still
    sees the full visible prefix.
    """
    import safety_gate as sg

    cache = {} if cache is None else cache
    epochs = epochs if epochs is not None else _epoch_by_policy(rows)
    source = stamp_clean_representatives(rows) if path == "live_c" else rows
    decisions: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory() as tmp:
        gate = sg.SafetyGate(baseline_path=Path(tmp) / "absent.yaml")
        gate._baseline_eligible = lambda result: (True, "replay: eligibility held open", {})
        gate.baseline.baselines_by_tier.clear()
        gate.baseline.baselines_by_tier.update(
            {int(t): float(q) for t, q in (initial_baselines or {}).items()}
        )
        current: dict[str, Any] = {}
        sg.configure_promotion_guard_archive(lambda pending_rows=(): current["view"])
        try:
            for index, row in enumerate(source):
                if not _is_candidate(row, include_within_noise=path == "live_c"):
                    continue
                if start_trial_id is not None and int(row["trial_id"]) < start_trial_id:
                    continue
                policy = _row_policy(row)
                pending: tuple[dict[str, Any], ...] = ()
                if path == "old":
                    current["view"] = _LazyView(
                        source, index, LEGACY_OBJECTIVE_POLICY, None, "legacy_unscoped", cache, "old"
                    )
                elif path == "live":
                    current["view"] = _LazyView(
                        source, index, policy, epochs[policy], "live", cache, "live"
                    )
                else:
                    pending = (row,)
                    current["view"] = _LazyView(
                        source, index + 1, policy, epochs[policy], "live", cache, "live_c"
                    )
                result = _eval_result(row)
                update = gate.update_baseline(
                    result, source_trial_id=int(row["trial_id"]), pending_journal_rows=pending
                )
                if update.updated:
                    decision = "promoted"
                elif update.reason.startswith("not a monotonic"):
                    continue  # ratchet skip, not a guard decision
                else:
                    decision = _classify(update.reason)
                tier = int(row["tier"])
                decisions.append(
                    {
                        "trial_id": int(row["trial_id"]),
                        "tier": tier,
                        "row_policy": policy,
                        "action_type": str(row.get("action_type") or ""),
                        "decision": decision,
                        "promotion_rule": update.promotion_rule,
                        "previous_quality": update.previous_quality,
                        "new_quality": float(update.new_quality),
                        "row_speed_tps": float(row.get("speed") or 0.0),
                        "frontdoor_speed_after": gate.baseline.frontdoor_speed,
                        "frontdoor_task_rate_qph_after": gate.baseline.frontdoor_task_rate_qph,
                        "frontier_size_after": len(current["view"].archive.frontier(tier))
                        if path == "live_c"
                        else None,
                    }
                )
        finally:
            sg.configure_promotion_guard_archive(None)
    counts = Counter(d["decision"] for d in decisions)
    return {"path": path, "counts": dict(sorted(counts.items())), "decisions": decisions}


class _LazyView:
    """PromotionGuardView built on first archive access (most candidates never need one).

    Archives are cached by (path family, visible prefix length, policy, fence)."""

    error = ""

    def __init__(self, rows, n_visible, policy, exclude_before_ts, scope, cache, family):
        self._rows = rows
        self._n = n_visible
        self._key = (family if family == "live_c" else "plain", n_visible, policy, exclude_before_ts)
        self._cache = cache
        self.objective_policy = policy
        self.scope = scope
        self.exclude_before_ts = exclude_before_ts

    @property
    def archive(self):
        if self._key not in self._cache:
            _family, n_visible, policy, exclude = self._key
            self._cache[self._key] = _view(self._rows[:n_visible], policy, exclude, "").archive
        return self._cache[self._key]

    def reproductions(self, tier: int, identity: str) -> list[dict[str, Any]]:
        return live_reproductions(
            self._rows[: self._n],
            tier=tier,
            identity=identity,
            objective_policy=self.objective_policy,
            exclude_before_ts=self.exclude_before_ts,
        )


def compare(
    rows: list[dict[str, Any]],
    *,
    start_trial_id: int | None = None,
    initial_baselines: dict[str, dict[int, float]] | None = None,
) -> dict[str, Any]:
    safety_log = logging.getLogger("autopilot.safety")
    previous_level = safety_log.level
    safety_log.setLevel(logging.CRITICAL)
    cache: dict = {}
    try:
        runs = {
            p: replay(
                rows,
                path=p,
                start_trial_id=start_trial_id,
                initial_baselines=(initial_baselines or {}).get(p),
                cache=cache,
            )
            for p in PATHS
        }
    finally:
        safety_log.setLevel(previous_level)
    differences = {}
    for left, right in (("old", "live"), ("live", "live_c")):
        a_map = {d["trial_id"]: d for d in runs[left]["decisions"]}
        b_map = {d["trial_id"]: d for d in runs[right]["decisions"]}
        diff = []
        for tid in sorted(set(a_map) | set(b_map)):
            a, b = a_map.get(tid), b_map.get(tid)
            if (a or {}).get("decision") != (b or {}).get("decision"):
                diff.append(
                    {
                        "trial_id": tid,
                        "row_policy": (a or b)["row_policy"],
                        left: (a or {}).get("decision", "not_reached(ratchet_skip)"),
                        right: (b or {}).get("decision", "not_reached(ratchet_skip)"),
                    }
                )
        differences[f"{left}_vs_{right}"] = diff
    return {
        "runs": {k: {"counts": v["counts"], "decisions": v["decisions"]} for k, v in runs.items()},
        "differences": differences,
    }


def forward_simulation(rows: list[dict[str, Any]], state: dict[str, Any]) -> dict[str, Any]:
    """Decision (c) going forward: the rate-era trials re-played as NEW trials after the fence.

    The stored journal has no row after the live ``pareto_exclude_before_ts``. This takes
    every row from the first tasks/hour trial on, shifts its timestamp to just after the
    fence (order and spacing kept), stamps it as the new loop would, and replays it under the
    live policy with the live state's tier baselines. It shows the empty-frontier rule (b)
    refusing first, then the frontier filling and the normal frontier rule taking over.
    """
    policy = str(state.get("pareto_objective_policy") or RATE_4D_OBJECTIVE_POLICY)
    fence = float(state.get("pareto_exclude_before_ts") or 0.0)
    baselines = {
        int(t): float(q)
        for t, q in ((state.get("baseline_state") or {}).get("baselines_by_tier") or {}).items()
    }
    first = next(i for i, r in enumerate(rows) if _row_policy(r) != LEGACY_OBJECTIVE_POLICY)
    t0 = parse_journal_ts(rows[first]["timestamp"])
    shifted = list(rows[:first])
    for row in rows[first:]:
        row = dict(row)
        ts = parse_journal_ts(row["timestamp"]) or t0
        row["timestamp"] = datetime.fromtimestamp(fence + 60.0 + (ts - t0), timezone.utc).isoformat()
        details = dict(row.get("eval_details") or {})
        details["objective_policy_live"] = policy
        row["eval_details"] = details
        shifted.append(row)
    safety_log = logging.getLogger("autopilot.safety")
    previous_level = safety_log.level
    safety_log.setLevel(logging.CRITICAL)
    try:
        run = replay(
            shifted,
            path="live_c",
            start_trial_id=int(rows[first]["trial_id"]),
            initial_baselines=baselines,
            epochs={policy: fence, LEGACY_OBJECTIVE_POLICY: None},
        )
    finally:
        safety_log.setLevel(previous_level)
    stamped = stamp_clean_representatives(shifted)
    final = _view(stamped, policy, fence, "live").archive
    return {
        "objective_policy": policy,
        "fence": fence,
        "initial_baselines": {str(k): v for k, v in sorted(baselines.items())},
        "counts": run["counts"],
        "decisions": run["decisions"],
        "final_frontier_sizes": {str(t): len(final.frontier(t)) for t in (1, 2, 3)},
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
            forward = forward_simulation(fixture["rows"], state)
            fixture["golden"]["forward_simulation"] = {
                **{k: v for k, v in forward.items() if k != "decisions"},
                "decisions": [compact(d) for d in forward["decisions"]],
            }
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
